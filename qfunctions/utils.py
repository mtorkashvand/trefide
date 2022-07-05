from functools import wraps
import h5py
import inspect
import logging as l
import numpy as np
import os
import tifffile

def load_movie_matlab(infile):
    l.info("loading movie from matlab...")
    indir = os.path.join(os.sep, 'input') # Hard-coded input directory
    infile = os.path.join(indir, infile)
    with h5py.File(infile, 'r') as f:
        movie = np.ascontiguousarray(f['movie3D'].value.transpose([2,1,0])).astype('double')
        print_mat_info(movie=movie)
        l.info("done loading movie")
        return movie

def load_movie_stim(infile, tiny_mode=False):
    l.info("loading movie and stimulus...")
    indir = os.path.join(os.sep, 'input') # Hard-coded input directory
    infile = os.path.join(indir, infile)
    with h5py.File(infile, 'r') as f:
        movie = np.ascontiguousarray(f['inputData'].value.transpose([2,1,0])).astype('double')
        stim = np.ascontiguousarray(f['stim']).squeeze().astype('double')
        print_mat_info(movie=movie, stim=stim)
        l.info("done loading movie")
        if tiny_mode:
            l.warn("TINY MODE - USING ONLY 1000 FRAMES OF MOVIE")
            return movie[:,:,2000:3000], stim[2000:3000]
        else:
            return movie, stim

def print_begin_end(func):
    @wraps(func)
    def print_begin_end_wrapper(*args, **kwargs):
        l.info(f'begin {func.__name__}...')
        result = func(*args, **kwargs)
        l.info(f'end {func.__name__}')
        return result
    return print_begin_end_wrapper

def dump_args(func):
    """Decorator to print arguments to a function call
    """
    @wraps(func)
    def dump_args_wrapper(*args, **kwargs):
        binding = inspect.signature(func).bind(*args, **kwargs)

        bound_arguments = binding.arguments
        bound_kwargs = binding.kwargs
        func_args_str = '\n'
        func_args_str += custom_dict_tostring(bound_arguments)
        func_args_str += custom_dict_tostring(bound_kwargs)
        l.debug(f'args to {func.__qualname__} ( {func_args_str})')
        return func(*args, **kwargs)
    return dump_args_wrapper

@dump_args
def save(outdir, stage_name, **kwargs):
    """
    Checkpoint function to be invoked at each important stage of the pipeline.
    The special stage_name "main_results" should be used to store the main final outputs.

    Args:
        outfile: hdf5 file handle
        stage_name: string to be used as prefix for each of the fields created

    Kwargs:
        name/value pairs to store
    """
    for key, value in kwargs.items():
        dest = os.path.join(outdir, stage_name) 
        os.makedirs(dest, exist_ok=True)
        with h5py.File(os.path.join(dest, f'{key}.h5'), 'a') as f:
            if key in f:
                l.warn(f"The intermediate item {key} has already been saved for stage {stage_name}. Overwriting...")
                del f[key]
                f[key] = value
            else:
                l.info(f"Saving {key} to {dest}")
                f[key] = value
    l.info("done saving")

@dump_args
def save_tif(outdir, stage_name, **kwargs):
    """
    Downcast a set of movie matrices to uint_N, and save as a single TIFF stack video.
    Expects a list of key=val movies.
    Note that tifffile's imsave expects time dimension to be first, so we tranpose here.
    Filename labels the movies from top to bottom.
    """
    d1s = []
    d2s = []
    d3s = []
    keys = []
    values = []
    for key, value in kwargs.items():
        keys.append(key)
        values.append(value)
        x, y, z = value.shape
        d1s.append(x)
        d2s.append(y)
        d3s.append(z)

    l.info(f'Making TIFF stack for: {keys}...')

    height = d1s[0]
    width = d2s[0]
    length = d3s[0]
    # Check all the first dimensions match, second dimensions match, and third dimensions match
    assert ([ height ] * len(d1s)) == d1s, "first dimensions do not match"
    assert ([ width ] * len(d2s)) == d2s, "second dimensions do not match"
    assert ([ length ] * len(d3s)) == d3s, "third dimensions do not match"

    N=8 # bit range to use

    border = np.ones((2, width, length), dtype=f'uint{N}') * 2**N # white border

    min_val = min([x.min() for x in values])
    max_val = max([x.max() for x in values])

    # Intersperse the values with white pixel borders.
    interspersed_values = []
    for i in range(len(values)-1):
        interspersed_values.append(to_uint_N_set_range(M=values[i], N=N, min_val=min_val, max_val=max_val))
        interspersed_values.append(border.copy())
    interspersed_values.append(to_uint_N_set_range(M=values[-1], N=N, min_val=min_val, max_val=max_val))

    # Stack the movies and borders, and transpose to put timedim first
    stack = np.transpose(np.vstack(interspersed_values), [2,0,1])

    filename = '.'.join(keys)
    filename += '.tif'
    os.makedirs(os.path.join(outdir, stage_name), exist_ok=True)
    tifffile.imsave(os.path.join(outdir, stage_name, filename), stack)

def setup_logging(filename, outdir):
    # Setup logging to file and STDERR
    logger = l.getLogger('') # only 1 logger, so no need for name
    run_level='foo'
    if run_level == 'prod':
        logger.setLevel(l.INFO)
    else: # 'dev' or 'tiny'
        logger.setLevel(l.DEBUG)
    # create file handler
    fh = l.FileHandler(os.path.join(outdir, filename))
    fh.setLevel(l.DEBUG)
    # create console handler
    ch = l.StreamHandler()
    ch.setLevel(l.DEBUG)
    # create formatter and add it to the handlers
    formatter = l.Formatter(fmt='%(relativeCreated)d - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.handlers=[]
    logger.addHandler(fh)
    logger.addHandler(ch)

def custom_dict_tostring(dictionary):
    ret_str = ''
    for key,value in dictionary.items():
        if type(value) is np.ndarray:
            #ret_str += f'\t{key} has shape {value.shape}\n'
            ret_str += f'_print_mat_info on {key}:\n'
            ret_str += _print_mat_info(arg=value)
        else:
            if len(value.__str__()) > 1000:
                ret_str += f'\t{key} = <more than 1K characters long>\n'
            else:
                ret_str += f'\t{key} = {value}\n'
    return ret_str

def _print_mat_info(**kwargs):
    ret_str = ""
    for key, value in kwargs.items():
        ret_str += f"{key}.shape: {value.shape}\n"
        ret_str += f"{key}.flags: {value.flags}\n"
        ret_str += f"{key}.dtype: {value.dtype}\n"
        ret_str += f"{key}.min: {value.min()}\n"
        ret_str += f"{key}.max: {value.max()}\n"
    return ret_str

def print_mat_info(**kwargs):
    """
    Convenience function. Using Kwargs for easier printing.
    Example: 
        a = np.array([1,2,3])
        b = np.array([4,5,6])
        print_mat_info(a=a, btranspose=b.T) # can't do "b.T=b.T"...
    """
    l.debug(_print_mat_info(**kwargs))

#@record_time
@dump_args
def to_uint_N(M, N, lo_prctile=0.0, hi_prctile=100.0):
    """
    Convert the matrix M to the specified uint type, preserving dynamic range.
    
    Args:
        M: the matrix to convert
        N: the number of bits in the desired uint type, e.g. 8 or 16
        lo_prctile: the minimum value that should be clipped to 0
        hi_prctile: the maximum value that should be clipped to <max_val>

    Returns:
        M_downcast: the downcast matrix
    """
    tol = 1E-6
    use_min = abs(lo_prctile - 0.0) < tol 
    use_max = abs(hi_prctile - 100.0) < tol
    l.debug(f'to_uint_N, use_min: {use_min}, use_max: {use_max}')
    if use_min and use_max:
        lo_q = M.min()
        hi_q = M.max()
    elif use_min:
        lo_q = M.min()
        hi_q = np.percentile(M, hi_prctile)
    elif use_max:
        lo_q = np.percentile(M, lo_prctile)
        hi_q = M.max()
    else:
        lo_q, hi_q = np.percentile(M, [lo_prctile, hi_prctile])
    return to_uint_N_set_range(M=M, N=N, min_val=lo_q, max_val=hi_q)

def to_uint_N_set_range(M, N, min_val, max_val):
    """
    Convert to uint matrix using a linear mapping from 'min_val' to 0 and 'max_val' to the maximum value of the specified uint range
    This should be used to, as opposed to a naive uint conversion, to preserve a comparable scale of values when comparing across matrices
    """
    # TODO - this consumes a lot of memory - why?

    # Avoid sorting M twice; np.percentile can be called with 2 target values
    if np.floor(np.log2(N)) != np.ceil(np.log2(N)):
        raise TypeError(f"{N} is not a power of 2, in to_uint_N")

    # Note: 1.0000000000000001 == 1 evaluates to true
    return (((M - min_val) / (max_val - min_val)) * (2**N - 1)).astype(f'uint{N}')



def from_uint_N(M, N, lo, hi, dtype):
    """
    Take the matrix M, and map it linearly to a range of [lo, hi] of datatype dtype
    """
    if dtype not in ['float32', 'float64']:
        raise ValueError('Can only convert to float32 or float64')

    # check if M is ndarray
    M = M.astype(dtype)
    M /= 2**N - 1
    M *= hi
    M += lo
    return M

def clip_range(M, N, lo_prctile=0.01, hi_prctile=0.99):
    """
    Take a matrix of double or single, cast to uintN and clip the range, and then cast back
    """
    min_before = M.min()
    max_before = M.max()
    dtype = M.dtype
    M = to_uint_N(M=M, N=N, lo_prctile=lo_prctile, hi_prctile=hi_prctile)
    return from_uint_N(M=M, N=N, lo=min_before, hi=max_before, dtype=dtype)

def rescale(M, new_min, new_max):
    old_min = M.min()
    old_max = M.max()
    return ((M - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def rescale_perc(M, lo_prctile=0.01, hi_prctile=0.99):
    new_min, new_max = np.percentile(M, [lo_prctile, hi_prctile])
    return rescale(M, new_min=new_min, new_max=new_max)

@dump_args
def reconstruct_batch_decompose(U, V, K, block_indices):
    """
    U - spatial components
    V - temporal components
    K - block ranks

    reconstruction based on:
    cpdef double[:,:,::1] batch_recompose(double[:, :, :, :] U,
                                      double[:,:,::1] V,
                                      size_t[::1] K,
                                      size_t[:,:] indices):

    Example inputs:
    U.shape: (4, 40, 40, 50)
    V.shape: (4, 50, 630)
    """ 
    # Get Block Size Info From Spatial
    num_blocks = U.shape[0]
    num_components = np.sum(K).astype('int')
    bheight = U.shape[1]
    bwidth = U.shape[2]
    t = V.shape[2]

    block_indices = block_indices.astype('int')
    K = K.astype('int')

    # Get Mvie Size Infro From Indices
    nbi = int(np.max(block_indices[:,0]) + 1)
    nbj = int(np.max(block_indices[:,1]) + 1)
    d1 = int(nbi * bheight)
    d2 = int(nbj * bwidth)

    # Allocate Space For reconstructed Movies
    U_final = np.zeros((d1, d2, num_components), dtype='double')
    V_final = np.zeros((num_components, t), dtype='double')

    # Loop Over Blocks
    V_cursor = 0
    U_cursor = 0
    for bdx in range(nbi*nbj):
        idx = block_indices[bdx,0] * bheight
        jdx = block_indices[bdx,1] * bwidth
        U_final[idx:idx+bheight, jdx:jdx+bwidth,U_cursor:U_cursor+K[bdx]] = U[bdx, :, :, :K[bdx]]
        V_final[V_cursor:V_cursor+K[bdx],:] = V[bdx,:K[bdx], :]
        U_cursor += K[bdx]
        V_cursor += K[bdx]
    
    return U_final, V_final
