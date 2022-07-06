"""
Must first build trefide docker image, then funimag image.
This is the fastest way to get local changes incorporated.

Expects a .mat v7.3 formatted input file.
- must have a field "inputData", containing a (width x height x time_duration) movie (should be uint16)
- must have a field "stim", containing a (time_duration) length array of the stimulation protocol

See example usage in run_pipeline.sh, including metric collection from `docker stats`. 
"""
import argparse
from functools import wraps
import h5py
import logging as l
import numpy as np
import scipy.sparse
import os
import time

import funimag.superpixel_analysis as sup
from qfunctions.utils import setup_logging, print_mat_info, load_movie_stim, print_begin_end, dump_args

from qfunctions.make_demix_figures import make_cell_masks_and_traces, make_superpixel_image

# TODO - what's the right way to put this in separate utils file, and still have it accept an open filehandle arg?
# Can make another layer of nesting to accept an arg, but that filehandle needs to be opened here (for the correct path),
# and yet can't be made globally available to the function defn when that lives in another file
def record_time(func):
    """
    Decorator to save the runtime of the invoked function into timefile.
    Requires an open handle to a file for output (this makes the results more useful)
    """
    @wraps(func)
    def record_time_wrapper(*args, **kwargs):
        global timefile
        fname = func.__name__
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        timefile.write(f"{fname}: {t1-t0:.3f}s\n")
        timefile.flush() # will be writing very few lines to this file, but want them in realtime
        return result
    return record_time_wrapper

def str2bool(v):
    """
    helper function for argparse with boolean inputs
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    p = argparse.ArgumentParser()
    ######################
    # High-level controls
    ######################
    p.add_argument('--tiny-mode', action='store_true', help='set this flag to process only a small slice of the input')
    p.add_argument('--save-diagnostics', action='store_true', help='set this flag to save diagnostic plots')
    p.add_argument('--infile', required=True, help='Input movie *.mat. Must contain fields "inputData" and "stim"')
    p.add_argument('--input-matlab', action='store_true', help='set this flag to load the movie after matlab')
    p.add_argument('--output-hdf5', action='store_true', help='set this flag to save final outputs as hdf5')

    ######################
    ## Demixing parameters
    ######################
    # int list args
    p.add_argument('--length-cut', nargs='+', type=int, required=True, help='')
    p.add_argument('--th', nargs='+', type=int, required=True, help='')
    p.add_argument('--patch-size', nargs='+', type=int, required=True, help='')

    # float list args
    p.add_argument('--cut-off-point', nargs='+', type=float, required=True, help='')
    p.add_argument('--residual-cut', nargs='+', type=float, required=True, help='')
    
    # float args
    p.add_argument('--corr-th-fix', type=float, required=True, help='')
    p.add_argument('--max-allow-neuron-size', type=float, required=True, help='')
    p.add_argument('--merge-corr-thr', type=float, required=True, help='')
    p.add_argument('--merge-overlap-thr', type=float, required=True, help='')
    
    # int args
    p.add_argument('--pass-num', type=int, required=True, help='')
    p.add_argument('--num-plane', type=int, required=True, help='')
    p.add_argument('--fudge-factor', type=int, required=True, help='')
    p.add_argument('--max-iter', type=int, required=True, help='')
    p.add_argument('--max-iter-fin', type=int, required=True, help='')
    p.add_argument('--update-after', type=int, required=True, help='')

    # bool args
    p.add_argument('--plot-en', type=str2bool, required=True, help='')
    p.add_argument('--TF', type=str2bool, required=True, help='')
    p.add_argument('--text', type=str2bool, required=True, help='')
    p.add_argument('--bg', type=str2bool, required=True, help='')

    args = p.parse_args()
    return args

@record_time
@dump_args
@print_begin_end
def run_demix(outdir, denoising_dir): 
    #NOTE - this could also be run directly without saving U and V; just provide additional args (..., U=None, V=None):
    # and load only if the matrices are not None
    setup_logging(filename='LOG.txt', outdir=outdir)

    # TODO - hardcoded filepaths
    U_file = os.path.join(denoising_dir, 'main_results', 'U.csc.npz')
    V_file = os.path.join(denoising_dir, 'main_results', 'V.npz')

    # TODO - hardcoded FOV dimensions for U. These could be read from the MOVIE_SHAPE flag file, stored in this npz filename, or stored inside the npz file
    l.debug(f'U_file: {U_file}')
    U = scipy.sparse.load_npz(U_file).toarray().reshape(80, 800, -1)
    l.debug(f'V_file: {V_file}')
    V = np.load(V_file)['V']

    print_mat_info(U=U)
    print_mat_info(V=V)

    l.info("STAGE: reconstruct movie from U and V...")
    Yd = np.matmul(U, V)
    l.info("done reconstruct movie from U and V")
    print_mat_info(Yd=Yd)

    U = U.reshape(-1, U.shape[2], order='F')
    l.debug("After reshaping:")
    print_mat_info(U=U, V_transpose=V.T)
    rlt = sup.demix(Yd=Yd, U=U, V=V.T,
                cut_off_point=args.cut_off_point,
                length_cut=args.length_cut,
                th=args.th,
                residual_cut=args.residual_cut,
                patch_size=args.patch_size,
                corr_th_fix=args.corr_th_fix,
                max_allow_neuron_size=args.max_allow_neuron_size,
                merge_corr_thr=args.merge_corr_thr,
                merge_overlap_thr=args.merge_overlap_thr,
                pass_num=args.pass_num,
                num_plane=args.num_plane,
                fudge_factor=args.fudge_factor,
                max_iter=args.max_iter,
                max_iter_fin=args.max_iter_fin,
                update_after=args.update_after,
                plot_en=args.plot_en,
                TF=args.TF,
                text=args.text,
                bg=args.bg)
    A=rlt['fin_rlt']['a']
    B=rlt['fin_rlt']['b']
    C=rlt['fin_rlt']['c']

    print_mat_info(A=A, B=B, C=C)

    os.makedirs(os.path.join(outdir, 'main_results'), exist_ok=True)
    raw_movie = None
    if args.output_hdf5:
        #if args.input_matlab:
        #    raw_movie = load_movie_matlab(infile=args.infile)
        #else:
        #    raw_movie, _ = load_movie_stim(infile=args.infile, tiny_mode=args.tiny_mode)

        # Want to take a convex combination of the correct pixels' time vectors from the input movie.
        # Normalize A so that each cell's weight vector sums to 1.
        # Transpose this, and multiply with the 2D movie of shape (n_pix, n_frames).
        #width, height, n_frames = raw_movie.shape
        #n_pix = width * height
        #raw_traces = (A / np.sum(A, axis=0)).T @ raw_movie.reshape(n_pix, n_frames)
        with h5py.File(os.path.join(outdir, 'main_results', 'A.h5'), 'w') as f:
            f['A'] = A
        #with h5py.File(os.path.join(outdir, 'main_results', 'raw_traces.h5'), 'w') as f:
        #    f['raw_traces'] = raw_traces

    else:
        scipy.sparse.save_npz(os.path.join(outdir, 'main_results', 'A.csc.npz'), 
                scipy.sparse.csc_matrix(A), 
                compressed=True)
        np.savez_compressed(os.path.join(outdir, 'main_results', 'B.npz'), B=B)
        np.savez_compressed(os.path.join(outdir, 'main_results', 'C.npz'), C=C)

    if args.save_diagnostics:
        # TODO - notice that using the raw trace may be unrealistic; if we only save U and V, then here
        # we should instead be loading those matrices and reconstructing the matrix before proceding
        if raw_movie is None: # avoid duplicate loading
            raw_movie, _ = load_movie_stim(infile=args.infile, tiny_mode=args.tiny_mode)

        make_cell_masks_and_traces(A=A, C=C, raw_movie=raw_movie, figdir=os.path.join(outdir, 'figures'))

        make_superpixel_image(
            connect_mat_1   = rlt['superpixel_rlt'][0]['connect_mat_1'], 
            pure_pix        = rlt['superpixel_rlt'][0]['pure_pix'],
            brightness_rank = rlt['superpixel_rlt'][0]['brightness_rank'],
            filename = os.path.join(outdir,'figures','pure_superpixel_initialization.png'))

if __name__ == '__main__':
    args = parse_args()
    if args.tiny_mode:
        raise NotImplementedError('TODO - --tiny-mode flag ')

    outdir = os.path.join(os.sep, 'output', 'demixing')
    os.makedirs(outdir, exist_ok=True)
    denoising_dir = os.path.join(os.sep, 'output', 'denoising')

    # Already has dump_args, only needs record_time
    sup.demix = record_time(sup.demix) 

    # NOTE - line buffering on timefile
    with open(os.path.join(outdir, 'TIMING.txt'), 'w', buffering=1) as timefile:
        try:
            run_demix(outdir=outdir, denoising_dir=denoising_dir)
        except Exception as e:
            l.exception(str(e))
            raise e

