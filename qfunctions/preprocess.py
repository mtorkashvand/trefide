import numpy as np
from numpy.linalg import norm
import scipy.interpolate as interp
from scipy.sparse.linalg import eigsh
from multiprocess_median import multiprocess_median
import logging as l
logger = l.getLogger(__name__)
#from run_pipeline import print_mat_info
from qfunctions.utils import print_mat_info

def subtract_mean_projection(mov):
    """
    TODO - this might be doable via a library function, or maybe could be more efficient
    (but it's extremely fast anyway)
    """
    original_shape = mov.shape
    mov = mov.reshape(-1, mov.shape[-1])

    # We make a rank-1 projection matrix for the average pixel's time vector
    mean_pixel_vector = np.mean(mov, axis=0)
    print_mat_info(mean_pixel_vector=mean_pixel_vector)
    proj_matrix = rank_one_projection(mean_pixel_vector)
    print_mat_info(proj_matrix=proj_matrix)

    # We get the projection of each pixel's time vector onto the mean pixel time vector, and subtract this from the mov
    # The result is that each pixel has a zero dot product with the mean pixel time vector
    mov = mov - np.matmul(mov, proj_matrix)
    assert np.abs(np.inner(mov[0,:], mean_pixel_vector)) < 1e-6, "failed to orthogonalize"
    # TODO - remove assert
    return mov.reshape(original_shape)

def rank_one_projection(vec):
    return np.outer(vec, vec) / (norm(vec)**2)

def subtract_top_principal_component(mov):
    """
    TODO - probably faster and simpler to do this by a rank-1 SVD on the original movie matrix?
    TODO - need to save a rank-1 matrix and add this back to U*V afterwards, not reasonable to just discard permanently
    Subtract from each pixel's time vector, its projection onto the top principal component

    mov: movie of shape (height, width, time)
    """
    original_shape = mov.shape
    mov = mov.reshape(-1, mov.shape[-1])
    mov_cent = mov - np.mean(mov, axis=0)[None,:] # subtract the mean time vector
    _, eigvec = eigsh(np.matmul(mov_cent.T, mov_cent), k=1)
    top_pc = eigvec.squeeze()
    print_mat_info(top_pc=top_pc)
    proj_matrix = rank_one_projection(top_pc)
    print_mat_info(proj_matrix=proj_matrix)
    print_mat_info(mov_cent=mov_cent)
    pc1 = np.matmul(mov_cent, proj_matrix)
    print_mat_info(pc1=pc1, mov_cent=mov_cent)
    return (mov - pc1[None, :]).reshape(original_shape)

def _subtract_mean_projection(mov):
    return mov - np.mean(mov, axis=(0,1))[None, None, :]

def fix_tsub_movie_length(movie, t_sub):
    """
    This function will only be used when accepting input from MATLAB.
    In that situation, we only have movie (no stim)
    """
    movie_length = movie.shape[2]
    del_idx = _del_idx_divisible(movie_length, np.array([]), t_sub)

    return np.delete(movie, del_idx, axis=2), del_idx

def simple_detrend(movie, stim, t_sub, median_window):
    """
    Fit a spline to explain the trends due to the stimulation protocol, and subtract this from the movie.

    Movie should already have done ChopMovieByBadFrames (remove saturated frames due to async camera mode)

    Parameters:
        movie: movie of size (height x width x time)
        stim: stimulation protocol of size (time)
        t_sub: the temporal downsampling factor that will be used later. Need to keep movie length divisible by this number.

    Returns:
        detrended_movie: movie - trend, with frames deleted
        stim: stim, with frames deleted
        del_idx: list of frames that were deleted
    """
    # TODO - methods for speeding this up: explicitly parallelize across pixels, transpose the movie so timedim is first (may or may not help)
    trend = multiprocess_median(movie=movie, window_size=median_window)
    print_mat_info(trend=trend)

    del_idx = get_del_idx(stim=stim, t_sub=t_sub)
    print_mat_info(del_idx=del_idx)

    return np.delete(movie - trend, del_idx, axis=2), np.delete(stim, del_idx), del_idx

def get_del_idx(stim, t_sub):
    """
    given the stim protocol, get indices that need to be deleted (to reduce whole-frame correlation)
    """
    # Get the location of step discontinuities in the stim protocol
    step_idx = get_steps(stim)
    print_mat_info(step_idx=step_idx)

    # Chop a few frames around each step discontinuity
    del_idx = cut_around_steps(step_idx, n_before=10, n_after=120)
    l.debug(f'del_idx: {del_idx}')

    cut_beginning = 120

    # Chop a few frames at beginning of movie
    del_idx = np.append(del_idx, np.arange(cut_beginning))

    # Chop frames from end of movie
    cut_end = 120
    del_idx = np.append(del_idx, np.arange(stim.shape[-1] - cut_end, stim.shape[-1]))

    del_idx = _del_idx_divisible(stim.shape[-1], del_idx, t_sub)
    return del_idx

def cut_around_steps(step_idx, n_before=10, n_after=100):
    """
    TODO - only cut after rising steps? needs experimentation to see what is required for achieving low rank
    returns list of extra indices to delete.
    indices at each
    """
    l.info(f'Cutting additional {n_before} frames before and {n_after} frames after each discontinuity index')
    tmp = step_idx - n_before
    l.debug('one')
    del_idx = tmp.copy()
    for i in range(n_before + n_after):
        del_idx = np.append(del_idx, tmp + i)
    print_mat_info(del_idx=del_idx)
    del_idx = np.unique(del_idx)
    print_mat_info(del_idx=del_idx)
    l.debug(f'cut_around_steps, del_idx: {del_idx}')
    return del_idx

def _del_idx_divisible(movie_length, raw_del_idx, t_sub):
    """
    Add indices to del_idx until (movie_length - len(raw_del_idx)) % t_sub == 0
    Prefer to delete frames as close to beginning as possible.
    """
    if raw_del_idx.size != 0:
        raw_del_idx = raw_del_idx[raw_del_idx <= movie_length]
    n_needed = (movie_length - len(raw_del_idx)) % t_sub
    i = 0
    n_added = 0
    while n_added != n_needed:
        if i == movie_length - 1:
            raise AssertionError(f'failed to find {n_needed} additional indices for deletion')
        if i not in raw_del_idx:
            raw_del_idx = np.append(raw_del_idx, i)
            n_added += 1
        i += 1
    return np.unique(np.sort(raw_del_idx))

# TODO - duplicated function
def print_mat_info(**kwargs):
    """
    Convenience function. Using Kwargs for easier printing.
    Example: 
        a = np.array([1,2,3])
        b = np.array([4,5,6])
        print_mat_info(a=a, btranspose=b.T) # can't do "b.T=b.T"...

    """
    for key, value in kwargs.items():
        l.debug(f"{key}.shape: {value.shape}")
        l.debug(f"{key}.flags: {value.flags}")
        l.debug(f"{key}.dtype: {value.dtype}")

def remove_redundant(arr, k):
    """
    We can have multiple knots at a single index.
    However if we have knots at consecutive indices less than k apart, this causes a problem
    ( "Nth leading minor not positive definite" - an error during matrix inversion in the normal equations )

    """
    redundant = []
    i = 1
    while i < len(arr): # use WHILE to recalculate list length each iteration
        if arr[i]-arr[i-1] < k: # stay at this position until we've deleted all neighbors closer than k units away
            arr = np.delete(arr, i)
        else:
            i += 1
    return arr

def get_steps(stim):
    return np.nonzero(np.convolve(stim > 0, np.array([1, -1])))[0].astype('int')

def get_knots(stim, step_idx, k=3, followup=20, spacing=100):
    """
    NOTE - we assume the beginning and end of a stimulation protocol step are at least k units apart.

    step_idx: array of indices where step discontinuities occur in the stim protocol
    k: degree of spline polynomial
    followup: add extra knots N frames before and after each step discontinuity - TODO tune this parameter
    spacing: the distance between the regularly scattered knots. This should be smaller than the width of a stim protocol step, so that we get knots in each portion of the trace.
    """
    l.debug(f"step_idx: {step_idx}")

    # Add knots at a regular spacing between each step index
    # We do this manually to control how many knots are placed at the step indices (which should be exactly k)
    #between_steps = np.zeros((len(step_idx) + 1, 0))
    between_steps = []

    def lin(start, stop):
        # + 2 based on Ian's code?
        tmp = np.linspace(start=start, stop=stop, num=(stop-start) / spacing + 2, endpoint=False, dtype='int')
        if len(tmp) > 0:
            # skip the start point, which we want to include exactly k times. stop point already excluded
            return tmp[1:]
        else:
            return tmp

    # For N step indices found, there will be N+1 intervals in which we want to place knots
    n_intervals = len(step_idx) + 1
    for i in range(n_intervals):
        if i == 0:
            between_steps.append(lin(0, step_idx[0]))
        elif i == n_intervals - 1:
            between_steps.append(lin(step_idx[-1], len(stim) - 1))
        else:
            between_steps.append(lin(step_idx[i-1], step_idx[i]))

    between_steps = np.concatenate(between_steps).flatten()
    l.debug(f"final between_steps: {between_steps}")

    # Add knots at: beginning, end, each step discontinuity, 'followup' frames before/after steps, and regularly spaced between steps
    # We want k+1 knots at beginning and end, k knots at each discontinuity, and 1 knot at the other locations
    knots = np.sort(np.concatenate([
                np.zeros(k + 1),
                np.repeat(step_idx, k),
                between_steps,
                step_idx + followup,
                step_idx - followup,
                np.ones(k + 1) * (len(stim) - 1)
            ])).astype('int')

    print_mat_info(knots=knots)

    l.debug(f"knots: {knots}")
    return step_idx, knots

def get_spline_trend(movie, knots, q=.03, k=3, additional_disc_idx=None):
    """
    Fit a spline along each pixel's time vector.
    We fit a spline naively once to that vector, then try to discard outlier timepoints by ignoring the top `q` percentile of residual error.
    """
    if additional_disc_idx:
        knots = np.sort(np.append(knots, np.repeat(disc_idx, order + 1)))

    #def spline_fit(y):
    #    bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
    #    return bspl(x)
    #trend = np.apply_along_axis(spline_fit, axis, data)

    x = np.arange(movie.shape[2])
    def robust_spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=k)
        resid = np.abs(bspl(x) - y)
        keep_idx = resid <= np.percentile(resid, (1 - q) * 100)
        bspl = interp.make_lsq_spline(
            x=x[keep_idx], y=y[keep_idx], t=knots, k=k)

        return bspl(x)

    trend = np.apply_along_axis(func1d=robust_spline_fit, axis=2, arr=movie)

    return trend
