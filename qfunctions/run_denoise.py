"""
Must first build trefide docker image, then funimag image.
This is the fastest way to get local changes incorporated.

Expects a .mat v7.3 formatted input file.
- must have a field "inputData", containing a (width x height x time_duration) movie (should be uint16)
- must have a field "stim", containing a (time_duration) length array of the stimulation protocol

See example usage in run_pipeline.sh, including metric collection from `docker stats`. 
"""

#!/usr/bin/env python
import argparse
import logging as l
import numpy as np
import os
import scipy.sparse
import time
from trefide.utils import psd_noise_estimate
from trefide.pmd import overlapping_batch_decompose
#from trefide.pmd import batch_decompose
from trefide.pmd import determine_thresholds
from trefide.reformat import overlapping_component_reformat
from functools import wraps


from trefide.qfunctions.make_denoise_figures import make_detrending_pixel_traces, make_preprocess_avg_images, make_pixel_histograms, make_correlation_images, make_denoising_avg_images
from trefide.qfunctions import utils
# import utils
from trefide.qfunctions.preprocess import simple_detrend, fix_tsub_movie_length
from trefide.qfunctions.utils import load_movie_matlab, load_movie_stim, print_begin_end, dump_args
# load_movie_stim is written expecting a previous step producing a single mat file with 'inputData' and 'stim' fields
# load_movie_matlab is written for when this code gets from a Segment_*.m file inside qstate pipeline

# TODO - what's the right way to put this in separate utils file, and still
# have it accept an open filehandle arg?  Can make another layer of nesting to
# accept an arg, but that filehandle needs to be opened here (for the correct
# path), and yet can't be made globally available to the function defn when
# that lives in another file
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
    p.add_argument('--save-diagnostics', action='store_true', help='set this flag to save TIFF stacks, correlation images, and other diagnostic plots')
    p.add_argument('--tiny-mode', action='store_true', help='set this flag to process only a tiny slice of the input')
    p.add_argument('--overlapping-denoise', action='store_true', help='set this flag to do the 4x overcomplete version of blockwise denoising')
    p.add_argument('--median-window', type=int, default=41, required=True, help='size of window for rolling median detrending')
    p.add_argument('--skip-preprocess', action='store_true', help='set this flag to skip preprocessing')
    p.add_argument('--input-matlab', action='store_true', help='set this flag to load inputs from matlab script')

    ######################
    ## PMD parameters
    ######################
    p.add_argument('--infile', required=True, help='Input movie *.mat. Must contain fields "inputData" and "stim"')
    p.add_argument('--consec-failures', type=int, required=True, help='consecutive failures. More means more conservative, retaining a higher rank matrix')
    p.add_argument('--max-components', type=int, required=True, help='')
    p.add_argument('--max-iters-main', type=int, required=True, help='')
    p.add_argument('--max-iters-init', type=int, required=True, help='')
    p.add_argument('--block-width', type=int, required=True, help='Note - FOV width must be divisble by block width')
    p.add_argument('--block-height', type=int, required=True, help='Note - FOV height must be divisible by block height')
    p.add_argument('--d-sub', type=int, required=True, help='')
    p.add_argument('--t-sub', type=int, required=True, help='Note - temporal dimension must be divisible by t_sub')

    args = p.parse_args()
    if args.save_diagnostics and args.input_matlab:
        raise TypeError('When loading from MATLAB during pipeline, we will only have the movie and not the stim. This makes diagnostic plots impossible.')

    return args

@print_begin_end
def preprocess(movie, stim):
    """
    Rolling median detrending, in addition to some frame clipping around the
    discontinuities in the stim protocol (where firing is empirically highly
    correlated across all cells).
    """
    detrended_movie, stim, del_idx = simple_detrend(movie=movie,
            stim=stim,
            t_sub=args.t_sub,
            median_window=args.median_window)

    utils.print_mat_info(detrended_movie=detrended_movie)

    if args.save_diagnostics:
        frames_clipped_movie = np.delete(movie, del_idx, axis=2) # the raw movie, with the same frames clipped as everyone else
        residual_movie = (frames_clipped_movie - detrended_movie)
        
        utils.print_mat_info(frames_clipped_movie=frames_clipped_movie, residual_movie=residual_movie)

        # Save pixel traces
        make_detrending_pixel_traces(
                raw_mov_trace        = np.mean(movie,                axis=(0,1)),
                frames_clipped_trace = np.mean(frames_clipped_movie, axis=(0,1)),
                detrended_trace      = np.mean(detrended_movie,      axis=(0,1)),
                residual_trace       = np.mean(residual_movie,       axis=(0,1)),
                stim=stim,
                filename=os.path.join(args.diagnostics_dir, 'detrending.pixel_traces.png'))

        # Save average images
        make_preprocess_avg_images(
                frames_clipped_img = np.mean(frames_clipped_movie, axis=2),
                detrended_img      = np.mean(detrended_movie,      axis=2),
                residual_img       = np.mean(residual_movie,       axis=2),
                filename           = os.path.join(args.diagnostics_dir, 'detrending.avg_images.png'))

        # Save pixel histograms
        make_pixel_histograms(
                raw_movie = movie,
                frames_clipped_movie = frames_clipped_movie,
                residual_movie = residual_movie,
                detrended_movie = detrended_movie,
                filename = os.path.join(args.diagnostics_dir, 'detrending.pixel_histograms.png'))

    return detrended_movie

@print_begin_end
def normalize(movie):
    height, width, num_frames = movie.shape
    movie = np.ascontiguousarray(movie.reshape(height*width, num_frames))
    estimated_noise_stddev = np.sqrt(np.asarray(psd_noise_estimate(movie)))
    utils.print_mat_info(estimated_noise_stddev=estimated_noise_stddev)
    movie = (movie / estimated_noise_stddev[:, None]).reshape(height, width, num_frames)
    utils.print_mat_info(movie=movie)
    return movie

@print_begin_end
def denoise(movie):
    l.info("determine_thresholds...")
    fov_height, fov_width, num_frames = movie.shape
    tol = 5e-3
    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, num_frames), 
            (args.block_height, args.block_width),
            args.consec_failures, args.max_iters_main, 
            args.max_iters_init, tol, 
            args.d_sub, args.t_sub, 5, True)
    l.info("done determine_thresholds")

    l.info("overlapping_batch_decompose...")
    # NOTE - only allowing overlapping batch decompose. Single-tiling leaves correlation artifacts, which will
    # probably hurt demixing
    # TODO - This could be investigated further; if single tiling is sufficient for good demixing, then it will be faster and produce smaller output!! Potential big win here (but low odds of success)
    spatial_components, temporal_components, block_ranks, block_indices, block_weights = overlapping_batch_decompose(
        fov_height, fov_width, num_frames,
        movie, args.block_height, args.block_width,
        spatial_thresh, temporal_thresh,
        args.max_components, args.consec_failures,
        args.max_iters_main, args.max_iters_init, tol,
        d_sub=args.d_sub, t_sub=args.t_sub)
    l.info("done overlapping_batch_decompose")

    l.info("overlapping_component_reformat...")
    U, V = overlapping_component_reformat(fov_height, fov_width, num_frames, 
            args.block_height, args.block_width,
            spatial_components,
            temporal_components,
            block_ranks,
            block_indices,
            block_weights)
    l.info("end overlapping_component_reformat")

    if args.save_diagnostics:
        l.info("save tiff stack of denoising...")
        denoised_movie = np.matmul(U, V)
        utils.save_tif(outdir=outdir, 
                stage_name='main_results', 
                normalized_movie=movie,
                denoised_movie=denoised_movie,
                residual=(movie - denoised_movie))

        # Make correlation images
        make_correlation_images(normalized_movie=movie, 
                denoised_movie=denoised_movie, 
                filename=os.path.join(args.diagnostics_dir, 'denoising_correlation_images.png'))

        # Save average images
        make_denoising_avg_images(normalized_img=np.mean(movie, axis=2),
                denoised_img=np.mean(denoised_movie, axis=2),
                residual_img=np.mean((movie - denoised_movie), axis=2),
                filename=os.path.join(args.diagnostics_dir, 'denoising_avg_images.png'))

    return U, V

@record_time
@dump_args
@print_begin_end
def run_denoise_pipeline(basedir, outdir):
    utils.setup_logging(filename='LOG.txt', outdir=outdir)

    os.makedirs(os.path.join(outdir, 'main_results'), exist_ok=True)
    if args.save_diagnostics:
        args.diagnostics_dir = os.path.join(outdir,'diagnostics')
        os.makedirs(args.diagnostics_dir, exist_ok=True)

    if args.input_matlab:
        movie = load_movie_matlab(infile=args.infile)
    else:
        movie, stim = load_movie_stim(infile=args.infile, tiny_mode=args.tiny_mode)
    fov_height, fov_width, original_length = movie.shape
    # For sifting through results more easily, save a flag file with the movie dimensions
    open(os.path.join(basedir, f'MOVIE_SHAPE.{fov_height}.{fov_width}.{original_length}.FLAG'), 'w').close()

    if not args.skip_preprocess:
        movie = preprocess(movie, stim)
    else:
        # Makes the movie length divisible by t_sub parameter
        movie, _ = fix_tsub_movie_length(movie, args.t_sub)
    movie = normalize(movie)
    U, V = denoise(movie)

    # Save a flag file to easily see the rank of U and V
    open(os.path.join(outdir, f'DENOISE_RANK.{U.shape[-1]}.FLAG'), 'w').close()

    utils.print_mat_info(U=U, V=V)
    np.savez_compressed(os.path.join(outdir, 'main_results', 'V.npz'), V=V) # TODO - try float16
    # NOTE - after experimenting on only 1 file, the choice of sparse format does not appear to affect the saved size
    # notice that DOK and LIL formats do not allow saving (checked CSC, CSR, COO, BSR, sizes all 13M)
    scipy.sparse.save_npz(os.path.join(outdir, 'main_results', 'U.csc.npz'), 
            scipy.sparse.csc_matrix(U.reshape(fov_height*fov_width, -1)), 
            compressed=True)

if __name__ == '__main__':
    args = parse_args()

    basedir = os.path.join(os.sep, 'output')
    outdir = os.path.join(basedir, 'denoising')
    os.makedirs(outdir, exist_ok=True)

    # Decorate imported functions for logging purposes
    simple_detrend                 = record_time(dump_args(simple_detrend))
    load_movie_stim                = record_time(dump_args(load_movie_stim))

    # The following methods can't dump args (no signature)
    overlapping_batch_decompose    = record_time(overlapping_batch_decompose)
    overlapping_component_reformat = record_time(overlapping_component_reformat)
    determine_thresholds           = record_time(determine_thresholds)
    psd_noise_estimate             = record_time(psd_noise_estimate)
    #batch_decompose                = record_time(batch_decompose)

    # NOTE - line buffering on timefile
    # Timing file is handled here so that run_denoise_pipeline can also be timed
    # with the same decorator
    with open(os.path.join(outdir, 'TIMING.txt'), 'w', buffering=1) as timefile:
        try:
            run_denoise_pipeline(basedir=basedir, outdir=outdir)
        except Exception as e:
            l.exception(str(e))
            raise e
