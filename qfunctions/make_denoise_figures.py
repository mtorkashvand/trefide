# import argparse
import cv2
import logging as l
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# import os

# import utils
from trefide.qfunctions.utils import print_begin_end
# from utils import print_begin_end

def correlation_image(Y):
    """
    correlation image calculation, based on paninski code
    TODO - need to understand this better
    """
    # Transpose Y to bring timedim in front
    Y = Y.transpose([2,0,1])

    # Normalize and center Y
    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    # Construct correlation kernel
    kernel = np.ones((3, 3), dtype='float32')
    kernel[1, 1] = 0

    Yconv = Y.copy()
    for idx, img in enumerate(Yconv):
        Yconv[idx] = cv2.filter2D(img, -1, kernel, borderType=0)
    
    MASK = cv2.filter2D(np.ones(Y.shape[1:], dtype='float32'), -1, kernel, borderType=0)

    corr_img = np.mean(Yconv * Y, axis=0) / MASK
    return corr_img

@print_begin_end
def make_correlation_images(normalized_movie, denoised_movie, filename):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(36,12))

    ## Make correlation images
    # Original movie
    raw_corr_img = correlation_image(normalized_movie)
    # Denoised movie
    # NOTE - Ian was adding a tiny gaussian noise to make the structure more easily visible
    # denoised_movie += np.random.randn(np.prod(normalized_movie.shape)).reshape(normalized_movie.shape)*0.01
    denoised_corr_img = correlation_image(denoised_movie)
    # Residual movie
    residual_movie = normalized_movie - denoised_movie
    residual_corr_img = correlation_image(residual_movie)

    ## Get shared scale
    global_min = min([raw_corr_img.min(), denoised_corr_img.min(), residual_corr_img.min()])
    global_max = max([raw_corr_img.max(), denoised_corr_img.max(), residual_corr_img.max()])

    ## Display
    # Original movie
    this_min = raw_corr_img.min()
    this_max = raw_corr_img.max()
    l.debug(f'Original correlation image: (range [{this_min:.1e}, {this_max:.1e}])')
    img = ax[0].imshow(raw_corr_img, vmin=global_min, vmax=global_max, cmap='viridis')
    ax[0].set_title(f'Raw Corr Image (range [{this_min:.1e}, {this_max:.1e}])')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, ax=ax[0], cax=cax)

    # Denoised movie
    this_min = denoised_corr_img.min()
    this_max = denoised_corr_img.max()
    l.debug(f'Denoised correlation image: (range [{this_min:.1e}, {this_max:.1e}])')
    img = ax[1].imshow(denoised_corr_img, vmin=global_min, vmax=global_max, cmap='viridis')
    ax[1].set_title(f'Denoised Corr Image (range [{this_min:.1e}, {this_max:.1e}])')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, ax=ax[1], cax=cax)

    # Residual movie
    this_min = residual_corr_img.min()
    this_max = residual_corr_img.max()
    l.debug(f'Residual correlation image: (range [{this_min:.1e}, {this_max:.1e}])')
    img = ax[2].imshow(residual_corr_img, vmin=global_min, vmax=global_max, cmap='viridis')
    ax[2].set_title(f'Residual Corr Image (range [{this_min:.1e}, {this_max:.1e}])')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, ax=ax[2], cax=cax)
    
    fig.suptitle(f'Correlation images, shared scale (vmin={global_min:.1e}, vmax={global_max:.1e})')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

@print_begin_end
def make_detrending_pixel_traces(raw_mov_trace, frames_clipped_trace, detrended_trace, residual_trace, stim, filename):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(16, 9))
    ax[0].plot(raw_mov_trace)
    ax[0].set(xlabel='time', ylabel='photon counts', title='raw_movie_avg_trace')

    ax[1].plot(frames_clipped_trace)
    ax[1].set(xlabel='time', ylabel='photon counts', title='frames_clipped_movie_avg_trace')

    ax[2].plot(detrended_trace)
    ax[2].set(xlabel='time', ylabel='diff of photon counts', title='detrended_movie_avg_trace')
    ax[2].axhline(linewidth=0.5, color='r')

    ax[3].plot(residual_trace)
    ax[3].set(xlabel='time', ylabel='photon counts', title='residual_avg_trace')

    ax[4].plot(stim)
    ax[4].set(xlabel='time', ylabel='protocol value', title='stim')

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

@print_begin_end
def make_denoising_pixel_traces(detrended_trace, denoised_trace, residual_trace, stim, filename):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 9))
    ax[0].plot(detrended_trace)
    ax[0].set(xlabel='time', ylabel='diff of photon counts', title='detrended_movie_avg_trace')
    ax[0].axhline(linewidth=0.5, color='r')

    ax[1].plot(denoised_trace)
    ax[1].set(xlabel='time', ylabel='diff of photon counts', title='denoised_movie_avg_trace')
    ax[1].axhline(linewidth=0.5, color='r')

    ax[2].plot(residual_trace)
    ax[2].set(xlabel='time', ylabel='diff of photon counts', title='residual_avg_trace')
    ax[2].axhline(linewidth=0.5, color='r')

    ax[3].plot(stim)
    ax[3].set(xlabel='time', ylabel='protocol value', title='stim')

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

@print_begin_end
def make_preprocess_avg_images(frames_clipped_img, detrended_img, residual_img, filename): 
    """
    TODO - are these useful? Should they be on a shared scale?
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,9))
    #vmin = min([frames_clipped_img.min(), residual_img.min(), detrended_img.min()])
    #vmax = max([frames_clipped_img.max(), residual_img.max(), detrended_img.max()])
    ax[0].imshow(frames_clipped_img) #, vmin=vmin, vmax=vmax)
    ax[0].set(title='raw movie, frames clipped')

    ax[1].imshow(detrended_img) #, vmin=vmin, vmax=vmax)
    ax[1].set(title='detrended movie')

    ax[2].imshow(residual_img) #, vmin=vmin, vmax=vmax)
    ax[2].set(title='residual from detrending')
    fig.suptitle('Average images, NOT shared scale')
    plt.tight_layout()
    plt.savefig(filename)

def make_denoising_avg_images(normalized_img, denoised_img, residual_img, filename):
    """
    TODO - are these useful? Should they be on a shared scale? Should they be combined with the images from preprocessing?
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,9))

    #vmin = min([normalized_img.min(), denoised_img.min(), residual_img.min()])
    #vmax = max([normalized_img.max(), denoised_img.max(), residual_img.max()])
    ax[0].imshow(normalized_img) #, vmin=vmin, vmax=vmax)
    ax[0].set(title='avg frame from normalized movie, before denoising')

    ax[1].imshow(denoised_img) #, vmin=vmin, vmax=vmax)
    ax[1].set(title='avg frame from movie after denoising')

    ax[2].imshow(residual_img) #, vmin=vmin, vmax=vmax)
    ax[2].set(title='avg frame from residual of denoising')
    fig.suptitle('Average images, NOT shared scale')
    plt.tight_layout()
    plt.savefig(filename)

def make_pixel_histograms(raw_movie, frames_clipped_movie, residual_movie, detrended_movie, filename):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 9))
    ax[0].hist(raw_movie.reshape(-1), bins='fd')
    ax[0].set(xlabel='value', ylabel='count', title=f'raw_movie (range: {raw_movie.min()}, {raw_movie.max()})')

    ax[1].hist(frames_clipped_movie.reshape(-1), bins='fd')
    ax[1].set(xlabel='value', ylabel='count', title=f'frames_clipped_movie (range: {frames_clipped_movie.min()}, {frames_clipped_movie.max()})')

    ax[2].hist(residual_movie.reshape(-1), bins='fd')
    ax[2].set(xlabel='value', ylabel='count', title=f'residual_movie (range: {residual_movie.min()}, {residual_movie.max()})')

    ax[3].hist(detrended_movie.reshape(-1), bins='fd')
    ax[3].set(xlabel='value', ylabel='count', title=f'detrended_movie (range: {detrended_movie.min()}, {detrended_movie.max()})')

    plt.tight_layout() # does not work with suptitle
    plt.savefig(filename)
    plt.clf()
