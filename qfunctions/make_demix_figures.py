import logging as l
import matplotlib.pyplot as plt
import numpy as np
import os
import funimag.superpixel_analysis as sup

from trefide.qfunctions.utils import print_begin_end
# from utils import print_begin_end

@print_begin_end
def make_cell_masks_and_traces(A, C, raw_movie, figdir): 
    # A.shape: (n_pix, n_cells)
    n_rows, n_cols, n_frames = raw_movie.shape

    # For clarity
    n_cells = A.shape[1]
    n_pix = A.shape[0]

    l.debug(f'n_rows: {n_rows}')
    l.debug(f'n_cols: {n_cols}')
    l.debug(f'n_cells: {n_cells}')
    l.debug(f'n_pix: {n_pix}')
    l.debug(f'n_frames: {n_frames}')

    # Want to take a convex combination of the correct pixels' time vectors from the input movie.
    # Normalize A so that each cell's weight vector sums to 1.
    # Transpose this, and multiply with the 2D movie of shape (n_pix, n_frames).
    raw_traces = (A / np.sum(A, axis=0)).T @ raw_movie.reshape(n_pix, n_frames)
    # raw_traces.shape: (n_cells, n_frames)

    os.makedirs(os.path.join(figdir, 'masks'), exist_ok=True) 
    os.makedirs(os.path.join(figdir, 'masks_overlaid'), exist_ok=True)
    os.makedirs(os.path.join(figdir, 'traces'), exist_ok=True)
    os.makedirs(os.path.join(figdir, 'raw_traces'), exist_ok=True)

    movie_avg_img = np.mean(raw_movie, axis=2)

    for i in range(n_cells):
        cell_mask = A[:, i].reshape(n_rows, n_cols, order='F')

        # Standalone cell masks
        fig, ax = plt.subplots(figsize=(16,9))
        ax.imshow(cell_mask);
        ax.set(title=f'Cell Mask {i}')
        plt.savefig(os.path.join(figdir, 'masks', f'cell_{i}.png'))
        plt.clf()

        # Cell masks overlaid on average image
        fig, ax = plt.subplots(figsize=(16,9))
        ax.imshow(movie_avg_img, cmap='gray')
        ax.imshow(cell_mask, cmap='jet', alpha=0.5)
        plt.savefig(os.path.join(figdir, 'masks_overlaid', f'cell_{i}.png'))
        plt.clf()

        # Cell traces from C matrix
        fig, ax = plt.subplots(figsize=(16,9))
        ax.plot(C[:,i]); 
        ax.set(xlabel='time', ylabel='normalized photon counts', title=f'Cell Trace {i}')
        plt.savefig(os.path.join(figdir, 'traces', f'cell_{i}.png'))
        plt.clf()

        # Cell traces using mask on raw movie
        fig, ax = plt.subplots(figsize=(16,9))
        ax.plot(raw_traces[i, :])
        ## plotting the brightest pixel in a cell:
        #mask = A[:,i].reshape(n_rows,n_cols)
        #x, y = np.unravel_index(np.argmax(mask), mask.shape)
        #ax.plot(raw_movie[x, y, :])
        ax.set(xlabel='time', ylabel='raw photon counts', title=f'Cell Trace {i} - Using Mask Dotted into Raw Movie')
        plt.savefig(os.path.join(figdir, 'raw_traces', f'cell_{i}.png'))
        plt.clf()

@print_begin_end
def make_superpixel_image(connect_mat_1, pure_pix, brightness_rank, filename):
    sup.pure_superpixel_single_plot(connect_mat_1,
                                   pure_pix,
                                   brightness_rank,
                                   text=True,
                                   pure=True);
    plt.savefig(filename)
