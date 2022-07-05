"""
Multiprocessing rolling median calculation for a movie of shape (n_rows x n_cols x n_frames).

Key issue: avoid duplicating movie by using a shared global variable for the movie array.
"""
from multiprocessing import Pool, RawArray, cpu_count
import numpy as np
#from scipy.signal import medfilt
from scipy.ndimage import median_filter
import h5py
import time
import logging as l
import cProfile
# from bisect import bisect_left, insort

def initializer(shared_movie_chunks, shared_movie_chunks_shapes, shared_result_chunks, window_size):
    """
    initialize workers with a shared global dictionary describing a few key dimensions
    """
    var_dict['shared_movie_chunks']        = shared_movie_chunks
    var_dict['shared_movie_chunks_shapes'] = shared_movie_chunks_shapes
    var_dict['shared_result_chunks']       = shared_result_chunks
    var_dict['window_size']                = window_size

def _worker_func(idx):
    cProfile.runctx('_worker_func(idx)', globals(), locals(), sort='tottime')

def worker_func(idx):
    """
    given a shared read-only array of shape (n_rows x n_cols),
    perform: apply_along_axis(matrix[idx*chunk_size:(idx+1)*chunk_size,:], axis=1, func1d=func)
    TODO - allow the function to be a parameter
    """
    def custom_medfilt(arr):
        return median_filter(arr, var_dict['window_size'])

    # TODO - results not equal
    #def custom_medfilt2(arr):
    #    # Credit: https://gist.github.com/f0k/2f8402e4dfb6974bfcf1
    #    def rolling_median(arr, width):
    #        l = list(arr[0].repeat(width))
    #        mididx = (width) // 2
    #        result = np.empty_like(arr)
    #        for idx, new_elem in enumerate(arr):
    #            old_elem = arr[max(0, idx - width)]
    #            del l[bisect_left(l, old_elem)]
    #            insort(l, new_elem)
    #            result[idx] = l[mididx]
    #        return result
    #    return rolling_median(arr, var_dict['window_size'])
    dest = np.frombuffer(var_dict['shared_result_chunks'][idx]).reshape(var_dict['shared_movie_chunks_shapes'][idx])
    result = np.apply_along_axis(
                  arr=np.frombuffer(var_dict['shared_movie_chunks'][idx]).reshape(var_dict['shared_movie_chunks_shapes'][idx]),
                  axis=1,
                  func1d=custom_medfilt)
    np.copyto(dest, result)

def multiprocess_median(movie, window_size):
    raw_movie_shape = movie.shape
    l.debug(f'raw_movie_shape: {raw_movie_shape}')
    l.debug(f'movie.dtype: {movie.dtype}')
    n_workers = cpu_count()
    l.debug(f'n_workers: {n_workers}')

    # reshape movie to 2d
    movie = movie.reshape(-1, movie.shape[-1])
    movie_shape = movie.shape

    chunk_indices = np.linspace(start=0, stop=movie_shape[0], num=n_workers+1, dtype='int')
    # allocate shared memory arrays (without lock)
    shared_movie_chunks = []
    shared_movie_chunks_shapes = []
    shared_result_chunks = []
    for i in range(n_workers):
        start = chunk_indices[i]
        stop = chunk_indices[i+1]
        num_frames = movie_shape[-1]
        current_len = stop - start
        shared_movie_chunks.append(RawArray('d', int(current_len * num_frames)))
        shared_movie_chunks_shapes.append((current_len, num_frames))
        shared_movie_chunk_np = np.frombuffer(shared_movie_chunks[i], dtype='double').reshape(current_len, num_frames)
        np.copyto(shared_movie_chunk_np, movie[start:stop,:])

        shared_result_chunks.append(RawArray('d', int(current_len * movie_shape[-1])))

    del movie, shared_movie_chunk_np

    global var_dict
    var_dict = {} # global dictionary for variables from initializer
    #t0 = time.time()
    with Pool(processes=n_workers, initializer=initializer, initargs=(shared_movie_chunks, shared_movie_chunks_shapes, shared_result_chunks, window_size)) as pool:
        del shared_movie_chunks
        pool.map(worker_func, range(n_workers))
    #l.info(f'calculation time: {time.time() - t0}')
    return np.concatenate([np.frombuffer(shared_result_chunks[i], dtype='double').reshape(shared_movie_chunks_shapes[i]) for i in range(n_workers)]).reshape(raw_movie_shape).astype('float32')

if __name__ == '__main__':
    window_size = 15
    logger = l.getLogger('')
    logger.setLevel(l.INFO)
    l.info('begin')
    with h5py.File('main_results/raw_mov.h5', 'r') as f:
        l.info('loading movie...')
        movie = np.ascontiguousarray(f['raw_mov'])
        #movie = movie[:,:,:100]
        l.info('multiprocess version')
        t0 = time.time()
        mp_result = multiprocess_median(movie=movie, window_size=window_size)
        l.info(f'parallel time: {time.time() - t0}')

    l.info('serial version, 3D')
    t0 = time.time()
    sp_result_3d = median_filter(movie, [1,1,window_size])
    l.info(f'serial time 3D: {time.time() - t0}')

    l.info('serial version, apply_along_axis')
    def custom_medfilt(arr):
        return median_filter(arr, window_size)
    t0 = time.time()
    original_shape = movie.shape
    sp_result_1d = np.apply_along_axis(arr=movie.reshape(-1, movie.shape[-1]), axis=1, func1d=custom_medfilt).reshape(original_shape)
    l.info(f'serial time apply_along_axis: {time.time() - t0}')

    l.info('check parallel results equal...')
    np.testing.assert_array_equal(sp_result_3d, mp_result)

    l.info('check serial results equal...')
    np.testing.assert_array_equal(sp_result_3d, sp_result_1d)
