import os, sys, time
import random, functools, itertools
import cv2

import numpy as np


def read_yuv2rgb(height, width, n_frames, file_name, file_dir=""):
    file_path = os.path.join(file_dir, file_name)
    yuv_data = np.fromfile(file_path, dtype='uint8')
    
    yuv_data = yuv_data.reshape((n_frames, height*3//2, width))
    rgb_data = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    
    for i in range(n_frames):
        yuv_frame = yuv_data[i]
        rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
        rgb_data[i] = rgb_frame
    
    return rgb_data


def read_yuv2gray(height, width, n_frames, file_name, file_dir=""):
    file_path = os.path.join(file_dir, file_name)
    yuv_data = np.fromfile(file_path, dtype='uint8')
    
    yuv_data = yuv_data.reshape((n_frames, height*3//2, width))
    gray_data = np.empty((n_frames, height, width), dtype=np.float)
    
    for i in range(n_frames):
        yuv_frame = yuv_data[i]
        gray_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2GRAY_I420)
        gray_data[i] = gray_frame
    
    return gray_data


def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff


def get_tensor_entries(tensor, size, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    nx, ny, nz = tensor.shape
    samples = np.random.choice(nx*ny*nz, size, replace=False)
    x_coords = samples%nx
    y_coords = ((samples - x_coords) // nx) % ny
    z_coords = ((samples - nx*y_coords - x_coords) // (nx*ny)) % nz
    val = tensor[x_coords, y_coords, z_coords]
    return np.vstack((x_coords, y_coords, z_coords, val))
    
    