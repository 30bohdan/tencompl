import os, sys, time
import random, functools, itertools
import cv2

import numpy as np
import matplotlib.pyplot as plt


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
    if tensor is None: return None
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
    

def sample_triples(size, nx, ny, nz):
    samples = np.random.choice(nx*ny*nz, size, replace=False)
    x_coords = samples%nx
    y_coords = ((samples - x_coords) // nx) % ny
    z_coords = ((samples - nx*y_coords - x_coords) // (nx*ny)) % nz
    return (x_coords, y_coords, z_coords)


def normalize(v):
    v_norm = v / np.linalg.norm(v)
    return v_norm


def orthonormalize(v, inplace=False):
    m = len(v)
    n = len(v[0])
    
    if not inplace:
        v_new = np.copy(v)
    else:
        v_new = v
    
    for i in range(m):
        for j in range(i):
            v_new[i] = v_new[i] - np.dot(v_new[i], v_new[j])*v_new[j]
        v_new[i] = normalize(v_new[i])
    return v_new


def compute_rse(pred, target, entries=None):
    if entries is not None:
        new_pred = pred[entries[0].astype(np.int), entries[1].astype(np.int), entries[2].astype(np.int)]
        new_target = target[entries[0].astype(np.int), entries[1].astype(np.int), entries[2].astype(np.int)]
    else:
        new_pred = pred
        new_target = target
    error = np.linalg.norm(new_pred-new_target) / np.linalg.norm(new_target)
    return error


def compute_metrics(pred, target, test_entries):
    num_test_entries = test_entries.shape[1]
    
    new_pred = pred[test_entries[0].astype(np.int), test_entries[1].astype(np.int), test_entries[2].astype(np.int)]
    new_target = target[test_entries[0].astype(np.int), test_entries[1].astype(np.int), test_entries[2].astype(np.int)]
    
    total_error = np.linalg.norm(new_target-new_pred)
    total_norm = np.linalg.norm(new_target)
    
    rse = total_error / total_norm
    mse = total_error**2 / num_test_entries
    rmse = total_error / np.sqrt(num_test_entries)
    psnr = 20*np.log10(np.max(np.abs(new_target))) - 10*np.log10(mse)
    
    return rse, mse, rmse, psnr


class Logger():
    
    def __init__(self):
        self.logs = {}
        
    def logs(self, log_dict, step=None):
        for key, value in log_dict:
            if key not in self.logs:
                self.logs[key]['x'] = []
                self.logs[key]['y'] = []
            self.logs[key]['y'].append(value)
            if step is not None:
                idx = step
            else:
                idx = self.logs[key]['x'][-1] + 1

            self.logs[key]['x'].append(idx)

    
    def reset(self, log_name):
        self.logs[log_name]['x'] = []
        self.logs[log_name]['y'] = []
        
    def plot(self, log_name, title=None, xlabel=None, ylabel=None):
        title = title or ''
        xlabel = xlabel or 'step'
        ylabel = ylabel or log_name
        plt.title(log_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        y = self.logs[log_name]['y']
        x = self.logs[log_name]['x']
        plt.plot(x, y)
        plt.show()