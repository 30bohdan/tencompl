import os, sys, time
import random, functools, itertools

import numpy as np

def read_yuv2rgb(height, width, n_frames, file_name, file_dir=""):
    file_path = os.file.join(file_dir, file_name)
    yuv_data = np.from_file(file, dtype='uint8')
    
    yuv_data = yuv_data.reshape((n_frames, height*3//2, width))
    rgb_data = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    
    for i in range(n_frames):
        yuv_frame = yuv_data[i]
        rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
        rgb_data[i] = rgb_frame
    
    return rgb_data