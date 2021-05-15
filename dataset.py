import sys, os, time
import random

import numpy as np
import pandas as pd

import torch


class Tensor(object):
    
    def __init__(self, rank, dim_x, dim_y, dim_z, size=None seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.rank = rank
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        
        if size is not None:
            self.entries_xyz = _generate_tensor_entries(size=size, seed=seed)
        else:
            self.entries_xyz = None
        
        # Or any other more appropriate name
        self.x_decomp = np.random.randn(rank, dim_x)
        self.y_decomp = np.random.randn(rank, dim_y)
        self.z_decomp = np.random.randn(rank, dim_z)
    
    def compute_tensor(self):
        tensor = np.zeros(self.dim_x * self.dim_y * self.dim_z)
        for i in range(self.rank):
            tmp = np.kron(self.y_decomp[i], self.z_decomp[i])
            tensor += np.kron(self.x_decomp[i], tmp)
        return tensor
    
    def compute_nuc_approx(self):
        val = 0
        for i in range(self.rank):
            val += (np.linalg.norm(self.x_decomp[i])
                    * np.linalg.norm(self.y_decomp[i])
                    * np.linalg.norm(self.z_decomp[i]))
        return val
    
    def _generate_tensor_entries(self, size=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if size=None:
            return None
        
        step = 0
        while step<size:
            i = np.random.randint(self.dim_x)
            j = np.random.randint(self.dim_y)
            k = np.random.randint(self.dim_z)
            
            if (i not in entries_xyz.keys()):
                entries_xyz[i] = {}
            if (j not in entries_xyz[i].keys()):
                entries_xyz[i][j] = {}
            if (k not in entries_xyz[i][j].keys()):
                val = 0
                for t in range(self.rank):
                    val += X[t, i]*Y[t, j]*Z[t, k]
                entries_xyz[i][j][k] = val
                step += 1
        return entries_xyz
    
    def fix_components(self):
        """
        Scales the components so that ||X_i|| = ||Y_i||*||Z_i||
        """
        for i in range(self.rank):
            norm_x = np.sqrt(np.sqrt(np.sum)