import sys, os, time
import random

import numpy as np
import pandas as pd

import torch


class Tensor(object):
    
    def __init__(
        self, rank, dim_x, dim_y, dim_z,
        x=None, x=None, z=None,
        n_entries=None, seed=None
    ):
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.rank = rank
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.n_entries = n_entries
        
        if size is not None:
            self.entries_dict = self.generate_tensor_entries(size=n_entries, seed=seed)
            self.entries_arr = self._from_dict_to_arr()
            self.entries_list = self._from_dict_to_list()
        else:
            self.entries_dict = None
            self.entries_arr = None
            self.entries_list = None
        
        # Or any other more appropriate name
        self.x_basis = x or np.random.randn(rank, dim_x)
        self.y_basis = y or np.random.randn(rank, dim_y)
        self.z_basis = z or np.random.randn(rank, dim_z)
    
    def compute_tensor(self):
        tensor = np.zeros(self.dim_x * self.dim_y * self.dim_z)
        for i in range(self.rank):
            tmp = np.kron(self.y_basis[i], self.z_basis[i])
            tensor += np.kron(self.x_basis[i], tmp)
        return tensor
    
    def compute_nuc_approx(self):
        val = 0
        for i in range(self.rank):
            val += (np.linalg.norm(self.x_basis[i])
                    * np.linalg.norm(self.y_basis[i])
                    * np.linalg.norm(self.z_basis[i]))
        return val
    
    def generate_tensor_entries(self, size=None, seed=None):
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
            
            if (i not in entries_dict.keys()):
                entries_dict[i] = {}
            if (j not in entries_dict[i].keys()):
                entries_dict[i][j] = {}
            if (k not in entries_dict[i][j].keys()):
                val = 0
                for t in range(self.rank):
                    val += X[t, i]*Y[t, j]*Z[t, k]
                entries_dict[i][j][k] = val
                step += 1
        return entries_dict
    
    def fix_components(self, eps=1e-8):
        """
        Scales the components so that ||X_i|| = ||Y_i||*||Z_i||
        """
        for i in range(self.rank):
            norm_x = np.sqrt(np.sqrt(np.sum(self.x_basis[i]**2)))
            norm_yz = np.sqrt(np.sqrt(np.sum(self.y_basis[i]**2) * np.sum(self.z_basis[i]**2)))
            if norm_x>eps and norm_yz>eps:
                self.x_basis = self.x_basis * (norm_yz/norm_x)
                self.y_basis = self.y_basis * np.sqrt(norm_x / norm_yz)
                self.z_basis = self.z_basis * np.sqrt(norm_x / norm_yz)
        return 
    
    def _from_dict_to_arr(self):
        entries_arr = np.empty((4, self.n_entries), dtype=np.float32)
        idx = 0
        for x_key, x_val in self.entries_dict.items():
            for y_key, y_val in x_val.items():
                for z_key, z_val in y_val:
                    entries_arr[0, idx] = x_key
                    entries_arr[1, idx] = y_key
                    entries_arr[2, idx] = z_key
                    entries_arr[3, idx] = z_val
                    idx += 1
        return entries_arr
    
    def _from_dict_to_list(self):
        entries_list = []
        
        idx = 0
        for x_key, x_val in self.entries_dict.items():
            for y_key, y_val in x_val.items():
                for z_key, z_val in y_val:
                    entries_list.append((x_key, y_key, z_key, z_val))
                    idx += 1
                    
        return entries_list
    
    def reset_entries(self):
        self.entries_arr = None
        self.entries_dict = None
        self.entries_list = None
    
    def set_entries_from_list(self, idx_entries):
        for i, j, k in idx_entries:
            val = 0
            for t in range(self.rank):
                val += X[t, i]*Y[t, j]*Z[t, k]
            self.entries_dict[i][j][k] = val
        self.entries_arr =self._from_dict_to_arr()
        self.entries_list = self._from_dict_to_list()
    
    def set_entries_from_arr(self, idx_entries):
        for i, j, k in zip(idx_entries[0], idx_entries[1], idx_entries[2]):
            val = 0
            for t in range(self.rank):
                val += X[t, i]*Y[t, j]*Z[t, k]
            self.entries_dict[i][j][k] = val
        self.entries_arr =self._from_dict_to_arr()
        self.entries_list = self._from_dict_to_list()
    
    def set_entries_from_dict(self, entries_dict):
        self.entries_dict = entries_dict
        self.entries_arr =self._from_dict_to_arr()
        self.entries_list = self._from_dict_to_list()
        
    def aul_f_sp(self, u, mu):
        val_n = np.sum(self.x_basis**2)
        val_n += np.sum(np.sum(self.y_basis**2, axis=1) * np.sum(self.z_basis**2, axis=1))
        
        idx_entries = entries_arr[:3].astype(int)
        entries_val = -self.entries_arr[-1]
        m_arr = np.arrange(m)
        tmp = (self.x_basis[m_arr[np.newaxis, :], (idx_entries[0])[:, np.newaxis]]
               * self.y_basis[m_arr[np.newaxis, :], (idx_entries[1])[:, np.newaxis]]
               * self.z_basis[m_arr[np.newaxis, :], (idx_entries[2])[:, np.newaxis]])
        entries_val += np.sum(tmp, axis = 1)
        val = val_n+ np.sum( (1/(2*mu))* entries_val**2-u*entries_val)        
        return (val, np.sqrt(np.sum(entries_val**2)), val_n/2)
    
    def update_xyz(
        self, x_basis, y_basis, z_basis
        
    ):
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.z_basis = z_basis