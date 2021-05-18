import sys, os
import time, random

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import wandb

import pdb


class ML_completion:
    
    def __init__(
        self, n, rank, n_entries,
        V_x=None, V_y=None, 
        V_z=None, x_vecs=None,
        y_vecs=None, z_vecs=None,
        coeffs=None, correlated=False,
        entries_arr=None
    ): 
        self.n = n
        self.rank = rank
        self.dim_x, self.dim_y, self.dim_z = n
        self.entries_arr = entries_arr
        
        #Initialize starting V_x, V_y, V_z
        self.V_x = V_x or np.random.randn(rank, self.dim_x)
        self.V_y = V_y or np.random.randn(rank, self.dim_y)
        self.V_z = V_z or np.random.randn(rank, self.dim_z)
        
        self.x_vecs = None
        self.y_vecs = None
        self.z_vecs = None
        self.coeffs = None
        
        self.n_entries = n_entries
        self.correlated = correlated
        
        self.x_dict, self.y_dict, self.z_dict = {}, {}, {}
        if entries_arr is None:
            self.x_coords, self.y_coords, self.z_coords = self.sample(
                n_entries, self.dim_x, self.dim_y, self.dim_z
            )
            if self.correlated:
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs = self.gen_biased(
                    self.dim_x, self.dim_y, self.dim_z, rank
                )
            else:
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs = self.gen(
                    self.dim_x, self.dim_y, self.dim_z, rank
                )
            
            self.x_vecs = x_vecs or self.x_vecs
            self.y_vecs = y_vecs or self.y_vecs
            self.z_vecs = z_vecs or self.z_vecs
            
            self.fill(
                self.x_coords, self.y_coords, self.z_coords,
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs, self.x_dict
            )
            self.fill(
                self.y_coords, self.z_coords, self.x_coords, 
                self.coeffs, self.y_vecs, self.z_vecs, self.x_vecs, self.y_dict
            )
            self.fill(
                self.z_coords, self.x_coords, self.y_coords, 
                self.coeffs, self.z_vecs, self.x_vecs, self.y_vecs, self.z_dict
            )
        else:
            for x_coord, y_coord, z_coord, val in zip(
                entries_arr[0], entries_arr[1], entries_arr[2], entries_arr[3]
            ):
                x_coord = int(x_coord)
                y_coord = int(y_coord)
                z_coord = int(z_coord)
                self.x_dict[x_coord] = self.x_dict.get(x_coord, None) or {}
                self.y_dict[y_coord] = self.y_dict.get(y_coord, None) or {}
                self.z_dict[z_coord] = self.z_dict.get(z_coord, None) or {}
                
                self.x_dict[x_coord][y_coord] = self.x_dict[x_coord].get(y_coord, None) or {}
                self.y_dict[y_coord][z_coord] = self.y_dict[y_coord].get(z_coord, None) or {}
                self.z_dict[z_coord][x_coord] = self.z_dict[z_coord].get(x_coord, None) or {}
                
                self.x_dict[x_coord][y_coord][z_coord] = val
                self.y_dict[y_coord][z_coord][x_coord] = val
                self.z_dict[z_coord][x_coord][y_coord] = val
                

    def gen(self, nx, ny, nz, rank):
        coeffs = np.ones(rank)
        x_vecs = np.random.normal(0,1,(rank,nx))
        y_vecs = np.random.normal(0,1,(rank,ny))
        z_vecs = np.random.normal(0,1,(rank,nz))
        return (coeffs, x_vecs, y_vecs, z_vecs)

    def gen_biased(self, nx, ny, nz, rank):
        """Generate random correlated tensor"""
        coeffs = np.zeros(rank)
        x_vecs = np.zeros((rank,nx))
        y_vecs = np.zeros((rank,ny))
        z_vecs = np.zeros((rank,nz))
        for i in range(rank):
            coeffs[i] = 0.5**i
            if(i==0):
                x_vecs[i] = np.sqrt(nx) * normalize(np.random.normal(0,1,nx))
                y_vecs[i] = np.sqrt(ny) * normalize(np.random.normal(0,1,ny))
                z_vecs[i] = np.sqrt(nz) * normalize(np.random.normal(0,1,nz))
            else:
                x_vecs[i] = np.sqrt(nx) * normalize(np.random.normal(0,0.5,nx) + x_vecs[0])
                y_vecs[i] = np.sqrt(ny) * normalize(np.random.normal(0,0.5,ny) + y_vecs[0])
                z_vecs[i] = np.sqrt(nz) * normalize(np.random.normal(0,0.5,nz) + z_vecs[0])
        return (coeffs, x_vecs,y_vecs,z_vecs)

    def T(self, i,j,k, coeffs, x_vecs, y_vecs, z_vecs):
        """Evaluate tensor given coordinates"""
        ans = 0
        for a in range(self.rank):
            ans += coeffs[a] * x_vecs[a][i] * y_vecs[a][j] * z_vecs[a][k]
        return ans
        
    def sample(self, size, nx, ny, nz):
        samples = np.random.choice(nx*ny*nz, size, replace=False)
        x_coords = samples%nx
        y_coords = ((samples - x_coords) // nx) % ny
        z_coords = ((samples - nx*y_coords - x_coords) // (nx*ny)) % nz
        return (x_coords, y_coords, z_coords)

    def fill(self, x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict):
        n_entries = x_coords.size
        for i in range(n_entries):
            #For x_dict coordinates are in order x,y,z
            if(x_coords[i] in x_dict.keys()):
                if(y_coords[i] in x_dict[x_coords[i]].keys()):
                    if(z_coords[i] in x_dict[x_coords[i]][y_coords[i]].keys()):
                        pass
                    else:
                        x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = self.T(
                            x_coords[i], y_coords[i], z_coords[i], coeffs,x_vecs, y_vecs, z_vecs
                        )
                else:
                    x_dict[x_coords[i]][y_coords[i]] = {}
                    x_dict[x_coords[i]][y_coords[i]][z_coords[i]]= self.T(
                        x_coords[i], y_coords[i], z_coords[i], coeffs, x_vecs, y_vecs, z_vecs
                    )
            else:
                x_dict[x_coords[i]] = {}
                x_dict[x_coords[i]][y_coords[i]] = {}
                x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = self.T(
                    x_coords[i], y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs
                )
    
    def normalize(v):
        u = v / np.linalg.norm(v)
        return u
    
    def orthonormalize(V):
        """Given rxn array output orthonormal basis"""
        a = len(V)
        b = len(V[0])
        for i in range(a):
            for j in range(i):
                V[i] = V[i] - np.dot(V[i],V[j])*V[j]
            V[i] = ML_completion.normalize(V[i])
        return V

    def eval_error_matrix(self, V_x,V_yz, nx, ny, nz, r):
        """Normalized MSE for unfolded matrix completion"""
        #take random sample of entries to speed up evaluation
        num_trials = 1000
        total_error = 0
        total_norm = 0
        for i in range(num_trials):
            x = np.random.randint(nx)
            y = np.random.randint(ny)
            z = np.random.randint(nz)
            prediction = 0
            for j in range(r):
                prediction += V_x[x][j] * V_yz[nz * y + z][j]
            true_val = self.T(x,y,z, coeffs, x_vecs,y_vecs, z_vecs)
            total_norm += np.square(true_val)
            total_error += np.square(prediction - true_val)
        return np.sqrt(total_error/total_norm)

    def power_altmin(V_x, V_y, V_z , x_dict, nx, ny, nz):
        """Altmin for naive tensor powering"""
        lsq_solution = []
        for i in range(nx):
            features = []
            target = []
            for y_coord in x_dict[i].keys():
                for z_coord in x_dict[i][y_coord].keys():
                    #subsample to speed up and get "unstuck"
                    check = np.random.randint(2)
                    if(check == 0):
                        features.append(np.multiply(V_y[y_coord], V_z[z_coord]))
                        target.append(x_dict[i][y_coord][z_coord])
            features = np.array(features)
            target = np.array(target)
            reg = LinearRegression(fit_intercept = False).fit(features, target)
            lsq_solution.append(reg.coef_)
        lsq_solution = np.array(lsq_solution)
        return (lsq_solution)

    def eval_error_direct(self, V_x, V_y, V_z, x_dict, nx, ny, nz):
        """Normalized MSE for naive tensor powering"""
        num_trials = 1000
        total_error = 0
        total_norm = 0
        for i in range(num_trials):
            x = np.random.randint(nx)
            y = np.random.randint(ny)
            z = np.random.randint(nz)
            prediction = 0
            for j in range(self.rank):
                prediction += V_x[x][j] * V_y[y][j] * V_z[z][j]
            true_val = x_dict[x][y][z]
            total_norm += np.square(true_val)
            total_error += np.square(prediction - true_val)
        return np.sqrt(total_error/total_norm)

    def subspace_altmin(self, V_x, V_y, V_z , x_dict, nx, ny, nz):
        """Altmin for our algorithm"""
        lsq_solution = []
        for i in range(nx):
            features = []
            target = []
            for y_coord in x_dict[i].keys():
                for z_coord in x_dict[i][y_coord].keys():
                    #subsample to speed up and get "unstuck"
                    check = np.random.randint(2)
                    if(check == 0):
                        features.append(np.tensordot(V_y[y_coord], V_z[z_coord] , axes = 0).flatten())
                        target.append(x_dict[i][y_coord][z_coord])
            features = np.array(features)
            target = np.array(target)
            reg = LinearRegression(fit_intercept = False).fit(features,target)
            lsq_solution.append(reg.coef_)
        lsq_solution = np.transpose(np.array(lsq_solution))
        svd = TruncatedSVD(n_components=self.rank)
        svd.fit(lsq_solution)
        return(np.transpose(svd.components_))

    def eval_error_subspace(self, V_x,V_y,V_z, x_dict, test_entries):
        """Normalized MSE for our algorithm"""
        nx, ny, nz = self.n
        features = []
        target = []
        
        #Find coefficients in V_x x V_y x V_z basis
        for x_coord in x_dict.keys():
            for y_coord in x_dict[x_coord].keys():
                for z_coord in x_dict[x_coord][y_coord].keys():
                    #speed up by using less entries
                    check = np.random.randint(10)
                    if(check == 0):
                        target.append(x_dict[x_coord][y_coord][z_coord])
                        part = np.tensordot(V_x[x_coord], V_y[y_coord], axes = 0).flatten()
                        full = np.tensordot(part, V_z[z_coord],axes = 0).flatten()
                        features.append(full)
        features = np.array(features)
        target = np.array(target)
        reg = LinearRegression(fit_intercept = False).fit(features, target)
        solution_coeffs = reg.coef_

        #Evaluate RMS error
        num_test_entries = test_entries.shape[1]
        total_error = 0
        total_norm = 0
        for i in range(num_test_entries):
            x = int(test_entries[0, i])
            y = int(test_entries[1, i])
            z = int(test_entries[2, i])
            part = np.tensordot(V_x[x], V_y[y], axes = 0).flatten()
            feature = np.tensordot(part, V_z[z], axes = 0).flatten()
            prediction = np.dot(feature, solution_coeffs)
            true_val = test_entries[3, i] 
            total_norm += np.square(true_val)
            total_error += np.square(prediction - true_val)
        return np.sqrt(total_error/total_norm)
    
    @staticmethod
    def get_coeffs(V_x, V_y, V_z, x_dict, n):
        nx, ny, nz = n
        features = []
        target = []
        
        #Find coefficients in V_x x V_y x V_z basis
        for x_coord in x_dict.keys():
            for y_coord in x_dict[x_coord].keys():
                for z_coord in x_dict[x_coord][y_coord].keys():
                    #speed up by using less entries
                    check = np.random.randint(10)
                    if(check == 0):
                        target.append(x_dict[x_coord][y_coord][z_coord])
                        part = np.tensordot(V_x[x_coord], V_y[y_coord], axes = 0).flatten()
                        full = np.tensordot(part, V_z[z_coord],axes = 0).flatten()
                        features.append(full)
        features = np.array(features)
        target = np.array(target)
        reg = LinearRegression(fit_intercept = False).fit(features, target)
        solution_coeffs = reg.coef_
        return solution_coeffs
    
    @staticmethod
    def recover_frame(idx_frame, sol_coeffs, V_x, V_y, V_z):
        nx, ny, nz = V_x.shape[0], V_y.shape[0], V_z.shape[0]
        pred = np.empty((ny, nz), dtype=np.float32)
        for i in range(ny):
            for j in range(nz):
                part = np.tensordot(V_x[idx_frame], V_y[i], axes = 0).flatten()
                feature = np.tensordot(part, V_z[j], axes = 0).flatten()
                pred[i, j] = np.dot(feature, sol_coeffs)
        return pred

    def run_for_tensor(
        self, max_iter, test_entries, 
        val_entries=None, threshold=1e-6,
        which_alg="Subspace Powering",
        logger=None
    ):
        nx, ny, nz = self.n
        rank = self.rank
        n_entries = self.n_entries
        
        V_x = np.copy(self.V_x)
        V_y = np.copy(self.V_y)
        V_z = np.copy(self.V_z)
        
        x_dict, y_dict, z_dict = self.x_dict, self.y_dict, self.z_dict

        V_x = ML_completion.orthonormalize(V_x)
        V_y = ML_completion.orthonormalize(V_y)
        V_z = ML_completion.orthonormalize(V_z)
        V_x = np.transpose(V_x)
        V_y = np.transpose(V_y)
        V_z = np.transpose(V_z)

        V_xmat = np.random.normal(0,1, (rank, nx))
        V_yzmat = np.random.normal(0,1, (rank, ny*nz))
        V_xmat = ML_completion.orthonormalize(V_xmat)
        V_yzmat = ML_completion.orthonormalize(V_yzmat)
        V_xmat = np.transpose(V_xmat)
        V_yzmat = np.transpose(V_yzmat)
        
        V_x_best = np.copy(V_x)
        V_y_best = np.copy(V_y)
        V_z_best = np.copy(V_z)
        best_error = 100
        curr_error = 1
        
        #AltMin Steps
        for i in range(max_iter):
            if(which_alg == "Matrix Alt Min" or which_alg == "all"):
                V_xmat, V_yzmat = self.matrix_altmin(V_xmat, V_yzmat)
                curr_error = self.eval_error_matrix(V_xmat, V_yzmat)
                
            if(which_alg == "Tensor Powering" or which_alg == "all"):
                if(curr_error > threshold):
                    V_x = self.power_altmin(V_x, V_y, V_z, x_dict, nx, ny, nz)
                    V_y = self.power_altmin(V_y, V_z, V_x, y_dict, ny, nz, nx)
                    V_z = self.power_altmin(V_z, V_x, V_y, z_dict, nz, nx, ny)
                    curr_error = eval_error_direct(V_x, V_y, V_z, x_dict)
                
            if(which_alg == "Subspace Powering" or which_alg == "all"):
                if(curr_error > threshold):
                    V_x = self.subspace_altmin(V_x, V_y, V_z, x_dict, nx, ny, nz)
                    V_y = self.subspace_altmin(V_y, V_z, V_x, y_dict, ny, nz, nx)
                    V_z = self.subspace_altmin(V_z, V_x, V_y, z_dict, nz, nx, ny)
                    if val_entries is not None:
                        val_error = self.eval_error_subspace(V_x,V_y,V_z, x_dict, val_entries)
                    curr_error = self.eval_error_subspace(V_x,V_y,V_z, x_dict, test_entries)
            
            if best_error > val_error:
                best_error = val_error
                V_x_best, V_y_best, V_z_best = V_x, V_y, V_z
            
            if logger is not None:
                logger.log({f"{which_alg}---test error":curr_error}, step=i)
                if val_entries is not None:
                    logger.log({f"{which_alg}---validation error":val_error}, step=i)
            
        if val_entries is not None:
            coeffs = self.get_coeffs(V_x_best, V_y_best, V_z_best, x_dict, self.n)
            return V_x_best, V_y_best, V_z_best, coeffs
        
        coeffs = get_coeffs(V_x, V_y, V_z, x_dict, self.n)
        return V_x, V_y, V_z, coeffs