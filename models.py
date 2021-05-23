import sys, os
import time, random

import numpy as np
import pandas as pd
import scipy

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import wandb

from utils import elapsed, sample_triples, normalize, orthonormalize

import pdb


class Tensor_completion(object):
    
    def __init__(
        self, n, rank, n_entries,
        V_x=None, V_y=None, 
        V_z=None, x_vecs=None,
        y_vecs=None, z_vecs=None,
        coeffs=None, correlated=False,
        entries_arr=None, noisy=False,
        noise_size=0.1, randominit=True,
        seed=2021, true_rank=None
    ): 
        random.seed(seed)
        np.random.seed(seed)
        self.n = n
        self.true_rank = true_rank or rank
        self.rank = rank
        self.dim_x, self.dim_y, self.dim_z = n
        self.entries_arr_true = np.copy(entries_arr)
        self.entries_arr = np.copy(entries_arr)
        
        self.noisy = noisy
        self.noise_size = noise_size
        self.randominit = randominit
        
        self.x_vecs = None
        self.y_vecs = None
        self.z_vecs = None
        self.coeffs = None
        
        self.n_entries = n_entries
        self.correlated = correlated
        
        self.x_dict, self.y_dict, self.z_dict = {}, {}, {}
        if entries_arr is None:
            x_coords, y_coords, z_coords = sample_triples(
                n_entries, self.dim_x, self.dim_y, self.dim_z
            )
            if self.correlated:
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs = self.gen_biased(
                    self.dim_x, self.dim_y, self.dim_z, true_rank
                )
            else:
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs = self.gen(
                    self.dim_x, self.dim_y, self.dim_z, true_rank
                )
            
            self.x_vecs = x_vecs or self.x_vecs
            self.y_vecs = y_vecs or self.y_vecs
            self.z_vecs = z_vecs or self.z_vecs
            
            self.fill(
                x_coords, y_coords, z_coords,
                self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs, self.x_dict
            )
            self.fill(
                y_coords, z_coords, x_coords, 
                self.coeffs, self.y_vecs, self.z_vecs, self.x_vecs, self.y_dict
            )
            self.fill(
                z_coords, x_coords, y_coords, 
                self.coeffs, self.z_vecs, self.x_vecs, self.y_vecs, self.z_dict
            )
            
            self.entries_arr = np.zeros((4, n_entries), dtype=np.float)
            for i, (x_coord, y_coord, z_coord) in enumerate(zip(x_coords, y_coords, z_coords)):
                self.entries_arr[0][i] = x_coord
                self.entries_arr[1][i] = y_coord
                self.entries_arr[2][i] = z_coord
                self.entries_arr[3][i] += self.x_dict[x_coord][y_coord][z_coord]
                self.entries_arr[3][i] += self.x_dict[x_coord][y_coord][z_coord]
                self.entries_arr[3][i] += self.x_dict[x_coord][y_coord][z_coord]
                self.entries_arr[3][i] /= 3
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
                
        if noisy:
            i = 0
            for x_coord, y_coord, z_coord in zip(entries_arr[0], entries_arr[1], entries_arr[2]):
                self.x_dict[x_coord][y_coord][z_coord] += np.random.normal(0, noise_size)
                self.y_dict[y_coord][z_coord][x_coord] += np.random.normal(0, noise_size)
                self.z_dict[z_coord][x_coord][y_coord] += np.random.normal(0, noise_size)
                self.entries_arr[3, i] += np.random.normal(0, noise_size)
                
                
        #Initialize starting V_x, V_y, V_z
        if randominit:
            entries_scale = np.sqrt(np.mean(self.entries_arr[3]**2))
            self.V_x = V_x or np.random.randn(rank, self.dim_x)*entries_scale**(1/3)
            self.V_y = V_y or np.random.randn(rank, self.dim_y)*entries_scale**(1/3)
            self.V_z = V_z or np.random.randn(rank, self.dim_z)*entries_scale**(1/3)
        else:
            self.V_x = V_x or self.initialization(self.y_dict, nz=self.dim_x)
            self.V_y = V_y or self.initialization(self.z_dict, nz=self.dim_y)
            self.V_z = V_z or self.initialization(self.x_dict, nz=self.dim_z)
            
    #Compute initial subspace estimates
    def initialization(self, x_dict, nz):
        r = self.rank
        p = min(float(self.n_entries / (self.dim_x*self.dim_y*self.dim_z)), 1.)
        M_x = np.zeros((nz,nz))
        for x in x_dict.keys():
            for y in x_dict[x].keys():
                for z1 in x_dict[x][y].keys():
                    for z2 in x_dict[x][y].keys():
                        val = x_dict[x][y][z1] * x_dict[x][y][z2]
                        if(z1 == z2):
                            val = val/p
                        else:
                            val = val/(p*p)
                        M_x[z1][z2] += val
        svd = TruncatedSVD(n_components=r)
        svd.fit(M_x)
        return(svd.components_)
    
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
    
    def get_entries(self, n_entries):
        if self.x_vecs is None or self.y_vecs is None or self.z_vecs is None or self.coeffs is None:
            raise NotImplementedError()
        x_coords, y_coord, z_coords = sample_triples(n_entries, self.x_dim, self.y_dim, self.z_dim)
        entries = np.empty((4, n_entries), dtype=np.float)
        for i, (x_coord, y_coord, z_coord) in enumerate(x_coords, y_coords, z_coords):
            entries[0][i] = x_coord
            entries[1][i] = y_coord
            entries[2][i] = z_coord
            entries[3][i] = self.T(x_coord, y_coord, z_coord, self.coeffs, self.x_vecs, self.y_vecs, self.z_vecs)
        
        return entries
    
    def fit(self):
        raise NotImplementedError()
    
    @staticmethod
    def predict(self, solution, idx_frames):
        raise NotImplementedError()


class LM_completion(Tensor_completion):

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
            prediction = np.clip(np.dot(feature, solution_coeffs), 0, 256)
            true_val = test_entries[3, i] 
            total_norm += np.square(true_val)
            total_error += np.square(prediction - true_val)
            
        rse = np.sqrt(total_error/total_norm)
        mse = total_error / num_test_entries
        rmse = np.sqrt(total_error / num_test_entries)
        psnr = 20*np.log10(np.max(np.abs(test_entries[3]))) - 10*np.log10(mse)
        return rse, mse, rmse, psnr
    
    def predict_entries(self, entries, V_x=None, V_y=None, V_z=None):
        nx, ny, nz = self.n
        features = []
        target = []
        x_dict = self.x_dict
        V_x = V_x or np.transpose(self.V_x)
        V_y = V_y or np.transpose(self.V_y)
        V_z = V_z or np.transpose(self.V_z)
        
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
        
        num_entries = entries.shape[1]
        pred = np.empty_like(entries[0], dtype=np.float)
        for i in range(num_test_entries):
            x = int(entries[0, i])
            y = int(entries[1, i])
            z = int(entries[2, i])
            part = np.tensordot(V_x[x], V_y[y], axes = 0).flatten()
            feature = np.tensordot(part, V_z[z], axes = 0).flatten()
            pred[i] = np.dot(feature, solution_coeffs)
        return np.clip(pred, 0, 256)
    
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
        return pred.clip(0, 256)

    def fit(
        self, max_iter, test_entries, 
        val_entries=None, threshold=1e-6,
        logger=None, inplace=False,
        rank=None, **kwargs
    ):
        nx, ny, nz = self.n
        rank = self.rank
        n_entries = self.n_entries
        
        V_x = np.copy(self.V_x)
        V_y = np.copy(self.V_y)
        V_z = np.copy(self.V_z)
        
        x_dict, y_dict, z_dict = self.x_dict, self.y_dict, self.z_dict

        V_x = orthonormalize(V_x)
        V_y = orthonormalize(V_y)
        V_z = orthonormalize(V_z)
        V_x = np.transpose(V_x)
        V_y = np.transpose(V_y)
        V_z = np.transpose(V_z)

        V_xmat = np.random.normal(0,1, (rank, nx))
        V_yzmat = np.random.normal(0,1, (rank, ny*nz))
        V_xmat = orthonormalize(V_xmat)
        V_yzmat = orthonormalize(V_yzmat)
        V_xmat = np.transpose(V_xmat)
        V_yzmat = np.transpose(V_yzmat)
        
        V_x_best = np.copy(V_x)
        V_y_best = np.copy(V_y)
        V_z_best = np.copy(V_z)
        best_error = 100
        curr_error = 1
        execution_time = 0
        
        #AltMin Steps
        for i in range(max_iter):
            elapsed()
            if(curr_error > threshold):
                V_x = self.subspace_altmin(V_x, V_y, V_z, x_dict, nx, ny, nz)
                V_y = self.subspace_altmin(V_y, V_z, V_x, y_dict, ny, nz, nx)
                V_z = self.subspace_altmin(V_z, V_x, V_y, z_dict, nz, nx, ny)
                if val_entries is not None:
                    val_errors = self.eval_error_subspace(V_x,V_y,V_z, x_dict, val_entries)
                    val_error = val_errors[0]
                    val_logs = {
                        f"val rse":val_errors[0],
                        f"val mse":val_errors[1],
                        f"val rmse":val_errors[2],
                        f"val psnr":val_errors[3],
                    }
                curr_errors = self.eval_error_subspace(V_x,V_y,V_z, x_dict, test_entries)
                test_logs = {
                        f"test rse":curr_errors[0],
                        f"test mse":curr_errors[1],
                        f"test rmse":curr_errors[2],
                        f"test psnr":curr_errors[3],
                    }
                cur_error = curr_errors[0]
            
            if val_entries is not None and best_error > val_error:
                best_error = val_error
                V_x_best, V_y_best, V_z_best = np.copy(V_x), np.copy(V_y), np.copy(V_z)
            
            if logger is not None:
                logger.log(test_logs, step=i)
                if val_entries is not None:
                    logger.log(val_logs, step=i)
                execution_time += elapsed()
                logger.log({"execution time": execution_time}, step=i)
        
        if val_entries is not None:
            coeffs = self.get_coeffs(V_x_best, V_y_best, V_z_best, x_dict, self.n)
            if inplace:
                self.V_x = np.transpose(V_x_best)
                self.V_y = np.transpose(V_y_best)
                self.V_z = np.transpose(V_z_best)
            return V_x_best, V_y_best, V_z_best, coeffs
        
        if inplace:
            self.V_x = np.transpose(V_x)
            self.V_y = np.transpose(V_y)
            self.V_z = np.transpose(V_z)
        
        coeffs = self.get_coeffs(V_x, V_y, V_z, x_dict, self.n)
        return V_x, V_y, V_z, coeffs
    
    @staticmethod
    def predict(solution, idx_frames):
        V_x, V_y, V_z, coeffs = solution
        recovered_frames = []
        for idx_frame in idx_frames:
            recovered_frames.append(LM_completion.recover_frame(idx_frame, coeffs, V_x, V_y, V_z))
        return recovered_frames
                                

class ALS_NN(Tensor_completion):
    
    def aul_f_sp(X, Y, Z, n, m, u, mu, entries = None):
        """Computes the value of a score function 
        ||x||^2+||y||^2||z||^2 - u*(T - T_rec)_omega+(1/(2*mu))*||T - T_rec||^2

        returns nuc_norm_val, funct_obj, constr_violation
        """
        val_n = 0
        val_n += np.sum(X**2);
        val_n += np.sum(np.sum(Y**2, axis = 1) * np.sum(Z**2, axis = 1) )
            
        num_entries = entries.shape[1]
        entries_i = np.asarray(entries[:3, :], dtype = int)
        ent = - entries[3]
        m_arr = np.array(range(m))
        tmp = Y[m_arr[np.newaxis, :], (entries_i[1])[:, np.newaxis]]* \
                X[m_arr[np.newaxis, :], (entries_i[0])[:, np.newaxis]]* \
                       Z[m_arr[np.newaxis, :], (entries_i[2])[:, np.newaxis]]
        ent += np.sum(tmp, axis = 1)
        val = val_n + np.sum( (1/(2*mu))* ent**2-u*ent)        
        return (val, np.sqrt(np.sum(ent**2)), val_n/2)
    
    def fix_components(X, Y, Z, n, m):
        """
        Scales the components so that ||X_i|| = ||Y_i||*||Z_i||
        """
        nx, ny, nz = n
        for i in range(m):
            norm_x = np.sqrt(np.sqrt(np.sum(X[i]**2)))
            norm_yz = np.sqrt(np.sqrt(np.sum(Y[i]**2)*np.sum(Z[i]**2)))
            if (norm_x>1e-8) and (norm_yz>1e-8):
                X[i] = X[i]*(norm_yz/norm_x)
                Y[i] = Y[i]*np.sqrt(norm_x/norm_yz)
                Z[i] = Z[i]*np.sqrt(norm_x/norm_yz)
        return (X, Y, Z)
    
    def eval_error_direct_fast(X, Y, Z, n, m, entries):
        nx, ny, nz = n
        total_error = 0
        total_norm = 0
        num_entries = entries.shape[1]
        entries_i = np.asarray(entries[:3, :], dtype = int)
        ent = - entries[3]
        m_arr = np.array(range(m))
        
        #entries of approx tensor from X, Y, Z
        tmp = Y[m_arr[np.newaxis, :], (entries_i[1])[:, np.newaxis]]* \
                X[m_arr[np.newaxis, :], (entries_i[0])[:, np.newaxis]]* \
                       Z[m_arr[np.newaxis, :], (entries_i[2])[:, np.newaxis]]
        error = ent + np.clip(np.sum(tmp, axis = 1), 0, 256)
        total_error = np.sqrt(np.sum(error**2))
        total_norm = np.sqrt(np.sum(entries[3]**2))
        
        rse = total_error / total_norm
        mse = np.mean(error**2)
        rmse = total_error / np.sqrt(num_entries)
        psnr = 20*np.log10(np.max(np.abs(entries[-1]))) - 10*np.log10(mse)
        
        return rse, mse, rmse, psnr
    
    def reg_fast_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        nx, ny, nz = n
        XT = X.T
        for r in range(nx):
            mask = (entries[0] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r + m
            num_feat = m         
            
            B = np.zeros(num_sampl)
            M_entr1 = np.sqrt(2*mu*lam) * np.eye(m)
        
            m_arr = np.array(range(m))
            M_entr2 = Y[m_arr[np.newaxis,:], (entries_r[1])[:, np.newaxis]] * Z[m_arr[np.newaxis,:], (entries_r[2])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r + mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            
            res = scipy.sparse.linalg.lsmr(M, B)
            XT[r] = res[0]
        X = XT.T
        return X
    
    def reg_fast_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        nx, ny, nz = n
        YT = Y.T
        for r in range(ny):
            mask = (entries[1] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r + m
            num_feat = m         
       
            B = np.zeros(num_sampl)
            M_entr1 = np.eye(m)
            for i in range(m):
                M_entr1[i, i] *= np.sqrt(2 * mu * lam * np.sum(Z[i]**2))
        
            m_arr = np.array(range(m))
            M_entr2 = X[m_arr[np.newaxis,:], (entries_r[0])[:, np.newaxis]] * Z[m_arr[np.newaxis,:], (entries_r[2])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r + mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            
            res = scipy.sparse.linalg.lsmr(M, B)
            YT[r] = res[0]
        Y = YT.T
        return Y
    
    def reg_fast_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        nx, ny, nz = n
        ZT = Z.T
        for r in range(nz):
            mask = (entries[2] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r + m
            num_feat = m         
       
            B = np.zeros(num_sampl)
            M_entr1 = np.eye(m)
            for i in range(m):
                M_entr1[i, i] *= np.sqrt(2 * mu * lam * np.sum(Y[i]**2))
        
            m_arr = np.array(range(m))
            M_entr2 = Y[m_arr[np.newaxis,:], (entries_r[1])[:, np.newaxis]] * X[m_arr[np.newaxis,:], (entries_r[0])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r+mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            
            res = scipy.sparse.linalg.lsmr(M, B)
            ZT[r] = res[0]
        Z = ZT.T
        return Z
    
    def compute_adjust_sp(X, Y, Z, n, m, mu, u, entries_a):
        """Adjusts u vector for Lagrangian"""
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        num_entries = entries.shape[1]
        y = np.zeros(num_entries)
        total_err = 0
        for i in range(num_entries):
            val = - entries_val[i]
            for j in range(m):
                val += Y[j, entries[1, i]]*X[j, entries[0, i]]*Z[j, entries[2, i]]
            total_err += val**2
            y[i] = u[i] - val/mu
        return (y, np.sqrt(total_err))
    
    def run_minimization(
        self, test_entries, 
        val_entries=None, threshold=1e-6,
        inplace=False, mu=1.0,
        tau=0.1, max_global_iter=30, 
        max_iter=1000, logger=None, lam=1.0
    ):
        nx, ny, nz = self.n
        m = self.rank
        n = self.n
        num_entries = self.n_entries
        
        X = np.copy(self.V_x)
        Y = np.copy(self.V_y)
        Z = np.copy(self.V_z)
        
        X_best = X
        Y_best = Y
        Z_best = Z
        best_error = 100
        
        mu = mu
        nu = 1.0
        
        global_iter = 0
        it = 0

        coef = 10.0
        power = 0.0
        u  = np.zeros(num_entries)
        entries_a = self.entries_arr
        execution_time = 0

        while global_iter<max_global_iter and it<max_iter:
            score, _, _ = ALS_NN.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)

            small_step = 0

            stop_condition = False 
            new_progress = 0
            while not stop_condition:
                it += 1
                progress = new_progress
                X = ALS_NN.reg_fast_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                Y = ALS_NN.reg_fast_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                Z = ALS_NN.reg_fast_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                new_score, err_obs, nuc_norm = ALS_NN.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
                new_progress = score - new_score
                score = new_score
                small_step+=1
                
                #Compute errors
                test_rse, test_mse, test_rmse, test_psnr = ALS_NN.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
                if val_entries is not None:
                    val_rse, val_mse, val_rmse, val_psnr = ALS_NN.eval_error_direct_fast(X, Y, Z, n, m, val_entries)
                
                execution_time += elapsed()
                logs = {
                    "mu": mu,
                    "train error": err_obs,
                    "nuc.norm": nuc_norm,
                    "test rse": test_rse,
                    "test mse": test_mse,
                    "test rmse": test_rmse,
                    "test psnr": test_psnr,
                    "execution time": execution_time
                }
                
                if val_entries is not None:
                    logs.update({
                        "val rse": val_rse,
                        "val mse": val_mse,
                        "val rmse": val_rmse,
                        "val psnr":val_psnr
                    })
                    if best_error > val_rse:
                        X_best = np.copy(X)
                        Y_best = np.copy(Y)
                        Z_best = np.copy(Z)
                
                if logger is not None:
                    logger.log(logs, step=it)
                        

                stop_condition = (
                    (small_step>5+10*global_iter) or (it>=max_iter) 
                    or (
                        (small_step >= 5) and 
                        (
                            (progress>=1.2*new_progress) or (new_progress<0.01*np.abs(score/(global_iter+1)))
                        )
                    )
                )
            
            if (lam>0):    
                X, Y, Z = ALS_NN.fix_components(X, Y, Z, n, m)
            u_new, err = ALS_NN.compute_adjust_sp(X, Y, Z, n, m, mu, u, entries_a)
            if (err < nu):
                if (lam>0.0):
                    u = u_new
                    power += 1
            else: 
                mu = mu * tau
                power = 1
                global_iter += 1
            nu = coef * mu**(0.1 + 0.2*power)
            
        return X_best, Y_best, Z_best


    def run_min_balanced(
        self, test_entries, 
        val_entries=None, threshold=1e-6,
        inplace=False, mu=1.0,
        tau=0.1, max_global_iter=30, 
        max_iter=1000, logger=None, lam=1.0
    ):
        nx, ny, nz = self.n
        n = self.n
        m = self.rank
        num_entries = self.n_entries
        
        X = np.copy(self.V_x)
        Y = np.copy(self.V_y)
        Z = np.copy(self.V_z)
        
        X_best = X
        Y_best = Y
        Z_best = Z
        best_error = 100
        
        mu = mu
        nu = 1.0
        
        global_iter = 0
        it = 0

        coef = 10.0
        power = 0.0
        u  = np.zeros(num_entries)
        entries_a = self.entries_arr
        execution_time = 0
        
        while global_iter<max_global_iter and it<max_iter:
            score, nuc_norm, err_obs = ALS_NN.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
            nu = err_obs / 1.5
            small_step = 0

            stop_condition = False 
            new_progress = 0
            
            
            while not stop_condition:
                it += 1
                progress = new_progress
                X = ALS_NN.reg_fast_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                Y = ALS_NN.reg_fast_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                Z = ALS_NN.reg_fast_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                if (lam>0):    
                    X, Y, Z = ALS_NN.fix_components(X, Y, Z, n, m)
                new_score, err_obs, nuc_norm = ALS_NN.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
                new_progress = score - new_score
                score = new_score
                small_step += 1
                test_rse, test_mse, test_rmse, test_psnr = ALS_NN.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
                if val_entries is not None:
                    val_rse, val_mse, val_rmse, val_psnr = ALS_NN.eval_error_direct_fast(X, Y, Z, n, m, val_entries)
                
                execution_time += elapsed()
                logs = {
                    "mu": mu,
                    "train error": err_obs,
                    "nuc.norm": nuc_norm,
                    "test rse": test_rse,
                    "test mse": test_mse,
                    "test rmse": test_rmse,
                    "test psnr": test_psnr,
                    "execution time": execution_time,
                }
                
                if val_entries is not None:
                    logs.update({
                        "val rse": val_rse,
                        "val mse": val_mse,
                        "val rmse": val_rmse,
                        "val psnr": val_psnr
                    })
                    if best_error > val_rse:
                        X_best = np.copy(X)
                        Y_best = np.copy(Y)
                        Z_best = np.copy(Z)
                
                if logger is not None:
                    logger.log(logs, step=it)

                stop_condition = (
                    (small_step>5+10*global_iter) 
                    or (it>=max_iter) 
                    or (
                        (small_step >= 5) and (
                            (progress>=1.2*new_progress) or (new_progress<0.01*np.abs(score/(global_iter+1)))
                        )
                    )
                )
            #update u and mu
            u_new, err = ALS_NN.compute_adjust_sp(X, Y, Z, n, m, mu, u, entries_a)
            if (err_obs > nu) and (power<6):
                if (lam>0.0):
                    u = u_new
                    power += 1
            else: 
                mu = 0.5*err_obs/(nuc_norm*(global_iter+1)**1.3)
                power = 1
                global_iter += 1
                nu = err*5

        return X_best, Y_best, Z_best
    
    def fit(
        self, test_entries, 
        val_entries=None, threshold=1e-6,
        inplace=False, mu=1.0,
        tau=0.1, max_global_iter=30, 
        max_iter=1000, logger=None, lam=1.0,
        which_alg="balanced", **kwargs
    ):
        res = None
        if which_alg=="balanced":
            res = self.run_min_balanced(
                test_entries=test_entries, val_entries=val_entries, 
                threshold=threshold,
                inplace=inplace, mu=mu, tau=tau,
                max_global_iter=max_global_iter,
                max_iter=max_iter,
                logger=logger, lam=lam
            )
        elif which_alg=="vanilla":
            res = self.run_minimization(
                test_entries=test_entries, val_entries=val_entries, 
                threshold=threshold,
                inplace=inplace, mu=mu, tau=tau,
                max_global_iter=max_global_iter,
                max_iter=max_iter,
                logger=logger, lam=lam
            )
        return res
    
    @staticmethod
    def predict(solution, idx_frames):
        X, Y, Z = solution
        ny = Y.shape[1]
        nz = Z.shape[1]
        rank = Y.shape[0]
        tensor = np.zeros(len(idx_frames) * ny * nz)
        for i in range(rank):
            tmp = np.kron(Y[i], Z[i])
            tensor += np.kron(X[i][idx_frames], tmp)
        
        return np.clip(tensor.reshape(len(idx_frames), ny, nz), 0, 256)
