import numpy as np;
from scipy.optimize import minimize 
import time
import matplotlib.pyplot as plt
import sklearn.linear_model
import copy
import random
from scipy.sparse import csr_matrix 
import scipy.sparse.linalg
import pandas as pd

class am: 


    def __init__(self):
        return None
    
    def aul_f_sp(X, Y, Z, n, m, u, mu, entries = None):
        """
        Computes the value of a score function 
        ||x||^2+||y||^2||z||^2 - u*(T - T_rec)_omega+(1/(2*mu))*||T - T_rec||^2

        returns nuc_norm_val, funct_obj, constr_violation
        """
        val_n = 0;
        val_n += np.sum(X**2);
        val_n += np.sum( np.sum(Y**2, axis = 1) * np.sum(Z**2, axis = 1) )
            
        num_entries = entries.shape[1]
        entries_i = np.asarray(entries[:3, :], dtype = int)
        ent = - entries[3]
        m_arr = np.array(range(m))
        tmp = Y[m_arr[np.newaxis, :], (entries_i[1])[:, np.newaxis]]* \
                X[m_arr[np.newaxis, :], (entries_i[0])[:, np.newaxis]]* \
                       Z[m_arr[np.newaxis, :], (entries_i[2])[:, np.newaxis]]
        ent += np.sum(tmp, axis = 1)
        val = val_n+ np.sum( (1/(2*mu))* ent**2-u*ent);        
        return (val, np.sqrt(np.sum(ent**2)), val_n/2);


    def compute_tensor(X, Y, Z, n, m):
        nx, ny, nz = n
        ten = np.zeros(nx*ny*nz);
        for i in range(m):
            ten += np.kron(X[i], np.kron(Y[i], Z[i]))
        return ten;

    def compute_nuc_approx(X, Y, Z, m):
        val = 0;
        for i in range(m):
            val += np.linalg.norm(X[i])*np.linalg.norm(Y[i])*np.linalg.norm(Z[i])
        return val

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

    def generate_ten_entries(X, Y, Z, n, m, num, seed = None):
        """
        Generates num of random indicies in nx*ny*nz tensor
        The result is stored in the dictionary 
        """
        nx, ny, nz = n;
        #entries = np.zeros((nx, ny, nz));
        step = 0;
        if seed is not None:
            np.random.seed(seed)
        entries_xyz = {}    
        while (step<num):
            i = np.random.randint(nx);
            j = np.random.randint(ny);
            k = np.random.randint(nz);
            if (i not in entries_xyz.keys()):
                entries_xyz[i] = {}
            if (j not in entries_xyz[i].keys()):
                entries_xyz[i][j] = {}
            if (k not in entries_xyz[i][j].keys()):
                val = 0;
                for t in range(m):
                    val = val+X[t, i]*Y[t, j]*Z[t, k];
                entries_xyz[i][j][k] = val;
                step+=1;
        return entries_xyz

    def sample_entries(n, num_entries, seed = None):
        """
        Sample num_entries positions in nx*ny*nz tensor
        """
        nx, ny, nz = n;
        #entries = np.zeros((nx, ny, nz));
        step = 0;
        if seed is not None:
            np.random.seed(seed)
        
        dict_xyz = {}    
        while (step<num_entries):
            i = np.random.randint(nx);
            j = np.random.randint(ny);
            k = np.random.randint(nz);
            if (i not in entries_xyz.keys()):
                dict_xyz[i] = {}
            if (j not in entries_xyz[i].keys()):
                dict_xyz[i][j] = {}
            if (k not in entries_xyz[i][j].keys()):
                dict_xyz[i][j][k] = 0;
                step+=1;
        return dict_xyz
        

    def generate_ten_entries1(tensor, n, num, seed = None):
        """
        Generates num of random indicies in nx*ny*nz tensor
        The result is stored in the dictionary 
        """
        nx, ny, nz = n;
        #entries = np.zeros((nx, ny, nz));
        step = 0;
        if seed is not None:
            np.random.seed(seed)
        entries_xyz = {}    
        while (step<num):
            i = np.random.randint(nx);
            j = np.random.randint(ny);
            k = np.random.randint(nz);
            if (i not in entries_xyz.keys()):
                entries_xyz[i] = {}
            if (j not in entries_xyz[i].keys()):
                entries_xyz[i][j] = {}
            if (k not in entries_xyz[i][j].keys()):
                val = 0;
                val = val+tensor[i,j,k];
                entries_xyz[i][j][k] = val;
                step+=1;
        return entries_xyz
    
    def from_dict_to_list(D, num_entries):
        L = [(0, 0, 0, 0.0)]*num_entries
        step = 0
        for x in D.keys():
            for y in D[x].keys():
                for z in D[x][y].keys():
                    L[step] = (x, y, z, D[x][y][z])
                    step += 1
        return L
    
    def from_dict_to_arr(D, num_entries):
        L = np.zeros((4, num_entries))
        step = 0
        for x in D.keys():
            for y in D[x].keys():
                for z in D[x][y].keys():
                    L[0][step], L[1][step], L[2][step], L[3][step] = (x, y, z, D[x][y][z])
                    step += 1
        return L
    
    def from_list_to_dict(L):
        D = {}
        for tup in L:
            x, y, z, val = tup
            if x not in D.keys:
                D[x] = {}
            if y not in D[x].keys:
                D[x][y] = {}
            if z not in D[x][y].keys:
                D[x][y][z] = val
        return D
    
    def subsample_entries(L, num_entr, seed = None):
        if seed is not None:
            np.random.seed(seed)
        return random.sample(L, num_entr)
            

#     def eval_error_direct(X, Y, Z, n, m, tensor, num_trials = 10000):
#         """
#         Estimate the L2 norm between the tensor given by X,Y,Z and tensor
#         """
#         nx, ny, nz = n
#         total_error = 0
#         total_norm = 0
#         for i in range(num_trials):
#             x = np.random.randint(nx)
#             y = np.random.randint(ny)
#             z = np.random.randint(nz)
#             prediction = 0
#             for j in range(m):
#                 prediction += X[j, x] * Y[j, y] * Z[j, z]
#             true_val = tensor[x, y, z]
#             total_norm += np.square(true_val)
#             total_error += np.square(prediction - true_val)
#         return np.sqrt(total_error/total_norm)
    
    
    def eval_error_direct_fast(X, Y, Z, n, m, entries, full_output=False):
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
        error = ent + np.sum(tmp, axis = 1)
        total_error = np.sqrt(np.sum(error**2))
        total_norm = np.sqrt(np.sum(entries[3]**2))
        
        rse = total_error / total_norm
        mse = np.mean(error**2)
        rmse = total_error / num_entries
        psnr = 20*np.log10(np.max(np.abs(entries[-1]))) - 10*np.log10(mse)
        
        if not full_output: return rse
        return rse, mse, rmse, psnr

    
    
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

    def reg_sp2_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0, verbose = False):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        start_time = time.time()
        nx, ny, nz = n
        num_sampl = num_entries+nx*m
        num_feat = nx*m 
        
        M_num_entries = nx*m+num_entries*m
        M_row_ind = np.zeros(M_num_entries)
        M_col_ind = np.zeros(M_num_entries)
        M_entries = np.zeros(M_num_entries)
        M_ind = 0;
       
        B = np.zeros(num_sampl)
        #W = np.zeros(num_sampl)+1/(2*mu)
        for i in range(nx*m):
            M_row_ind[M_ind] = i
            M_col_ind[M_ind] = i
            M_entries[M_ind] = np.sqrt(2*mu*lam)
            M_ind = M_ind+1
        row = nx*m;
        
        M_row_ind[row:] = np.repeat(np.array(range(row, row + num_entries)), m)
        M_col_ind[row:] = np.tile(np.array(range(m))*nx, num_entries)+np.repeat(entries[0], m)
        m_arr = np.array(range(m))
        M_entr2 = Y[m_arr[np.newaxis,:], (entries[1])[:, np.newaxis]]*Z[m_arr[np.newaxis,:], (entries[2])[:, np.newaxis]]
        M_entries[row:] = M_entr2.reshape(num_entries*m)
        
        B[row : ] = entries_val+mu*u
        
        #opt = sklearn.linear_model.LinearRegression(fit_intercept=False)
        ##opt.fit(M, B, W)
        #opt.fit(M, B)
        #X = opt.coef_.reshape((m, nx))
        init_time = time.time()
        if verbose:
            print("X iteration initialization-1 time: {}".format(init_time - start_time))
        M = csr_matrix((M_entries, (M_row_ind, M_col_ind)), shape=(num_sampl, num_feat))
        init_time = time.time()
        if verbose:
            print("X iteration initialization-2 time: {}".format(init_time - start_time))
        res = scipy.sparse.linalg.lsmr(M, B)
        X = res[0].reshape((m, nx))
        res_time = time.time()
        if verbose:
            print("X iteration minimization time: {}".format(res_time - init_time))
        return X
    
    def reg_sp2_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0, verbose = False):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        start_time = time.time()
        nx, ny, nz = n
        num_sampl = num_entries+ny*m
        num_feat = ny*m
        #M = np.zeros((num_sampl, num_feat))
        
        M_num_entries = ny*m+num_entries*m
        M_row_ind = np.zeros(M_num_entries)
        M_col_ind = np.zeros(M_num_entries)
        M_entries = np.zeros(M_num_entries)
        M_ind = 0;
        
        B = np.zeros(num_sampl)
        #W = np.zeros(num_sampl)+1/(2*mu)
        for i in range(ny*m):
            M_row_ind[M_ind] = i
            M_col_ind[M_ind] = i
            M_entries[M_ind] = np.sqrt(2*mu*lam*np.sum(Z[i//ny]**2))
            M_ind = M_ind+1
        
        row = ny*m;

        M_row_ind[row:] = np.repeat(np.array(range(row, row + num_entries)), m)
        M_col_ind[row:] = np.tile(np.array(range(m))*ny, num_entries)+np.repeat(entries[1], m)
        m_arr = np.array(range(m))
        M_entr2 = X[m_arr[np.newaxis,:], (entries[0])[:, np.newaxis]]*Z[m_arr[np.newaxis,:], (entries[2])[:, np.newaxis]]
        M_entries[row:] = M_entr2.reshape(num_entries*m)
        
        B[row : ] = entries_val+mu*u
        
        M = csr_matrix((M_entries, (M_row_ind, M_col_ind)), shape=(num_sampl, num_feat))
        init_time = time.time()
        if verbose:
            print("Y iteration initialization time: {}".format(init_time - start_time))
      
        
        res = scipy.sparse.linalg.lsmr(M, B)
        Y = res[0].reshape((m, ny))
        res_time = time.time()
        #print("Difference between solution {}".format(np.linalg.norm(Y - Y1)))
        if verbose:
            print("Y iteration minimization time: {}".format(res_time - init_time))
        return Y
    
    def reg_sp2_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        nx, ny, nz = n
        num_sampl = num_entries+nz*m
        num_feat = nz*m
        
        M_num_entries = nz*m+num_entries*m
        M_row_ind = np.zeros(M_num_entries)
        M_col_ind = np.zeros(M_num_entries)
        M_entries = np.zeros(M_num_entries)
        M_ind = 0;
        
        B = np.zeros(num_sampl)
        #W = np.zeros(num_sampl)+1/(2*mu)
        for i in range(nz*m):
            M_row_ind[M_ind] = i
            M_col_ind[M_ind] = i
            M_entries[M_ind] = np.sqrt(2*mu*lam*np.sum(Y[i//nz]**2))
            M_ind = M_ind+1
        
        row = nz*m;
        
        M_row_ind[row:] = np.repeat(np.array(range(row, row + num_entries)), m)
        M_col_ind[row:] = np.tile(np.array(range(m))*nz, num_entries)+np.repeat(entries[2], m)
        m_arr = np.array(range(m))
        M_entr2 = Y[m_arr[np.newaxis,:], (entries[1])[:, np.newaxis]]*X[m_arr[np.newaxis,:], (entries[0])[:, np.newaxis]]
        M_entries[row:] = M_entr2.reshape(num_entries*m)
        
        B[row : ] = entries_val+mu*u
                
        M = csr_matrix((M_entries, (M_row_ind, M_col_ind)), shape=(num_sampl, num_feat))    
        #opt = sklearn.linear_model.LinearRegression(fit_intercept=False)
        #opt.fit(M, B, W)
        #Z = opt.coef_.reshape((m, nz))
        
        res = scipy.sparse.linalg.lsmr(M, B)
        Z = res[0].reshape((m, nz))
        res_time = time.time()
        return Z

    def reg_fast_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0, verbose = False):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        start_time = time.time()
        nx, ny, nz = n
        XT = X.T
        for r in range(nx):
            mask = (entries[0] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r+m
            num_feat = m         
       
            B = np.zeros(num_sampl)
            M_entr1 = np.sqrt(2*mu*lam)*np.eye(m)
        
            m_arr = np.array(range(m))
            M_entr2 = Y[m_arr[np.newaxis,:], (entries_r[1])[:, np.newaxis]]*Z[m_arr[np.newaxis,:], (entries_r[2])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r+mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            init_time = time.time()
            
            res = scipy.sparse.linalg.lsmr(M, B)
            XT[r] = res[0]
        res_time = time.time()
        X = XT.T
        if verbose:
            print("X iteration minimization time: {}".format(res_time - init_time))
        return X
    
    def reg_fast_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = 1.0, verbose = False):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        start_time = time.time()
        nx, ny, nz = n
        YT = Y.T
        for r in range(ny):
            mask = (entries[1] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r+m
            num_feat = m         
       
            B = np.zeros(num_sampl)
            M_entr1 = np.eye(m)
            for i in range(m):
                M_entr1[i, i] *= np.sqrt(2*mu*lam*np.sum(Z[i]**2))
        
            m_arr = np.array(range(m))
            M_entr2 = X[m_arr[np.newaxis,:], (entries_r[0])[:, np.newaxis]]*Z[m_arr[np.newaxis,:], (entries_r[2])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r+mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            init_time = time.time()
            
            res = scipy.sparse.linalg.lsmr(M, B)
            YT[r] = res[0]
        res_time = time.time()
        Y = YT.T
        if verbose:
            print("Y iteration minimization time: {}".format(res_time - init_time))
        return Y
        
    
    def reg_fast_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, verbose = False, lam = 1.0):
        entries = np.asarray(entries_a[:3, :], dtype = int)
        entries_val = entries_a[3]
        nx, ny, nz = n
        ZT = Z.T
        for r in range(nz):
            mask = (entries[2] == r)
            entries_r = entries[:, mask]
            entries_val_r = entries_val[mask]
            num_entries_r = entries_r.shape[1]
            num_sampl = num_entries_r+m
            num_feat = m         
       
            B = np.zeros(num_sampl)
            M_entr1 = np.eye(m)
            for i in range(m):
                M_entr1[i, i] *= np.sqrt(2*mu*lam*np.sum(Y[i]**2))
        
            m_arr = np.array(range(m))
            M_entr2 = Y[m_arr[np.newaxis,:], (entries_r[1])[:, np.newaxis]]*X[m_arr[np.newaxis,:], (entries_r[0])[:, np.newaxis]]
            
        
            B[m : ] = entries_val_r+mu*u[mask]
            
            M = np.vstack((M_entr1, M_entr2))
            init_time = time.time()
            
            res = scipy.sparse.linalg.lsmr(M, B)
            ZT[r] = res[0]
        res_time = time.time()
        Z = ZT.T
        if verbose:
            print("Z iteration minimization time: {}".format(res_time - init_time))
        return Z
    
    

    def run_minimization(
        X, Y, Z, n, m, entries_xyz,
        num_entries, test_entries, mu = 1.0,
        tau = 0.1, max_global_iter = 8, 
        max_iter = 1000,  verbose = True, lam = 1.0
    ):
        mu = mu
        nu = 1.0
        
        global_iter = 0
        it = 0

        coef = 10.0
        power = 0.0
        start = time.time()
        u  = np.zeros(num_entries)
        entries_a = am.from_dict_to_arr(entries_xyz, num_entries)
        
        # columns = ['iter', 'obj_val', 'obs_err', 'test_err', 'nuc_norm', 'time', 'mu']
        comp_log = np.zeros((max_iter, 7))

        while ((global_iter < max_global_iter) and (it<max_iter)):
            score, _, _ = am.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)

            small_step = 0
#             entries_a = am.from_dict_to_arr(entries_xyz, num_entries)

            stop_condition = False 
            new_progress = 0
            while ( not stop_condition):
                progress = new_progress
                X = am.reg_sp2_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, verbose = False, lam = lam)
                Y = am.reg_sp2_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, verbose = False, lam = lam)
                Z = am.reg_sp2_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                new_score, err_obs, nuc_norm = am.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
                new_progress = score - new_score
                score = new_score
                print("Score = {}, progress = {}, err = {}, nuc.norm = {}".format(score, new_progress, err_obs, nuc_norm))
                small_step+=1
                err1 = am.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
                print('eval_error_direct %f' % err1)

                current_time = time.time()-start
                
                comp_log[it] = np.array([it+1, score, err_obs, err1, nuc_norm, current_time, mu])
                it += 1

                stop_condition = (small_step>5+10*global_iter) or (it>=max_iter) or \
                				 ( (small_step >= 5) and \
                				 ( (progress>=1.2*new_progress) or (new_progress<0.01*np.abs(score/(global_iter+1))) ) )
            if (lam>0):    
                X, Y, Z = am.fix_components(X, Y, Z, n, m)
            print("Minimization completed")
            u_new, err = am.compute_adjust_sp(X, Y, Z, n, m, mu, u, entries_a)
            print("Parameters are: mu = {}, nu = {}, err = {}".format(mu, nu, err))
            if (err < nu):
                if (lam>0.0):
                  u = u_new
                  power += 1
                  print("u changed {} times".format(power))
            else: 
                mu = mu*tau
                power = 1
                global_iter += 1
                print("Scale changed. New scale: {}".format(mu))
            nu = coef*mu**(0.1+0.2*power)
            #print(time.time() - start)
            #print(aul_f(X_0, Y_0, Z_0, n, m1, u, mu, tensor, entries = entries_xyz))
            #ten = compute_tensor(X_0, Y_0, Z_0, n, m1)
            #prog_1[step//5] = np.sqrt(np.sum((ten - tensor)**2))
            #print('F2 norm %f' % prog_1[step//5])
            #err1 = am.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
            #print('eval_error_direct %f' % err1)

        log_dataframe = pd.DataFrame(
            comp_log, columns = ['iter', 'obj_val', 'obs_err', 'test_err', 'nuc_norm', 'time', 'mu']
        )
        return log_dataframe, (X, Y, Z)


    def run_min_balanced(X, Y, Z, n, m, entries_xyz, num_entries, test_entries, mu = 1.0,
                         tau = 0.1, max_global_iter = 30, max_iter = 1000,  verbose = True, lam = 1.0):
        mu = mu
        nu = 1.0
        
        global_iter = 0
        it = 0

        coef = 10.0
        power = 0.0
        start = time.time()
        u  = np.zeros(num_entries)
        entries_a = am.from_dict_to_arr(entries_xyz, num_entries)
        
        # columns = ['iter', 'obj_val', 'obs_err', 'test_err', 'nuc_norm', 'time', 'mu']
        comp_log = np.zeros((max_iter, 7))

        while ((global_iter < max_global_iter) and (it<max_iter)):
            score, nuc_norm, err_obs = am.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
            nu = err_obs/1.5
            small_step = 0
#             entries_a = am.from_dict_to_arr(entries_xyz, num_entries)

            stop_condition = False 
            new_progress = 0
            
            
            while ( not stop_condition):
                progress = new_progress
                X = am.reg_sp2_update_x(X, Y, Z, n, m, u, mu, entries_a, num_entries, verbose = False, lam = lam)
                Y = am.reg_sp2_update_y(X, Y, Z, n, m, u, mu, entries_a, num_entries, verbose = False, lam = lam)
                Z = am.reg_sp2_update_z(X, Y, Z, n, m, u, mu, entries_a, num_entries, lam = lam)
                if (lam>0):    
                    X, Y, Z = am.fix_components(X, Y, Z, n, m)
                new_score, err_obs, nuc_norm = am.aul_f_sp(X, Y, Z, n, m, u, mu, entries = entries_a)
                new_progress = score - new_score
                score = new_score
                print("Score = {}, progress = {}, err = {}, nuc.norm = {}".format(score, new_progress, err_obs, nuc_norm))
                small_step+=1
                err1 = am.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
                print('eval_error_direct %f' % err1)

                current_time = time.time()-start
                
                comp_log[it] = np.array([it+1, score, err_obs, err1, nuc_norm, current_time, mu])
                it += 1

                stop_condition = ((small_step>5+10*global_iter) 
                                  or (it>=max_iter) 
                                  or ((small_step >= 5) 
                                      and ((progress>=1.2*new_progress) or (new_progress<0.01*np.abs(score/(global_iter+1)))))
                                 )
            print("Global step completed")
            #update u and mu
            u_new, err = am.compute_adjust_sp(X, Y, Z, n, m, mu, u, entries_a)
            print("Old parameters were: mu = {}, nu = {}".format(mu, nu))
            if (err_obs > nu) and (power<6):
                if (lam>0.0):
                    u = u_new
                    power += 1
                    print("u changed {} times".format(power))
            else: 
                mu = 0.5*err_obs/(nuc_norm*(global_iter+1)**1.3)
                power = 1
                global_iter += 1
                print("Scale changed. New scale: {}".format(mu))
                nu = err*5
            print(f"New parameters are: mu = {mu}, nu = {nu}")
            #print(time.time() - start)
            #print(aul_f(X_0, Y_0, Z_0, n, m1, u, mu, tensor, entries = entries_xyz))
            #ten = compute_tensor(X_0, Y_0, Z_0, n, m1)
            #prog_1[step//5] = np.sqrt(np.sum((ten - tensor)**2))
            #print('F2 norm %f' % prog_1[step//5])
            #err1 = am.eval_error_direct_fast(X, Y, Z, n, m, test_entries)
            #print('eval_error_direct %f' % err1)

        log_dataframe = pd.DataFrame(comp_log, columns = ['iter', 'obj_val', 'obs_err', 'test_err', 'nuc_norm', 'time', 'mu'])
        return log_dataframe, (X, Y, Z)