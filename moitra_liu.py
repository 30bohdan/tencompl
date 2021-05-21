import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

class ML_completion:
    
    def __init__(
        self, n, rank, num_samples, save_file, randominit = True, 
        correlated = False, noisy = False, noise_size = 0.1,
        num_iter = 400, num_runs = 1, threshold = 10**(-6)
    ): 
        self.n = n
        self.rank = rank
        self.num_samples = num_samples
        self.save_file = save_file
        self.randominit = randominit
        self.correlated = correlated
        self.noisy = noisy
        self.noise_size = noise_size
        self.num_iter = num_iter
        self.num_runs = num_runs
        self.threshold = threshold

    def gen(self, nx, ny, nz, rank):
        coeffs = np.ones(rank)
        x_vecs = np.random.normal(0,1,(rank,nx))
        y_vecs = np.random.normal(0,1,(rank,ny))
        z_vecs = np.random.normal(0,1,(rank,nz))
        return (coeffs, x_vecs,y_vecs,z_vecs)

    #generate random correlated tensor
    def gen_biased(self, nx, ny, nz, rank):
        coeffs = np.zeros(rank)
        x_vecs = np.zeros((rank,nx))
        y_vecs = np.zeros((rank,ny))
        z_vecs = np.zeros((rank,nz))
        for i in range(rank):
            coeffs[i] = 0.5**i
            if(i==0):
                x_vecs[i] = np.sqrt(nx) *normalize(np.random.normal(0,1,nx))
                y_vecs[i] = np.sqrt(ny) *normalize(np.random.normal(0,1,ny))
                z_vecs[i] = np.sqrt(nz) *normalize(np.random.normal(0,1,nz))
            else:
                x_vecs[i] = np.sqrt(nx) *normalize(np.random.normal(0,0.5,nx) + x_vecs[0])
                y_vecs[i] = np.sqrt(ny) *normalize(np.random.normal(0,0.5,ny) + y_vecs[0])
                z_vecs[i] = np.sqrt(nz) *normalize(np.random.normal(0,0.5,nz) + z_vecs[0])
        return (coeffs, x_vecs,y_vecs,z_vecs)

    #evaluate tensor given coordinates
    def T(self, i,j,k, coeffs, x_vecs, y_vecs, z_vecs):
        ans = 0
        for a in range(self.rank):
            ans += coeffs[a] * x_vecs[a][i] * y_vecs[a][j] * z_vecs[a][k]
        return ans
    
    
    #sample observations, a is num_samples
    #returns 3 lists of coordinates
    def sample(self, a, nx, ny, nz):
        samples = np.random.choice(nx*ny*nz, a, replace=False)
        x_coords = samples%nx
        y_coords = (((samples - x_coords)/nx)%ny).astype(int)
        z_coords = (((samples - nx*y_coords - x_coords)/(nx*ny))%nz).astype(int)
        return (x_coords, y_coords, z_coords)
    #Given samples and tensor T, construct dictionary x_dict that stores the observations

    def fill(self, x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict):
        num_samples = x_coords.size
        for i in range(num_samples):
        #For x_dict coordinates are in order x,y,z
            if(x_coords[i] in x_dict.keys()):
                if(y_coords[i] in x_dict[x_coords[i]].keys()):
                    if(z_coords[i] in x_dict[x_coords[i]][y_coords[i]].keys()):
                        pass
                    else:
                        x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = self.T(x_coords[i] , 
                                                                y_coords[i] , z_coords[i], coeffs,x_vecs, y_vecs, z_vecs)
                else:
                    x_dict[x_coords[i]][y_coords[i]] = {}
                    x_dict[x_coords[i]][y_coords[i]][z_coords[i]]= self.T(x_coords[i] , 
                                                                     y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs)
            else:
                x_dict[x_coords[i]] = {}
                x_dict[x_coords[i]][y_coords[i]] = {}
                x_dict[x_coords[i]][y_coords[i]][z_coords[i]] = self.T(x_coords[i] , 
                                                                y_coords[i] , z_coords[i], coeffs, x_vecs, y_vecs, z_vecs)
    
    def normalize(v):
        u = v/np.linalg.norm(v)
        return u
    
    #given rxn array, output orthonormal basis
    def orthonormalize(V):
        a = len(V)
        b = len(V[0])
        for i in range(a):
            for j in range(i):
                V[i] = V[i] - np.dot(V[i],V[j])*V[j]
            V[i] = ML_completion.normalize(V[i])
        return V

    #implicit sparse matrix multiplication where M is stored as a dictionary
    def mult(M,v,nu):
        u = np.zeros(nu)
        for coord1 in M.keys():
            for coord2 in M[coord1].keys():
                u[coord1] += M[coord1][coord2] * v[coord2]
        return u

    #Compute initial subspace estimates
    def initialization(self, x_dict, p, r, nz):
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
    #Unfold and perform matrix completion via altmin
    # def matrix_altmin(V_x, V_yz):
    #     #Solve for next iteration of x
    #     lsq_solution = []
    #     for i in range(n):
    #         features = []
    #         target = []
    #         for y_coord in x_dict[i].keys():
    #             for z_coord in x_dict[i][y_coord].keys():
    #                 features.append(V_yz[n*y_coord + z_coord])
    #                 target.append(x_dict[i][y_coord][z_coord])
    #         features = np.array(features)
    #         target = np.array(target)
    #         reg = LinearRegression(fit_intercept = False).fit(features, target)
    #         lsq_solution.append(reg.coef_)
    #     x_solution = np.array(lsq_solution)
    #     #Solve for next iteration of yz
    #     lsq_solution2 = []
    #     for i in range(n):
    #         for j in range(n):
    #             features = []
    #             target = []
    #             if i in y_dict.keys() and j in y_dict[i].keys():
    #                 for x_coord in y_dict[i][j].keys():
    #                     features.append(x_solution[x_coord])
    #                     target.append(y_dict[i][j][x_coord])
    #                 features = np.array(features)
    #                 target = np.array(target)
    #                 reg = LinearRegression(fit_intercept = False).fit(features, target)
    #                 lsq_solution2.append(reg.coef_)
    #             else:
    #                 lsq_solution2.append(np.zeros(r))
    #     newV_x = x_solution
    #     newV_yz =np.array(lsq_solution2)
    #     return(newV_x, newV_yz)

    #Normalized MSE for unfolded matrix completion
    def eval_error_matrix(self, V_x,V_yz, nx, ny, nz, r):
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

    #altmin for naive tensor powering
    def power_altmin(V_x, V_y, V_z , x_dict, nx, ny, nz):
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
        return(lsq_solution)

    #Normalized MSE for naive tensor powering
    def eval_error_direct(self, V_x,V_y,V_z, x_dict, nx, ny, nz):
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

    #altmin for our algorithm
    def subspace_altmin(self, V_x, V_y, V_z , x_dict, nx, ny, nz):
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
            np.savetxt(self.save_file, features, delimiter=",")
            reg = LinearRegression(fit_intercept = False).fit(features,target)
            lsq_solution.append(reg.coef_)
        lsq_solution = np.transpose(np.array(lsq_solution))
        svd = TruncatedSVD(n_components=self.rank)
        svd.fit(lsq_solution)
        return(np.transpose(svd.components_))
        
    #Normalized MSE for our algorithm
    def eval_error_subspace(self, V_x,V_y,V_z, x_dict, coeffs, x_vecs,y_vecs, z_vecs):
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
        #print(reg.score(features, target))
        #print(solution_coeffs)
        #Evaluate RMS error
        num_trials = 1000
        total_error = 0
        total_norm = 0
        for i in range(num_trials):
            x = np.random.randint(nx)
            y = np.random.randint(ny)
            z = np.random.randint(nz)
            part = np.tensordot(V_x[x], V_y[y], axes = 0).flatten()
            feature = np.tensordot(part, V_z[z], axes = 0).flatten()
            prediction = np.dot(feature, solution_coeffs)
            true_val = self.T(x,y,z, coeffs, x_vecs,y_vecs, z_vecs) 
            total_norm += np.square(true_val)
            total_error += np.square(prediction - true_val)
        return np.sqrt(total_error/total_norm)

    #Normalized MSE for our algorithm
    def eval_error_subspace1(self, V_x,V_y,V_z, x_dict, test_entries):
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
        #print(reg.score(features, target))
        #print(solution_coeffs)
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
    def recover_frame(idx_frame, sol_coeffs, V_x, V_y, V_z, n):
        nx, ny, nz = n
        pred = np.empty((ny, nz), dtype=np.float32)
        for i in range(ny):
            for j in range(nz):
                part = np.tensordot(V_x[idx_frame], V_y[i], axes = 0).flatten()
                feature = np.tensordot(part, V_z[j], axes = 0).flatten()
                pred[i, j] = np.dot(feature, sol_coeffs)
        return pred

    def run(self):
        nx, ny, nz = self.n
        rank = self.rank
        r = rank
        num_samples = self.num_samples
        p = min(float(num_samples/ (nx*ny*nz)), 1)
        curr_error = 1.0
        error = []
        if(self.correlated):
            coeffs, x_vecs,y_vecs,z_vecs = self.gen_biased(nx, ny, nz, rank)
        else:
            coeffs, x_vecs,y_vecs,z_vecs = self.gen(nx, ny, nz, rank)
        x_coords,y_coords,z_coords = self.sample(num_samples, nx, ny, nz)
        #x_dict,y_dict, z_dict each stores all observed entries
        #x_dict has coordinates in order x,y,z
        #y_dict has coordinates in order y,z,x
        #z_dict has coordinates in order z,x,y
        x_dict = {}
        y_dict = {}
        z_dict = {}
        self.fill(x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict)
        self.fill(y_coords, z_coords, x_coords, coeffs, y_vecs, z_vecs, x_vecs, y_dict)
        self.fill(z_coords, x_coords, y_coords, coeffs, z_vecs, x_vecs, y_vecs, z_dict)
        #Add Noise
        if(self.noisy):
            for x_coord in x_dict.keys():
                for y_coord in x_dict[x_coord].keys():
                    for z_coord in x_dict[x_coord][y_coord].keys():
                        x_dict[x_coord][y_coord][z_coord] += np.random.normal(0,self.noise_size)
                        y_dict[y_coord][z_coord][x_coord] += np.random.normal(0,self.noise_size)
                        z_dict[z_coord][x_coord][y_coord] += np.random.normal(0,self.noise_size)
        #Initialization
        if(self.randominit):
            V_x = np.random.normal(0,1,(r,nx))
            V_y = np.random.normal(0,1,(r,ny))
            V_z = np.random.normal(0,1,(r,nz))
            V_x = ML_completion.orthonormalize(V_x)
            V_y = ML_completion.orthonormalize(V_y)
            V_z = ML_completion.orthonormalize(V_z)
            V_x = np.transpose(V_x)
            V_y = np.transpose(V_y)
            V_z = np.transpose(V_z)
        else:
            V_x = np.transpose(self.initialization(y_dict, p, r, nx))
            V_y = np.transpose(self.initialization(z_dict, p, r, ny))
            V_z = np.transpose(self.initialization(x_dict, p, r, nz))
        #For unfolding and matrix completion
        V_xmat = np.random.normal(0,1, (r,nx))
        V_yzmat = np.random.normal(0,1, (r, ny*nz))
        V_xmat = ML_completion.orthonormalize(V_xmat)
        V_yzmat = ML_completion.orthonormalize(V_yzmat)
        V_xmat = np.transpose(V_xmat)
        V_yzmat = np.transpose(V_yzmat)
        V_x2 = np.copy(V_x)
        V_y2 = np.copy(V_y)
        V_z2 = np.copy(V_z)
        print(nx, ny, nz)
        print(r)
        print(self.num_samples)
        which_alg = "Subspace Powering"
        #AltMin Steps
        for i in range(self.num_iter):
            print(i)
            if(which_alg == "Matrix Alt Min" or which_alg == "all"):
                print("Matrix Alt Min")
                V_xmat, V_yzmat = self.matrix_altmin(V_xmat, V_yzmat)
                curr_error = self.eval_error_matrix(V_xmat, V_yzmat)
                print(curr_error)
                error.append(curr_error)
            if(which_alg == "Tensor Powering" or which_alg == "all"):
                print("Tensor Powering")
                if(curr_error > self.threshold):
                    V_x = self.power_altmin(V_x,V_y,V_z, x_dict, nx, ny, nz)
                    V_y = self.power_altmin(V_y,V_z,V_x, y_dict, ny, nz, nx)
                    V_z = self.power_altmin(V_z,V_x,V_y, z_dict, nz, nx, ny)
                    curr_error = eval_error_direct(V_x,V_y,V_z,x_dict)
                print(curr_error)
                error.append(curr_error)
            if(which_alg == "Subspace Powering" or which_alg == "all"):
                print("Subspace Powering")
                if(curr_error > self.threshold):
                    V_x2 = self.subspace_altmin(V_x2,V_y2,V_z2, x_dict, nx, ny, nz)
                    V_y2 = self.subspace_altmin(V_y2,V_z2,V_x2, y_dict, ny, nz, nx)
                    V_z2 = self.subspace_altmin(V_z2,V_x2,V_y2, z_dict, nz, nx, ny)
                    #curr_error_dir = self.eval_error_direct(V_x,V_y,V_z,x_dict, nx, ny, nz)
                    curr_error = self.eval_error_subspace(V_x2,V_y2,V_z2, x_dict, coeffs, x_vecs,y_vecs, z_vecs)
                print(curr_error)
                #print('dir-err %f' % curr_error_dir)
                error.append(curr_error)
        all_errors.append(error)
        to_save = np.transpose(np.array(all_errors))
        avg_errors = np.mean(to_save, axis = 0)
        np.savetxt(self.save_file, to_save, delimiter=",")



    def run_for_ten(self, V_x, V_y, V_z, entries, test_entries):
        nx, ny, nz = self.n
        rank = self.rank
        r = rank
        num_samples = self.num_samples

        entries_i = np.asarray(entries[:3, :], dtype = int)
        #coeffs = np.repeat(1, num_samples)
        p = min(float(num_samples/ (nx*ny*nz)), 1)
        curr_error = 1.0
        error = []

        if(self.correlated):
            coeffs, x_vecs,y_vecs,z_vecs = self.gen_biased(nx, ny, nz, rank)
        else:
            coeffs, x_vecs,y_vecs,z_vecs = self.gen(nx, ny, nz, rank)
        x_coords,y_coords,z_coords = (entries_i[0], entries_i[1], entries_i[2])

        #x_dict,y_dict, z_dict each stores all observed entries
        #x_dict has coordinates in order x,y,z
        #y_dict has coordinates in order y,z,x
        #z_dict has coordinates in order z,x,y
        x_dict = {}
        y_dict = {}
        z_dict = {}
        self.fill(x_coords, y_coords, z_coords, coeffs, x_vecs, y_vecs, z_vecs, x_dict)
        self.fill(y_coords, z_coords, x_coords, coeffs, y_vecs, z_vecs, x_vecs, y_dict)
        self.fill(z_coords, x_coords, y_coords, coeffs, z_vecs, x_vecs, y_vecs, z_dict)
        for i in range(num_samples):
            x_coord,y_coord,z_coord = (int(entries_i[0, i]), entries_i[1, i], entries_i[2, i])
            x_dict[x_coord][y_coord][z_coord] =  entries[3, i]
            y_dict[y_coord][z_coord][x_coord] = entries[3, i]
            z_dict[z_coord][x_coord][y_coord] = entries[3, i]

        V_x = ML_completion.orthonormalize(V_x)
        V_y = ML_completion.orthonormalize(V_y)
        V_z = ML_completion.orthonormalize(V_z)
        V_x = np.transpose(V_x)
        V_y = np.transpose(V_y)
        V_z = np.transpose(V_z)

        V_xmat = np.random.normal(0,1, (r,nx))
        V_yzmat = np.random.normal(0,1, (r, ny*nz))
        V_xmat = ML_completion.orthonormalize(V_xmat)
        V_yzmat = ML_completion.orthonormalize(V_yzmat)
        V_xmat = np.transpose(V_xmat)
        V_yzmat = np.transpose(V_yzmat)
        V_x2 = np.copy(V_x)
        V_y2 = np.copy(V_y)
        V_z2 = np.copy(V_z)
        print(nx, ny, nz)
        print(r)
        print(self.num_samples)
        which_alg = "Subspace Powering"
        #AltMin Steps
        for i in range(self.num_iter):
            print(i)
            if(which_alg == "Matrix Alt Min" or which_alg == "all"):
                print("Matrix Alt Min")
                V_xmat, V_yzmat = self.matrix_altmin(V_xmat, V_yzmat)
                curr_error = self.eval_error_matrix(V_xmat, V_yzmat)
                print(curr_error)
                error.append(curr_error)
            if(which_alg == "Tensor Powering" or which_alg == "all"):
                print("Tensor Powering")
                if(curr_error > self.threshold):
                    V_x = self.power_altmin(V_x,V_y,V_z, x_dict, nx, ny, nz)
                    V_y = self.power_altmin(V_y,V_z,V_x, y_dict, ny, nz, nx)
                    V_z = self.power_altmin(V_z,V_x,V_y, z_dict, nz, nx, ny)
                    curr_error = eval_error_direct(V_x,V_y,V_z,x_dict)
                print(curr_error)
                error.append(curr_error)
            if(which_alg == "Subspace Powering" or which_alg == "all"):
                print("Subspace Powering")
                if(curr_error > self.threshold):
                    V_x2 = self.subspace_altmin(V_x2,V_y2,V_z2, x_dict, nx, ny, nz)
                    V_y2 = self.subspace_altmin(V_y2,V_z2,V_x2, y_dict, ny, nz, nx)
                    V_z2 = self.subspace_altmin(V_z2,V_x2,V_y2, z_dict, nz, nx, ny)
                    #curr_error_dir = self.eval_error_direct(V_x,V_y,V_z,x_dict, nx, ny, nz)
                    curr_error = self.eval_error_subspace1(V_x2,V_y2,V_z2, x_dict, test_entries)
                print(curr_error)
                #print('dir-err %f' % curr_error_dir)
                error.append(curr_error)
#         all_errors.append(error)
#         to_save = np.transpose(np.array(all_errors))
#         avg_errors = np.mean(to_save, axis = 0)
#         np.savetxt(self.save_file, to_save, delimiter=",")
        return V_x2, V_y2, V_z2



