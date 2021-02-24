import numpy as np
import matplotlib.pyplot as plt 
from abc import ABC
import cvxpy as cp
from tqdm import tqdm 

''' 
Usage:
>>> krr = KernelRidgeRegression()
>>> krr.fit(X,y, reg_param = 0.1)
>>> y_pred = krr.predict(X_test)
''' 

class Algorithm(ABC):
    def __init__(self):
        self.alpha = None
        self.reg_param = 0

    def fit(self, kernel, labels, reg_param):
        '''
            kernel: nxn kernel matrix
            labels: nx1 numpy array
            reg_param: float (regularization parameter)
        '''
        pass
    
    def predict(self, matrix):
        pass

    def evaluatePerformance(self, matrix, labels, verbose = True):
        predictions = self.predict(matrix)
        error = np.sum(np.sign(predictions) != np.sign(labels)) / labels.shape[0]
        
        if verbose:
            print("Score : {:.3f}".format(1-error))
        
        return error 

    def searchBestParameters(self,training_kernel, training_labels, val_kernel, val_labels, grid, no_plot = True):
        '''
            grid : a list of floats representing all regularization parameters
        '''
        all_errors = []

        for reg_param in grid:
            self.fit(training_kernel, training_labels, reg_param)
            error = self.evaluatePerformance(val_kernel, val_labels, verbose = False)
            all_errors.append(error)
        
        if not no_plot:
            plt.plot(grid, all_errors, label="Validation error")
            plt.xscale('log')
            plt.legend(loc='upper left')
            plt.xlabel(r"$\lambda$", fontsize=16)
            plt.show()

        return all_errors
    
    def cross_validate(self, exp, kernels, grid, validation_ratio = 0.2, N_CROSS_VAL = 5):
        '''
            Performs cross validation: splits the dataset in N_VAL chunks and alternatively 
            uses the i-th chunk as the validation set and all others for training

            - experience (class Experience) 
            - kernels (list of numpy array) : list of 2000x2000 matrices K(x_i, x_j)
            - grid (numpy array) : array of parameters to try  
            Optional parameters
                - N_VAL (int) : Number of cross-validation slices
        '''

        n_tot = kernels[0].shape[0]
        
        n_train = int((1-validation_ratio) * n_tot)
        n_val = n_tot - n_train 
        
        n_grid = grid.shape[0]
        n_datasets = len(kernels)
        errors = np.zeros((n_datasets, n_grid))

        print("Cross validation with {} slices. Training set: {}, Validation set: {}".format(N_CROSS_VAL, n_train, n_val))

        for i in tqdm(range(N_CROSS_VAL)):
            for n in range(n_datasets):
                train = np.zeros((n_train, n_train))
                val = np.zeros((n_train, n_val))
                

                val[:i*n_val] = kernels[n][:i*n_val, i*n_val:(i+1)*n_val]
                val[i*n_val:] = kernels[n][(i+1)*n_val:, i*n_val:(i+1)*n_val]

                train[:i*n_val, :i*n_val] = kernels[n][:i*n_val, :i*n_val]

                if i != N_CROSS_VAL - 1:
                    train[i*n_val:, i*n_val:] = kernels[n][(i+1)*n_val:, (i+1)*n_val:]

                if i != 0 and i != N_CROSS_VAL - 1:
                    train[i*n_val:, :i*n_val] = kernels[n][(i+1)*n_val:, :i*n_val]
                    train[:i*n_val, i*n_val:] = kernels[n][:i*n_val, (i+1)*n_val:]
                
                lbl_train = np.zeros(n_train)
                
                if i == N_CROSS_VAL - 1:
                    lbl_train = exp.labels[n][:n_train]
                    lbl_val = exp.labels[n][n_train:]
                else:
                    lbl_train = np.concatenate((exp.labels[n][:i*n_val], exp.labels[n][(i+1)*n_val:]))
                    lbl_val = exp.labels[n][i*n_val:(i+1)*n_val]

                errors[n] += self.searchBestParameters(train, lbl_train, val, lbl_val, grid)
            
        best_idx = np.argmin(errors[0] + errors[1] + errors[2])

        print("Best parameter: {}".format(grid[best_idx]))

        return best_idx
        
class KernelRidgeRegression(Algorithm):

    def fit(self, training_kernel, training_labels, reg_param=0):
        n = training_kernel.shape[0]
        alpha = np.linalg.solve(training_kernel + reg_param * n * np.eye(n),training_labels)

        self.alpha = alpha 
        self.reg_param = reg_param
        
    def predict(self, mat):
        predictions = np.einsum('i, ij->j', self.alpha, mat)
        return predictions 

class SVM(Algorithm):

    def fit(self, training_kernel, training_labels, reg_param=0):
        n = training_kernel.shape[0]
        alpha = cp.Variable(n)

        constraints = []
        for i in range(n):
            constraints += [training_labels[i] * alpha[i] >=0 , training_labels[i] * alpha[i] <= 1/(2*reg_param*n)]
        
        prob = cp.Problem(cp.Minimize(-2 * alpha.T @ training_labels + cp.quad_form(alpha, training_kernel) ), constraints)
        prob.solve()

        self.alpha = alpha.value
        self.reg_param = reg_param

    def predict(self, mat):
        predictions = np.einsum('i, ij->j', self.alpha, mat)
        return predictions         


