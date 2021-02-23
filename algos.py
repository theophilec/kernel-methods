import numpy as np
import matplotlib.pyplot as plt 
from abc import ABC
import cvxpy as cp

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

    def searchBestParameters(self,training_kernel, training_labels, val_kernel, val_labels, grid):
        '''
            grid : a list of floats representing all regularization parameters
        '''
        all_errors = []

        for reg_param in grid:
            self.fit(training_kernel, training_labels, reg_param)
            error = self.evaluatePerformance(val_kernel, val_labels, verbose = False)
            all_errors.append(error)
        
        plt.plot(grid, all_errors, label="Validation error")
        plt.xscale('log')
        plt.legend(loc='upper left')
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.show()

        return all_errors

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


