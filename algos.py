import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
import cvxpy as cp
from tqdm import tqdm
from scipy.optimize import minimize
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
        self.beta = None

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

    def searchBestParameters(self,training_kernel, training_labels, val_kernel, val_labels, grid, no_plot = True, fit_bias=False):
        '''
            grid : a list of floats representing all regularization parameters
        '''
        all_errors = []

        for reg_param in grid:
            try:
                self.fit(training_kernel, training_labels, reg_param, fit_bias=fit_bias)
            except cp.error.DCPError:
                print(f"DCPError on lambda= {reg_param}, smallest eigenvalue {np.min(np.linalg.eigvals(training_kernel))}")
                quit()
            except cp.error.SolverError:
                print(f"SolverError on lambda= {reg_param}, smallest eigenvalue {np.min(np.linalg.eigvals(training_kernel))}")
                quit()
            error = self.evaluatePerformance(val_kernel, val_labels, verbose = False)
            all_errors.append(error)

        if not no_plot:
            plt.plot(grid, all_errors, label="Validation error")
            plt.xscale('log')
            plt.legend(loc='upper left')
            plt.xlabel(r"$\lambda$", fontsize=16)
            plt.show()

        return all_errors

    def cross_validate(self, exp, kernels, grid, N_CROSS_VAL = 5, fit_bias=False, verbose=True):
        '''
            Performs cross validation: splits the dataset in N_VAL chunks and alternatively
            uses the i-th chunk as the validation set and all others for training

            - experience (class Experience)
            - kernels (list of numpy array) : list of 2000x2000 matrices K(x_i, x_j)
            - grid (numpy array) : array of parameters to try
            Optional parameters
                - N_CROSS_VAL (int) : Number of cross-validation slices
        '''

        n_tot = kernels[0].shape[0]
        validation_ratio = 1 / N_CROSS_VAL
        n_train = int((1-validation_ratio) * n_tot)
        n_val = n_tot - n_train

        n_grid = grid.shape[0]
        n_datasets = len(kernels)
        errors = np.zeros((n_datasets, n_grid))

        if verbose:
            print("Cross validation with {} slices. Training set: {}, Validation set: {}".format(N_CROSS_VAL, n_train, n_val))
        idx = np.arange(n_tot)
        np.random.shuffle(idx)

        for i in tqdm(range(N_CROSS_VAL)):
            for n in range(n_datasets):
                # print(str(i) + " " + str(n))
                train_slice = np.concatenate((idx[:i*n_val], idx[(i+1)*n_val:]))
                val_slice = idx[i*n_val:(i+1)*n_val]

                train = np.zeros((n_train, n_train))
                val = np.zeros((n_train, n_val))

                # kernel_matrix = center_K(kernels[n])
                kernel_matrix = kernels[n]
                train = kernel_matrix[:, train_slice][train_slice]
                val = kernel_matrix[:, val_slice][train_slice]

                lbl_train = exp.labels[n][train_slice]
                lbl_val = exp.labels[n][val_slice]

                assert len(train.shape) == 2
                assert len(val.shape) == 2

                errors[n] += self.searchBestParameters(train, lbl_train, val, lbl_val, grid, fit_bias=fit_bias)

        best_idx = np.argmin(errors[0] + errors[1] + errors[2])

        if verbose:
            print("Best parameter: {}".format(grid[best_idx]))
            print("Score 1: {}".format(1-errors[0][best_idx]/N_CROSS_VAL))
            print("Score 2: {}".format(1-errors[1][best_idx]/N_CROSS_VAL))
            print("Score 3: {}".format(1-errors[2][best_idx]/N_CROSS_VAL))
        return best_idx, errors[:, best_idx] / N_CROSS_VAL

class KernelRidgeRegression(Algorithm):

    def fit(self, training_kernel, training_labels, reg_param=0, fit_bias = False):
        n = training_kernel.shape[0]
        if not fit_bias:
            alpha = np.linalg.solve(training_kernel + reg_param * n * np.eye(n),training_labels)
        else:
            H = np.linalg.inv(training_kernel+reg_param * n * np.eye(n))
            beta =  (-reg_param * H @ training_labels) @ np.ones(n) * 1/(1 + 1 / n * np.ones(n).T @ training_kernel @ H.T @ np.ones(n))
            alpha = np.linalg.solve(training_kernel + reg_param * n * np.eye(n), training_labels - beta * np.ones(n))
            # print(beta)
            self.beta = beta
        self.alpha = alpha
        self.reg_param = reg_param

    def predict(self, mat):
        predictions = np.einsum('i, ij->j', self.alpha, mat)
        if self.beta is not None:
            predictions += self.beta
        return predictions

class SVM(Algorithm):

    def fit(self, training_kernel, training_labels, reg_param=0, fit_bias=False):

        # assert np.all(np.linalg.eigvals(training_kernel) > 1e-8), f"Smallest eigenvalue: {np.min(np.linalg.eigvals(training_kernel))}"
        # print(f"Smallest eigenvalue for {reg_param}: {np.min(np.linalg.eigvals(training_kernel))}")

        n = training_kernel.shape[0]
        alpha = cp.Variable(n)

        constraints = []
        for i in range(n):
            constraints += [training_labels[i] * alpha[i] >=0 , training_labels[i] * alpha[i] <= 1/(2*reg_param*n)]

        prob = cp.Problem(cp.Minimize(-2 * alpha.T @ training_labels + cp.quad_form(alpha, training_kernel)), constraints)
        prob.solve()

        self.alpha = alpha.value
        self.reg_param = reg_param

    def predict(self, mat):
        predictions = np.einsum('i, ij->j', self.alpha, mat)
        return predictions

class KernelLogisticRegression(Algorithm):

    def fit(self, training_kernel, training_labels, reg_param=0, fit_bias=False):
        """ Use Newton's algorithm."""
        n = training_kernel.shape[0]

        def J(alpha):
            Kalpha = training_kernel @ alpha
            return logistic(training_labels * Kalpha).mean() + reg_param * alpha.T @ Kalpha

        def J_grad(alpha):
            Kalpha = training_kernel @ alpha
            P = logistic_prime(training_labels * Kalpha)
            return 1 / n * training_kernel @ (P * training_labels) + reg_param * Kalpha

        def J_hessian(alpha):
            Kalpha = training_kernel @ alpha
            W_diag = logistic_prime2(training_labels * Kalpha)
            return 1 / n * training_kernel @ (training_kernel * W_diag) + reg_param * training_kernel


        n = training_kernel.shape[0]

        solution = minimize(J, np.zeros(n), jac=J_grad, hess=J_hessian, method="Newton-CG")

        self.alpha = solution.x
        self.reg_param = reg_param

    def predict(self, mat):
        """


        predictions = p(1 | f(x)) - 0.5.
        If > 0: p(1|f(x)) > 0.5 -> y = 1 (predicted)
        """
        predictions = sigmoid(np.einsum('i, ij->j', self.alpha, mat)) - 0.5
        return predictions

def logistic(u):
    return np.log(1 + np.exp(-u))

def logistic_prime(u):
    return -sigmoid(-u)

def logistic_prime2(u):
    return sigmoid(u)*sigmoid(-u)

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def center_K(K):
    """
    Center kernel matrix.

    Reference: slide 72.
    """
    n = K.shape[0]
    rows = K.sum(axis=1)
    columns = K.sum(axis=0)
    return K - 1 / n * (rows + columns) - 1 / n ** 2 * K.sum()
