import numpy as np 
from numba import jit 
from abc import ABC


class LinearKernel():

    def computeVectorizedKernel(self, X,Y):
        return X @ Y.T

class GaussianKernel():
    def __init__(self, sigma):
        self.sigma = sigma 

    def computeVectorizedKernel(self, X, Y):
        dist = np.linalg.norm((X[:,None]-Y), axis = 2)**2
        K = np.exp(-dist/self.sigma**2)

        return K
    
class SubstringKernel():

    def __init__(self, n, lambd):
        self.n = n
        self.lambd = lambd 

    @staticmethod
    @jit(nopython=True)
    def _computeKernel(args, lambd, n ):
        '''
            args is a tuple (s1, s2) where s1 and s2 are the two strings
        '''

        s1, s2 = args[0], args[1]
        n1, n2 = len(s1), len(s2)
        prev = np.zeros((n1, n2))
        nxt = np.zeros((n1, n2))
        K = np.zeros(n)
        for i in range(n1):
            for j in range(n2):
                if s1[i] == s2[j]:
                    prev[i,j] = lambd**2
                    K[0] += prev[i,j]

        K[0] /= lambd ** 2
        for l in range(1, n):
            B = np.zeros((n1, n2))

            for i in range(0, n1):
                for j in range(0,n2):
                    B[i,j] = prev[i,j]
                    if i != 0: # Problem w/ -1 indexing in Python....
                        B[i,j] += lambd * B[i-1,j]
                    if j!=0:
                        B[i,j] += lambd * B[i,j-1]
                    if i!=0 and j!=0:
                        B[i,j] -= lambd **2 * B[i-1, j-1]
                        
                    if s1[i] == s2[j] and i-1 !=0 and j-1 != 0 :
                        nxt[i,j] = lambd ** 2 * B[i-1,j-1]
                        K[l] += nxt[i,j]

            K[l] /= lambd ** (2*(l+1))
            prev = nxt.copy()
        return K[-1]
    
    def computeKernel(self, args):
        return self._computeKernel(args, self.lambd, self.n)
