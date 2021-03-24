#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cvxpy
import time
import numpy as np
import multiprocessing as mp
from tqdm import tqdm



import io_utils
import kernels
import algos
from importlib import reload
reload(kernels)
reload(io_utils)
reload(algos)

np.random.seed(42)

N_PROCESS = 6

SAVE = True

KERNEL = "SS"

L =3

TEST_KERNEL = True

def pooled_kernel_compute(exp, kern, N_PROCESS):
    kernels_wd_train = []

    with mp.Pool(N_PROCESS) as pool:
        # Compute kernels


        for n in range(3):
            print("Processing dataset {}".format(n))
            n_train = exp.raw[n].shape[0]

            t = time.time()
            args = [(exp.raw[n][i], exp.raw[n][j]) for i in range(n_train) for j in range(i, n_train) ]
            result = pool.map(kern.computeKernel, args)
            e = time.time()
            print("Kernel computation took {:.2f} seconds".format(e-t))

            count = 0
            K = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(i, n_train):
                    K[i,j] = result[count]
                    count +=1

            for i in range(n_train):
                for j in range(0, i):
                    K[i,j] = K[j,i]

            kernels_wd_train.append(K.copy())

    return kernels_wd_train

def pooled_kernel_test_compute(exp, kern, N_PROCESS):
    kernels_wd_train = []

    with mp.Pool(N_PROCESS) as pool:
        # Compute kernels

        for n in range(3):
            print("Processing dataset {}".format(n))
            n_train = exp.raw[n].shape[0]
            n_test = exp.raw_test[n].shape[0]

            t = time.time()
            args = [(exp.raw[n][i], exp.raw_test[n][j]) for i in range(n_train) for j in range(n_test) ]
            result = pool.map(kern.computeKernel, args)
            e = time.time()
            print("Kernel computation took {:.2f} seconds".format(e-t))

            count = 0
            K = np.zeros((n_train, n_test))
            for i in range(n_train):
                for j in range(n_test):
                    K[i,j] = result[count]
                    count +=1

            kernels_wd_train.append(K.copy())

    return kernels_wd_train

for W in [0.2]:

    if KERNEL == "WD":
        kernel_name = KERNEL + "_" + str(L)
    elif KERNEL == "SS":
        kernel_name = KERNEL + "_" + str(L) + "_" + "0" + str(10 * W)

    exp = io_utils.Experiment()

    try:
        training_kernels = exp.load("kernels/{}".format(kernel_name))
    except OSError:
        print("Kernels are not precomputed. Computing...")
        exp.create_new_experiment()
        if KERNEL == "WD":
            kern = kernels.WeightedDegreeKernel(L)
        elif KERNEL == "SS":
            kern = kernels.SubstringKernel(L, W)
        training_kernels = pooled_kernel_compute(exp, kern, N_PROCESS)
        if SAVE:
            print("Saving kernel: {W} {L}")
            exp.save("kernels/{}".format(kernel_name), training_kernels)
    
    if TEST_KERNEL:
        training_kernels = exp.load("kernels/{}".format(kernel_name))
        exp.create_new_experiment()
        exp.load_all_test_datasets()
        print(exp.raw)
        if KERNEL == "WD":
            kern = kernels.WeightedDegreeKernel(L)
        elif KERNEL == "SS":
            kern = kernels.SubstringKernel(L, W)
        test_kernels = pooled_kernel_test_compute(exp, kern, N_PROCESS)
        print("Saving kernel: {W} {L}")
        exp.save("kernels/{}".format(kernel_name), training_kernels, test_kernels)
