"""Cross validation for string kernels.

If kernels are not already computed in `kernels/`,
compute.
"""
import sys
import multiprocessing as mp
import time
from importlib import reload

import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('code')
import algos
import io_utils
import kernels

np.random.seed(42)

N_PROCESS = 7

SAVE = True

KERNEL = "SS"

for L in [3, 7, 9]:
    for W in [0.1, 0.3, 0.5, 0.7]:

        if KERNEL == "WD":
            kernel_name = KERNEL + "_" + str(L)
        elif KERNEL == "SS":
            kernel_name = KERNEL + "_" + str(L) + "_" + "0" + str(int(10 * W))

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
            training_kernels = io_utils.pooled_kernel_compute(exp, kern, N_PROCESS)
            if SAVE:
                print("Saving kernels.")
                exp.save("kernels/{}".format(kernel_name), training_kernels)

        TRIALS = 5
        LAMBDA_LOW = -3
        LAMBDA_HIGH = 3

        lambda_vals = np.logspace(LAMBDA_LOW, LAMBDA_HIGH, TRIALS)

        krr = algos.KernelRidgeRegression()
        best_id, errors = krr.cross_validate(
            exp, training_kernels, lambda_vals, verbose=False, fit_bias=False
        )
        if best_id == 0 or best_id == TRIALS - 1:
            print("Warning: HP search hit edge of grid.")
        print(f"KRR {kernel_name} {lambda_vals[best_id]} {1 - errors}")
