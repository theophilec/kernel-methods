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


# In[2]:


import io_utils
import kernels
import algos
from importlib import reload
reload(kernels)
reload(io_utils)
reload(algos)

np.random.seed(42)

N_PROCESS = 12

SAVE = True

KERNEL = "SS"

L = 3
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
            print("Saving kernel: {W} {L}")
            exp.save("kernels/{}".format(kernel_name), training_kernels)
