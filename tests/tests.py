import io_utils
import algos
import kernels

import os 
import numpy as np 
from tqdm import tqdm 
import shutil
# Create dataset 
print("Testing data preprocessing...")
exp = io_utils.Experiment()
exp.create_new_experiment()
exp.load_all_test_datasets()

# Gaussian kernel
print("Testing kernel computation...")
gaussian_kernel = kernels.GaussianKernel(0.1)
gaussian_train, gaussian_test = [], []
for i in tqdm(range(3)):
    gaussian_train.append(gaussian_kernel.computeVectorizedKernel(exp.feats[i], exp.feats[i]))
    gaussian_test.append(gaussian_kernel.computeVectorizedKernel(exp.feats[i], exp.feats_test[i]))

# Save/load
print("Testing save/load functions...")
os.mkdir('tests')
exp.save('tests/test', gaussian_train, gaussian_test)
loaded_gaussian_train, loaded_gaussian_test = exp.load('tests/test', with_test=True)
shutil.rmtree('tests')

# Test KRR
print("Testing learning algorithms...")

grid = np.logspace(-5, 1, 5)
krr = algos.KernelRidgeRegression()
best_idx_param = krr.cross_validate(exp, loaded_gaussian_train, grid)

for i in range(3):
    krr.fit(loaded_gaussian_train[i], exp.labels[i], grid[best_idx_param])
    preds = krr.predict(gaussian_test[i]) 

# Test SVM
grid = np.logspace(-5, 1, 2) #SVM is slow...
svm = algos.SVM()
best_idx_param = svm.cross_validate(exp, loaded_gaussian_train, grid)
for i in range(3):
    svm.fit(loaded_gaussian_train[i], exp.labels[i], grid[best_idx_param])
    predictions = svm.predict(gaussian_test[i])