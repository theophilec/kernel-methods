import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time
import os 

np.random.seed(42)  # remove ?
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../data")
class Experiment:
    def __init__(self):
        self.labels = []
        self.raw = []
        self.raw_test = []
        self.feats = []
        self.feats_test = []

    def train_val_single_dataset(self, i, VALIDATION_RATIO = 0.2):
        tr = np.array(pd.read_csv(os.path.join(DATA_ROOT, 'Xtr{}.csv'.format(i)))["seq"])
        tr_feats = np.loadtxt(open(os.path.join(DATA_ROOT, "Xtr{}_mat100.csv".format(i)), "rb"), delimiter=" ")
        tr_labels = np.array(pd.read_csv(os.path.join(DATA_ROOT, 'Ytr{}.csv'.format(i)))["Bound"])

        tr_labels[tr_labels == 0] = -1

        return tr, tr_feats, tr_labels

    def load_all_test_datasets(self):
        for i in range(3):
            self.feats_test.append(np.loadtxt(open(os.path.join(DATA_ROOT, "Xte{}_mat100.csv".format(i)), "rb"), delimiter=" "))
            self.raw_test.append(np.array(pd.read_csv(os.path.join(DATA_ROOT, 'Xte{}.csv'.format(i)))["seq"]))

    def create_new_experiment(self):
        for i in range(3):
            r_tr, feats_tr, lbl_tr = self.train_val_single_dataset(i)

            self.raw.append(r_tr)
            self.labels.append(lbl_tr)
            self.feats.append(feats_tr)

    def load(self, prefix, with_test = False):
        kernels_trainval, kernels_test = [], []

        for i in tqdm(range(3)):
            kernels_trainval.append(np.loadtxt("{}_kernel_trainval_{}.txt".format(prefix, i)))

            if with_test:
                kernels_test.append(np.loadtxt("{}_train_test_{}.txt".format(prefix, i)))

            self.labels.append(np.loadtxt("{}_labels_trainval_{}.txt".format(prefix, i)))
            self.feats.append(np.loadtxt("{}_feats_trainval_{}.txt".format(prefix, i)))

        if with_test:
            return kernels_trainval, kernels_test
        else:
            return kernels_trainval

    def save(self, prefix, kernels_trainval, kernels_test=None):
        for i in tqdm(range(3)):
            np.savetxt('{}_kernel_trainval_{}.txt'.format(prefix, i), kernels_trainval[i])

            if kernels_test is not None:
                np.savetxt('{}_train_test_{}.txt'.format(prefix, i), kernels_test[i])
            np.savetxt('{}_labels_trainval_{}.txt'.format(prefix, i), self.labels[i])

            with open('{}_raw_trainval_{}.txt'.format(prefix, i), 'w') as f:
                f.write("\n".join(self.raw[i]))

            np.savetxt('{}_feats_trainval_{}.txt'.format(prefix, i), self.feats[i])

def train_val_dataset(exp):

    raw_train_1, feats_train_1, labels_train_1 = exp.train_val_single_dataset(0)
    raw_train_2, feats_train_2, labels_train_2 = exp.train_val_single_dataset(1)
    raw_train_3, feats_train_3, labels_train_3 = exp.train_val_single_dataset(2)

    all_train_raw = np.vstack((raw_train_1, raw_train_2, raw_train_3))

    all_train_feats = np.vstack((feats_train_1, feats_train_2, feats_train_3))

    all_train_labels = np.hstack((labels_train_1, labels_train_2, labels_train_3))


    return all_train_raw, all_train_feats, all_train_labels

def load_raw_test(i):
    test = np.array(pd.read_csv(os.path.join(DATA_ROOT, 'Xte{}.csv'.format(i)))["seq"])
    return test


def parse_output(predictions, filename):
    '''
        predictions : list of predictions
    '''
    predictions = np.sign(predictions)
    predictions[predictions == -1] = 0

    with open(filename, 'w') as f:
        f.write("Id,Bound\n")
        count = 0
        for i in range(predictions.shape[0]):
            f.write("{},{}\n".format(count, int(predictions[i])))
            count += 1

def pooled_kernel_compute(exp, kern, n_process=1):
    kernels_train = []

    with mp.Pool(n_process) as pool:
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

            kernels_train.append(K.copy())
        return kernels_train
