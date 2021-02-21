import pandas as pd 
import numpy as np 

np.random.seed(42)

class Experiment:
    def __init__(self):
        self.labels_train = []
        self.labels_val = []
        self.raw_train = []
        self.raw_val = []
        self.raw_test = []
        self.feats_train = []
        self.feats_val = []
        self.feats_test = []
    
    def train_val_single_dataset(self, i, VALIDATION_RATIO = 0.2):
        tr = np.array(pd.read_csv('data/Xtr{}.csv'.format(i))["seq"])
        tr_feats = np.loadtxt(open("data/Xtr{}_mat100.csv".format(i), "rb"), delimiter=" ")
        tr_labels = np.array(pd.read_csv('data/Ytr{}.csv'.format(i))["Bound"])

        n = tr.shape[0]
        tr_labels[tr_labels == 0] = -1

        shuffled_idx = np.arange(n)
        np.random.shuffle(shuffled_idx)

        raw_val = tr[shuffled_idx[:int(VALIDATION_RATIO*n)]]
        raw_train = tr[shuffled_idx[int(VALIDATION_RATIO*n):]]
        feats_val = tr_feats[shuffled_idx[:int(VALIDATION_RATIO*n)]]
        feats_train = tr_feats[shuffled_idx[int(VALIDATION_RATIO*n):]]
        labels_val = tr_labels[shuffled_idx[:int(VALIDATION_RATIO*n)]]
        labels_train = tr_labels[shuffled_idx[int(VALIDATION_RATIO*n):]]

        return raw_train, raw_val, feats_train, feats_val, labels_train, labels_val

    def load_all_test_datasets(self):
        for i in range(3):
            self.feats_test.append(np.loadtxt(open("data/Xte{}_mat100.csv".format(i), "rb"), delimiter=" "))
            self.raw_test.append(np.array(pd.read_csv('data/Xte{}.csv'.format(i))["seq"]))

    def create_new_experiment(self):
        for i in range(3):
            r_tr, r_val, feats_tr, feats_v, lbl_tr, lbl_val = self.train_val_single_dataset(i)
            self.raw_train.append(r_tr)
            self.raw_val.append(r_val)
            self.labels_train.append(lbl_tr)
            self.labels_val.append(lbl_val)
            self.feats_train.append(feats_tr)
            self.feats_val.append(feats_v)

    def load(self, prefix, with_test = False):
        kernels_train, kernels_val, kernels_test = [], [], []

        for i in range(3):
            kernels_val.append(np.loadtxt("{}_train_val_{}.txt".format(prefix, i)))
            kernels_train.append(np.loadtxt("{}_train_train_{}.txt".format(prefix, i)))
            if with_test:
                kernels_test.append(np.loadtxt("{}_train_test_{}.txt".format(prefix, i)))
            self.labels_train.append(np.loadtxt("{}_labels_train_{}.txt".format(prefix, i)))
            self.labels_val.append(np.loadtxt("{}_labels_val_{}.txt".format(prefix, i)))
            self.feats_train.append(np.loadtxt("{}_feats_train_{}.txt".format(prefix, i)))
            self.feats_val.append(np.loadtxt("{}_feats_val_{}.txt".format(prefix, i)))

        if with_test:
            return kernels_train, kernels_val, kernels_test 
        else:
            return kernels_train, kernels_val

    def save(self, prefix, kernels_train, kernels_val, kernels_test=None):
        for i in range(3):
            np.savetxt('{}_train_train_{}.txt'.format(prefix, i), kernels_train[i])
            np.savetxt('{}_train_val_{}.txt'.format(prefix, i), kernels_val[i])
            if kernels_test is not None:
                np.savetxt('{}_train_test_{}.txt'.format(prefix, i), kernels_test[i])
            np.savetxt('{}_labels_train_{}.txt'.format(prefix, i), self.labels_train[i])
            np.savetxt('{}_labels_val_{}.txt'.format(prefix, i), self.labels_val[i])
            with open('{}_raw_train_{}.txt'.format(prefix, i), 'w') as f:
                f.write("\n".join(self.raw_train[i]))
            
            with open('{}_raw_train_{}.txt'.format(prefix, i), 'w') as f:
                f.write("\n".join(self.raw_val[i]))

            np.savetxt('{}_feats_train_{}.txt'.format(prefix, i), self.feats_train[i])
            np.savetxt('{}_feats_val_{}.txt'.format(prefix, i), self.feats_val[i])

def train_val_dataset(exp):
    
    raw_train_1, raw_val_1, feats_train_1, feats_val_1, labels_train_1, labels_val_1 = exp.train_val_single_dataset(0)
    raw_train_2, raw_val_2, feats_train_2, feats_val_2, labels_train_2, labels_val_2 = exp.train_val_single_dataset(1)
    raw_train_3, raw_val_3, feats_train_3, feats_val_3, labels_train_3, labels_val_3 = exp.train_val_single_dataset(2)

    all_train_raw = np.vstack((raw_train_1, raw_train_2, raw_train_3))
    all_val_raw = np.vstack((raw_val_1, raw_val_2, raw_val_3))

    
    all_train_feats = np.vstack((feats_train_1, feats_train_2, feats_train_3))
    all_val_feats = np.vstack((feats_val_1, feats_val_2, feats_val_3))
    
    all_train_labels = np.hstack((labels_train_1, labels_train_2, labels_train_3))
    all_val_labels = np.hstack((labels_val_1, labels_val_2, labels_val_3))

    
    return all_train_raw, all_val_raw, all_train_feats, all_val_feats, all_train_labels, all_val_labels

def load_raw_test(i):
    test = np.array(pd.read_csv('data/Xte{}.csv'.format(i))["seq"])
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