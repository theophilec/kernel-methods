import pandas as pd 
import numpy as np 

def train_val_single_dataset(i, VALIDATION_RATIO = 0.2):
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

def train_val_dataset():
    
    raw_train_1, raw_val_1, feats_train_1, feats_val_1, labels_train_1, labels_val_1 = train_val_single_dataset(0)
    raw_train_2, raw_val_2, feats_train_2, feats_val_2, labels_train_2, labels_val_2 = train_val_single_dataset(1)
    raw_train_3, raw_val_3, feats_train_3, feats_val_3, labels_train_3, labels_val_3 = train_val_single_dataset(2)

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

def load_all_test_datasets():
    te1_feats = np.loadtxt(open("data/Xte0_mat100.csv", "rb"), delimiter=" ")
    te2_feats = np.loadtxt(open("data/Xte1_mat100.csv", "rb"), delimiter=" ")
    te3_feats = np.loadtxt(open("data/Xte2_mat100.csv", "rb"), delimiter=" ")
    all_feats = np.vstack((te1_feats, te2_feats, te3_feats))
    
    return all_feats

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
                
def load_kernels(prefix):
    kernels_train, kernels_val, kernels_test, labels_train, labels_vals, feats_train, feats_val = [], [], [], [], [], [], []

    for i in range(3):
        kernels_val.append(np.loadtxt("{}_train_val_{}.txt".format(prefix, i)))
        kernels_train.append(np.loadtxt("{}_train_train_{}.txt".format(prefix, i)))
        kernels_test.append(np.loadtxt("{}_train_test_{}.txt".format(prefix, i)))
        labels_train.append(np.loadtxt("{}_labels_train_{}.txt".format(prefix, i)))
        labels_vals.append(np.loadtxt("{}_labels_val_{}.txt".format(prefix, i)))
        feats_train.append(np.loadtxt("{}_feats_train_{}.txt".format(prefix, i)))
        feats_val.append(np.loadtxt("{}_feats_val_{}.txt".format(prefix, i)))

    return kernels_train, labels_train, kernels_val, labels_vals, kernels_test, feats_train, feats_val