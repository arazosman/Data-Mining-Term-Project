'''
    Data Mining - Term Project
    Student: Osman Araz
    ID: 16011020
    Delivery Date: 10.12.2019

    Functions for dataset pre-processing.
'''

# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring

import numpy as np
import csv

#######################################

def readDataset(path):
    '''
    Reading dataset from given CSV file.
    '''
    X, Y = [], []

    with open(path, newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        size = len(next(file)) # skipping headers

        for row in file:
            X.append(row[:size-1])   # features
            Y.append(row[size-1])    # labels

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)

    return X, Y

#######################################

def shuffleDataset(X, Y):
    '''
    Shuffling X and Y with same indexes.
    '''
    #np.random.seed(0)
    perm = np.random.permutation(X.shape[0])

    X_shuffled = X[perm, :]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled

#######################################

def normalizeFeatures(X):
    '''
    Normalizing features with min-max normalization in range of (0, 1).
    '''
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    return X_normalized

#######################################

def splitDataset(X, Y, rate):
    size = X.shape[0]
    split_point = int(size*rate)

    X_train = X[:split_point, :].T
    Y_train = Y[:split_point].reshape(1, -1)
    X_test = X[split_point:, :].T
    Y_test = Y[split_point:].reshape(1, -1)

    return X_train, Y_train, X_test, Y_test

#######################################

def crossValidationSplit(X, Y, fold, fold_size):
    X_train = np.hstack((X[:, :fold], X[:, fold+fold_size:]))
    Y_train = np.hstack((Y[:, :fold], Y[:, fold+fold_size:]))
    X_validation = X[:, fold:fold+fold_size]
    Y_validation = Y[:, fold:fold+fold_size]

    return X_train, Y_train, X_validation, Y_validation
