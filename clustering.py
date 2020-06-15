'''
    Data Mining - Term Project
    Student: Osman Araz
    ID: 16011020
    Delivery Date: 10.12.2019

    Part 2 - Clustering:
    Clustering with K-Means.
'''

# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring

import numpy as np
from preprocess_dataset import readDataset, shuffleDataset, normalizeFeatures

#######################################

def findDistance(s1, s2):
    '''
    L2 distance between samples.
    '''
    return np.sqrt(np.sum(np.square(s1-s2)))

#######################################

def findClosestCluster(X, C1, C2):
    S = [None]*len(X)

    for i in range(len(X)):
        dist_C1 = findDistance(X[i], C1)
        dist_C2 = findDistance(X[i], C2)
        S[i] = 0 if dist_C1 < dist_C2 else 1

    return S

#######################################

def assignNewCentroids(X, S, C1, C2):
    C1_new, C2_new = 0*C1, 0*C2
    C1_count, C2_count = 0.1, 0.1

    for i in range(X.shape[0]):
        if S[i] == 0:
            C1_new += X[i]
            C1_count += 1
        else:
            C2_new += X[i]
            C2_count += 1

    return C1_new/C1_count, C2_new/C2_count

#######################################

def evaluateModel(S, Y):
    same = 0

    for i in range(S.shape[0]):
        same += S[i] == Y[i]

    accuracy = same / S.shape[0]

    if accuracy < .5:
        S = 1-S
        accuracy = 1-accuracy
        same = S.shape[0]-same

    return S, accuracy, same

#######################################

def main():
    X, Y = readDataset("echocardiogram.csv")
    X, Y = shuffleDataset(X, Y)
    X = normalizeFeatures(X)

    C1, C2 = X[0], X[1] # initializing centroids as first two samples

    print("Center of clusters over iterations (normalized values):\n")
    print("initial:")
    print("C1 = {}".format(C1))
    print("C2 = {}\n".format(C2))

    for i in range(5): # for 5 iteration
        S = findClosestCluster(X, C1, C2)
        C1, C2 = assignNewCentroids(X, S, C1, C2)
        print("iteration {}:".format(i+1))
        print("C1 = {}".format(C1))
        print("C2 = {}\n".format(C2))

    S = np.asarray(S, dtype=np.int8)
    S, accuracy, same = evaluateModel(S, Y)

    print("------------------------------------------------\n")
    print("Predictions (predicted cluster -> real cluster):\n")

    for i in range(len(S)):
        print("  {} -> {}".format(S[i], Y[i]))

    print("\nAccuracy of the model: {}%".format(format(100*accuracy, ".4f")))
    print("{} samples correctly clustered over {} samples.".format(same, S.shape[0]))

#######################################

np.set_printoptions(precision=5, suppress=True) # ignoring scientific notation

if __name__ == "__main__":
    main()
