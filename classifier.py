'''
    Data Mining - Term Project
    Student: Osman Araz
    ID: 16011020
    Delivery Date: 10.12.2019

    Part 1 - Classifier:
    Classification with Artificial Neural Networks.
'''

# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring

import numpy as np
import matplotlib.pyplot as plt
from preprocess_dataset import readDataset, shuffleDataset, normalizeFeatures, splitDataset, crossValidationSplit

#######################################

def relu(z): # ReLU activation function
    return z * (z >= 0)

#######################################

def relu_derivative(z): # derivation of ReLU activation function
    return 1 * (z >= 0)

#######################################

def sigmoid(z): # sigmoid activation function
    return 1 / (1 + np.exp(-z))

#######################################

def sigmoid_derivative(z): # derivation of sigmoid activation function
    s = sigmoid(z)
    return s * (1-s)

#######################################

def initializeParameters(l1, l2, l3):
    '''
    l1: number of hidden units in input layer
    l2: number of hidden units in hidden layer
    l3: number of hidden units in output layer
    '''
    W1 = np.random.randn(l2, l1)
    W2 = np.random.randn(l3, l2)
    B1 = np.zeros((l2, 1))
    B2 = np.zeros((l3, 1))

    return W1, B1, W2, B2

#######################################

def forwardPropogation(W1, B1, W2, B2, X):
    '''
    W1: weight matrix for hidden layer
    B1: bias array for hidden layer
    W2: weight matrix for output layer
    B2: bias array for output layer
    '''
    Z1 = np.dot(W1, X) + B1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2

#######################################

def neuralNetwork(W1, B1, W2, B2, X, Y, k_fold=10, alpha=0.1, num_of_iterations=1000):
    m = X.shape[1] # number of samples in X

    fold = 1 # current fold
    fold_size = int(m / k_fold)
    iters_per_fold = int(num_of_iterations / k_fold)

    train_hist = []
    validation_hist = []
    epoch = 0

    for i in range(0, m, fold_size): # cross-validation
        X_train, Y_train, X_validation, Y_validation = crossValidationSplit(X, Y, i, fold_size)
        A0 = X_train

        m_train = X_train.shape[1] # number of samples in current train set
        m_validation = X_validation.shape[1] # number of samples in current validation set

        j = 0

        while epoch < num_of_iterations and j < iters_per_fold:
            # forward propogation:
            Z1, A1, Z2, A2 = forwardPropogation(W1, B1, W2, B2, A0) # for train set
            _, _, _, A_validation = forwardPropogation(W1, B1, W2, B2, X_validation) # for validation set

            # backward propogation for W2 and B1:
            dA = A2 - Y_train
            dZ = dA * sigmoid_derivative(Z2)
            dW2 = np.dot(dZ, A1.T) / m_train
            dB2 = np.sum(dZ) / m_train

            # backward propogation for W1 and B1:
            dA = np.dot(W2.T, dZ)
            dZ = dA * relu_derivative(Z1)
            dW1 = np.dot(dZ, A0.T) / m_train
            dB1 = np.sum(dZ) / m_train

            # gradient descent:
            W1 -= alpha * dW1
            B1 -= alpha * dB1
            W2 -= alpha * dW2
            B2 -= alpha * dB2

            # computing loss
            train_loss = np.square(A2 - Y_train).sum() / m_train
            validation_loss = np.square(A_validation - Y_validation).sum() / m_validation
            train_hist.append(train_loss)
            validation_hist.append(validation_loss)

            if epoch % 250 == 0:
                print("- fold {}: epoch {} -> train_loss: {}, val_loss: {}".format(fold, epoch, format(train_hist[-1], ".5f"), format(validation_hist[-1], ".5f")))

            j += 1
            epoch += 1

        fold += 1

    return train_hist, validation_hist

#######################################

def predict(W1, B1, W2, B2, X):
    _, _, _, A2 = forwardPropogation(W1, B1, W2, B2, X)

    return np.round(A2)

#######################################

def main():
    X, Y = readDataset("echocardiogram.csv")
    X, Y = shuffleDataset(X, Y)
    X = normalizeFeatures(X)
    X_train, Y_train, X_test, Y_test = splitDataset(X, Y, 0.8)
    print("Dataset sizes: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, "\n")

    W1, B1, W2, B2 = initializeParameters(X_train.shape[0], 10, 1)
    train_hist, validation_hist = neuralNetwork(W1, B1, W2, B2, X_train, Y_train, k_fold=7, alpha=0.5, num_of_iterations=10000)

    Y_train_pred = predict(W1, B1, W2, B2, X_train)
    Y_test_pred = predict(W1, B1, W2, B2, X_test)

    print("\n-> Train accuracy: {}%".format(format(100 - np.mean(np.abs(Y_train_pred - Y_train)) * 100, ".2f")))
    print("-> Test accuracy: {}%".format(format(100 - np.mean(np.abs(Y_test_pred - Y_test)) * 100, ".2f")))

    plt.figure()
    plt.plot(validation_hist, "r-")
    plt.plot(train_hist, "b-")
    plt.legend(['Validation Loss', 'Train Loss'], loc='upper right')
    plt.title("Loss per Epochs")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

#######################################

np.set_printoptions(suppress=True) # ignoring scientific numeric notation

if __name__ == "__main__":
    main()
