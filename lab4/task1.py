#task1: data preparation

import numpy as np
import tensorflow as tf


#load dataset
def load_raw_mnist():
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    '''
    X (train/test): shape (N, 28, 28), pixel values in [0, 255]
    y (train/test): shape (N,), labels 0-9
    '''
    return X_train, y_train, X_test, y_test 


#filter the choisen classes
def filter_classes(X, y, classes):
    if classes is None:
        return X, y  #they will be used for Task 3 

    mask = np.isin(y, classes) #true value on the positions of the chosen classes 
    return X[mask], y[mask] #x and y with that classes


#normalize [0,1] and flatten images (vectors size: 28x28 -> 784)
def normalize_and_flatten(X):
    X = X.astype("float32") / 255.0 #normalize to [0, 1]
    X = X.reshape((X.shape[0], -1))  #(N, 28, 28) -> (N, 784)
    return X


#processing pipeline: load, filter, normalize, flatten
def prepare_data(classes=(5, 7), flatten=True):
    X_train, y_train, X_test, y_test = load_raw_mnist()

    #validate classes
    if classes is not None:
        if len(classes) != 2:
            raise ValueError("classes must contain exactly 2 digits")
        if classes[0] == classes[1]:
            raise ValueError("classes must be two different digits")
        if not all(0 <= c <= 9 for c in classes):
            raise ValueError("MNIST classes must be between 0 and 9")

    # Filter classes
    X_train, y_train = filter_classes(X_train, y_train, classes)
    X_test, y_test = filter_classes(X_test, y_test, classes)

    # Normalize + flatten
    if flatten:
        X_train = normalize_and_flatten(X_train)
        X_test = normalize_and_flatten(X_test)
    else:
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    return X_train, X_test, y_train, y_test
