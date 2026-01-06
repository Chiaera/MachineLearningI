
import numpy as np
import pandas as pd
from pathlib import Path

# FUNCTIONS for the tasks ------------------------------------------------------------

#dataset
def load_data(path, c=2, shuffle=True):
    df = pd.read_csv(path, sep=r'\s+').values

    n, D = df.shape
    d = D - c

    #shuffle
    if shuffle:
        data = df[np.random.permutation(n), :]
    else:
        data = df

    X = data[:, :d] #features (nxd)
    Y = data[:, -c:] #target(cxd)

    return X, Y

#normalization feature X
def minmax_normalize_train(X_train):
    min_values = X_train.min(axis=0)
    ranges = X_train.max(axis=0) - min_values
    ranges[ranges == 0] = 1.0

    X_norm = (X_train - min_values) / ranges
    
    return X_norm, min_values, ranges

def minmax_X(X, min, ranges):
    return (X - min) / ranges


#normalization target Y
def minmax_normalize_train_Y(Y_train):
    min = Y_train.min(axis=0)
    ranges = Y_train.max(axis=0) - min
    ranges[ranges == 0] = 1.0
    Y_norm = (Y_train - min) / ranges
    return Y_norm, min, ranges

def minmax_Y(Y, min, ranges):
    return (Y - min) / ranges


# MAIN --------------------------------------------------------------------------------

def main():
    directory = Path(__file__).resolve().parent
    data_path = directory/"data"/"data.txt"

    X, Y = load_data(path=data_path, c=2, shuffle=True)

    n, d = X.shape
    
    Y = Y.astype(float)

    Xn, minX, rangeX = minmax_normalize_train(X)
    Yn, minY, rangeY = minmax_normalize_train_Y(Y)
    

if __name__ == '__main__':
    main()