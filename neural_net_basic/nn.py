import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from modules import *

def main():
    data = pd.read_csv('train.csv')
    data = np.array(data)
    m, n = data.shape
    # print("shape of data:", data.shape)
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:16].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[16:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape
    W1, b1, W2, b2 = gradient_descent(X_dev, Y_dev, alpha=0.10, iterations=50000, M=m)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    accuracy= get_accuracy(dev_predictions, Y_dev)
    print('Accuracy on dev set:', accuracy)

if __name__ == "__main__":
    main()
