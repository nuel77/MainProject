import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from modules import *

def main():
    X, y  =datasets.make_blobs(n_samples=16,n_features=16,centers=10,shuffle=True)
    m,n= X.shape
    print("shape of data:", X.shape, y.shape)
    W1, b1, W2, b2 = gradient_descent(X, y, alpha=0.10, iterations=100, M=m)

    dev_predictions = make_predictions(X, W1, b1, W2, b2)
    accuracy= get_accuracy(dev_predictions, y)
    print('Accuracy on dev set:', accuracy)

if __name__ == "__main__":
    main()
