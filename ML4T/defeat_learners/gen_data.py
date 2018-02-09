"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.random((100,2))
    #Y = np.random.random(size = (100,))*200-100
    Y = np.sin(X[:,0]) + np.cos(X[:,1])
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.random((100,4))
    Y = np.zeros(100)
    #X = np.random.normal(size=(50,4))
    #Y = np.random.random(size = (100,))*200-100
    for row in range(0,25):
        X[row,:] = X[row, :]**-5
        Y[row] = -5
    for row in range(25,50):
        X[row,:] = -X[row,:]**-3
        Y[row] = -3
    for row in range(50,75):
        X[row,:] = (X[row,:])**10
        Y[row] = 10
    for row in range(75,100):
        X[row,:] = (X[row,:])**20
        Y[row] = 20

    #Y = np.linspace(1,10000, num=50) + np.random.normal(0,1000,50)
    #print Y
    return X, Y

def author():
    return 'dnguyen333'

if __name__=="__main__":
    print "they call me Bill."
