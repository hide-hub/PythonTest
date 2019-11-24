# Try to identify the logistic regression by two ways
# one is minimization of the cross entropy itteratively
# the other is using within variance-covariance matrices
# following codes are based on the Udemy class
# "Deep Learning Prerequisite: logistic regression" provided by LazyProgrammer.com

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W):
    return sigmoid( X.dot( W ) )

# cross-entropy
def cross_entropy(T, Y):
    E = 0
    for i in range( len(T) ):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

# classification rate
def classification_rate( T, P ):
    return np.mean( T==P )


N = 100
D = 2

N_per_class = np.int(N / 2)

X = np.random.randn(N, D)

# prepare first half points of X to upper right
X[:N_per_class, :] = X[:N_per_class, :] + 2*np.ones( (N_per_class, D) )
# and rest of X to bottom left
X[N_per_class:, :] = X[N_per_class:, :] - 2*np.ones( (N_per_class, D) )

# Target categories
T = np.array( [0]*N_per_class + [1]*(N - N_per_class) )

# create bias dimension
ones = np.ones((N, 1))
Xb = np.concatenate( (X, ones), axis=1 )

W = np.random.randn( (3,1) )
z = Xb.dot( W )

learning_rate     = 0.01
l2_regularization = 0.1

for i in range(5000):
    if i % 100 == 0:
        print z
    W += learning_rate * Xb.dot( T - z ) + l2_regularization * W


