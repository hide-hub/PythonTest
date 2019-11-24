# Try to identify the logistic regression by two ways
# one is minimization of the cross entropy itteratively (LDA)
# the other is using within variance-covariance matrices
# following codes are based on the Udemy class
# "Deep Learning Prerequisite: logistic regression" provided by LazyProgrammer.com

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

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

W = np.random.randn( D + 1 ) # plus 1 is bias part
z = sigmoid( Xb.dot( W ) )

learning_rate     = 0.001
l2_regularization = 0.1

# histry of the cost (cross entropy) or error
ctrain = cross_entropy( T, z )
cost_hist = [ctrain]

for i in range(5000):
    if i % 100 == 0:
        print( ctrain )
    W -= learning_rate * ( Xb.T.dot( z - T ) - l2_regularization * W )
    z = sigmoid( Xb.dot( W ) )
    ctrain = cross_entropy( T, z )
    cost_hist.append( ctrain )

# plt.show( cost_hist )
# plt.show()

# print the result of minimizing cross-entropy
print( "The Final Line for minimization of cross-entropy : " +  str(W) )



# Linear Discriminant Analysis (using scatter matrices)
# following codes are based on the bellow web site
# https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2

# above logistic regression code are flex for dimension and classes number
# but following code assumes the dimension is 2 + bias term
from numpy import matlib as mlib
within_class_scatter_matrix = np.zeros( (D, D) )
meanX00 = np.mean( X[:N_per_class, :], 0 )
meanX01 = np.mean( X[N_per_class:, :], 0 )


tmpX = X[:N_per_class,:] - mlib.repmat( meanX00, N_per_class, 1 )
within_class_scatter_matrix += tmpX.T.dot( tmpX )
tmpX = X[N_per_class:,:] - mlib.repmat( meanX01, N-N_per_class, 1 )
within_class_scatter_matrix += tmpX.T.dot( tmpX )

between_class_scatter_matrix = np.zeros( (D, D) )
meanX = np.mean( X, 0 )
meanX = meanX.reshape( D, 1 )
meanX00 = meanX00.reshape( D, 1 )
meanX01 = meanX01.reshape( D, 1 )
between_class_scatter_matrix += ( (meanX - meanX00).dot( (meanX - meanX00).T ) ) * N_per_class
between_class_scatter_matrix += ( (meanX - meanX01).dot( (meanX - meanX01).T ) ) * (N - N_per_class)

eigval, eigvec = np.linalg.eig( \
    np.linalg.inv( within_class_scatter_matrix ).dot( \
    between_class_scatter_matrix ) )

pairs = [(np.abs( eigval[i] ), eigvec[:,i]) for i in range( len(eigval) ) ]

pairs = sorted( pairs, key=lambda x : x[0], reverse=True )

for pair in pairs:
    print( pair[0] )

w_matrix = np.hstack( (pairs[0][1].reshape(D,1), pairs[1][1].reshape(D,1)) ).real






