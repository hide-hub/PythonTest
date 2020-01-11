# this is initial test program for Udemy course, Linear Regression in Python
# the most code is derived from the course
# and some parts of this program is original code for comparison

import numpy             as np
import matplotlib.pyplot as plt
import csv

# read csv data file
X = []
Y = []
A = []
for line in open( 'data_1d.csv' ):
    x, y = line.split(',')
    X.append( x )
    Y.append( y )
    A.append( np.array( [x, y, 1] ) )

# let's turn X and Y into numpy arrays
X = np.array( X, dtype=float )
Y = np.array( Y, dtype=float )
A = np.array( A, dtype=float )

# plot them to see what it look like
plt.scatter( X, Y )
plt.show()


# apply the data for calculating a and b
N = len( X )
demoninator = X.dot( X ) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / demoninator
b = ( X.dot(X)*Y.mean() - X.dot(Y)*X.mean() ) / demoninator

# test the linear solution of homogenious coordinate line fitting
eval, evec = np.linalg.eig( A.T.dot( A ) )
import numpy.matlib as mlib
ys = mlib.repmat( evec[1,:], 3, 1 )
evec = evec / -ys

# show the result
Y_hat = a*X + b

plt.scatter( X, Y )
plt.plot( X, Y_hat )
plt.show()


