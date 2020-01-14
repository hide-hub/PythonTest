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


# test (homogenous coordinate simple linear solution)
# this result is not good (maybe because there is no normalization process now)
Y_test1 = evec[0,2]*X + evec[2,2]
plt.scatter( X, Y )
plt.plot( X, Y_test1 )
plt.show()

# calculate R-squared
# R-squared is, 1 - (residual / n times of variance )
d1 = Y - Y_hat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print( "R-squared in 1st solution is ", r2 )

d1 = Y - Y_test1
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print( "R-squared in 2nd solution is ", r2 )






