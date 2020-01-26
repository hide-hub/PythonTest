# this is test program for linear regression
# the code is copied from Udemy course, Linear Regression in Python
import numpy as np
from matplotlib           import pyplot as plt
from mpl_toolkits.mplot3d import axis3d

# load the data
X = []
Y = []
for line in open( 'data_2d.csv' ):
    x1, x2, y = line.split(',')
    X.append( [float( x1 ), float( x2 ), 1] )
    Y.append(  float( y ) )

# turn X and Y as numpy arrays
X = np.array( X )
Y = np.array( Y )

# # let's plot the data to see what the data looks like
# fig = plt.figure()
# ax  = fig.add_subplot( 111, projection='3d' )
# ax.scatter( X[:,0], X[:,1], Y )

# calculate weights
w = np.linalg.solve( np.dot( X.T, X ), np.dot( X.T, Y ) )
# w = np.linalg.inv( np.dot(X.T, X) ).dot( X.T.dot( Y ) )

Y_hat = np.dot( X, w )

# R-Squared
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - np.dot( d1, d1 ) / np.dot( d2, d2 )
print( 'R-Squared:', R2 )

fig = plt.figure()
ax  = fig.add_subplot( 111, projection='3d' )
ax.scatter( X[:,0], X[:,1], Y )
ax.scatter( X[:,0], X[:,1], Y_hat )
plt.show()

