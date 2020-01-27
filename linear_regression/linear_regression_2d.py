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



# appendix
# if there is completely random values in X,
# will the R-Squared value improved?
X = []
Y = []
for line in open( 'data_2d.csv' ):
    x1, x2, y = line.split(',')
    X.append( [ float(x1), float(x2), np.random.rand(), 1 ] )
    Y.append( float(y) )

X = np.array(X)
Y = np.array(Y)

w2 = np.linalg.solve( np.dot( X.T, X ), np.dot( X.T, Y ) )
Y_hat2 = X.dot( w2 )
d1 = Y - Y_hat2
d2 = Y - Y.mean()
R22 = 1 - d1.dot(d1) / d2.dot(d2)

# the R-Squared value is slightly improved
# the reason of this is the random values are just sample of random values
# normally, random values doesn't have any correlation to other values
# but sampled random value chould have slight correlation to other values
# np.linalg.solve() calculates the relation ship between X.T.dot(X) and X.T.dot(Y)
# acoording to least mean squared method
# the result, weight w, has some value to explain the Y from input X including random values
# the result of prediction is slightly improved because of sampled random values



