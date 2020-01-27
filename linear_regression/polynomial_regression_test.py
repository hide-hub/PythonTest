# Polynomial regression test

import numpy as np
from matplotlib import pyplot as plt

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split( ',' )
    x = float(x)
    X.append( [x, x**2, 1] )
    Y.append( float(y) )

X = np.array( X )
Y = np.array( Y )

# # show the scatter plot
# plt.scatter( X[:,0], Y )
# plt.show()

# calculate the weights
w = np.linalg.solve( np.dot( X.T, X ), np.dot( X.T, Y ) )
Y_hat = X.dot( w )

# R-Squared
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - np.dot( d1.T, d1 ) / np.dot( d2.T, d2 )

# plot it all together
plt.scatter( X[:,0], Y )
#plt.scatter( X[:,0], Y_hat )
pairs = [ (X[i,0], Y_hat[i]) for i in range( len(Y_hat) ) ]
pairs = sorted( pairs, key=lambda x: x[0] )
sortedX     = [ pairs[i][0] for i in range( len(pairs) ) ]
sortedY_hat = [ pairs[i][1] for i in range( len(pairs) ) ]
plt.plot( sortedX, sortedY_hat )
plt.show()


