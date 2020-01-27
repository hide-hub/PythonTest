# this program is L2 Regularization test program
# I've learned L2 Regularization in the Udemy Course, Logistic Regression
# This is for blush up

import numpy as np
from matplotlib import pyplot as plt

N = 50
X = np.linspace( 0, 10, N )
Y = 0.5*X + np.random.randn( N )

Y[-1] += 30
Y[-2] += 30

plt.scatter( X, Y )
plt.show()

X = np.vstack( [X, np.ones(N)] ).T
w_ml = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
Yhat_ml = X.dot( w_ml )

plt.scatter( X[:,0], Y )
plt.plot( X[:,0], Yhat_ml )
plt.show()

l2 = 1000
w_map = np.linalg.solve( l2*np.eye(2) + X.T.dot(X), X.T.dot(Y) )
Yhat_map = X.dot( w_map )

plt.scatter( X[:,0], Y )
plt.plot( X[:,0], Yhat_ml,  label='maximum likelihood' )
plt.plot( X[:,0], Yhat_map, label='map' )
plt.legend()
plt.show()

# map solution is much robust for outliers


