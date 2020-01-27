# multi regression test

# The data (X1, X2, X3) are for each patient
# X1 : systolic blood pressure
# X2 : age in years
# X3 : weight in pounds

import numpy  as np
import pandas as pd
from matplotlib           import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel( 'mlr02.xls' )
X  = df.as_matrix()

# 3D show
fig = plt.figure()
ax  = fig.add_subplot( 111, projection='3d' )
ax.scatter( df.X1, df.X2, df.X3 )
plt.show()

# 2D show for X1 and X2
plt.scatter( X[:,1], X[:,0] )
plt.show()

# 2D show for X1 and X3
plt.scatter( X[:,2], X[:,0] )
plt.show()

df['ones'] = 1
Y = df['X1']    # estimate blood pressure
# there are 3 ways for the selection of explanatory values
X = df[[ 'X2', 'X3', 'ones']]   # both X2 and X3 are explanatory values
X2only = df[['X2', 'ones']]     # only X2 is explanatory value
X3only = df[['X3', 'ones']]     # only X3 is explanatory value

# calculate each weights
w_both   = np.linalg.solve( np.dot( X.T, X ),           np.dot( X.T, Y ) )
w_x2only = np.linalg.solve( np.dot( X2only.T, X2only ), np.dot( X2only.T, Y ) )
w_x3only = np.linalg.solve( np.dot( X3only.T, X3only ), np.dot( X3only.T, Y ) )

# the function for calculating R-Squared value
def calcR2( Y, Y_hat ):
    d1 = Y - Y_hat
    d2 = Y - Y.mean()
    R2 = 1 - np.dot( d1.T, d1 ) / np.dot( d2.T, d2 )
    return R2

# compare R-Squared value of them
Yh_both   = X.dot( w_both )
Yh_x2only = X2only.dot( w_x2only )
Yh_x3only = X3only.dot( w_x3only )

print( 'compare the result of regressions' )
print( 'using both X2 and X3 \t:', calcR2( Y, Yh_both ) )
print( 'using X2 only \t:', calcR2(Y, Yh_x2only) )
print( 'using X3 only \t:', calcR2(Y, Yh_x3only) )

