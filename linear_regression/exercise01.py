# Thunder Basin Antelope Study    -- mlr01.xls
# 
# The data (X1, X2, X3, X4) are for each year.
# X1 = spring fawn count/100
# X2 = size of adult antelope population/100
# X3 = annual precipitation (inches)
# X4 = winter severity index (1=mild, 5=severe)

import numpy  as np
import pandas as pd
from matplotlib           import pyplot as plt
from numpy                import matlib as mlib
from mpl_toolkits.mplot3d import axis3d

df = pd.read_excel( 'DataFromCengage/mlr01.xls' )

N = df.shape[0] 
D = df.shape[1] - 1


# データのプロットテストをする場合はフラグを True にする
TryToShowPlot = False
if TryToShowPlot:
    # there are 4 variables (one variable is target)
    # so try 3 times to plot all data
    tmp = df.as_matrix()
    for i in [0, 2, 3]:     # index 1 is target variable
        for j in [0, 2, 3]:
            if i >= j:
                continue
            fig = plt.figure()
            ax  = fig.add_subplot( 111, projection='3d' )
            ax.scatter( tmp[:,i], tmp[:,j], tmp[:,1] )
            plt.grid()
            plt.show()

TryToShowOneByOne = False
if TryToShowOneByOne:
    tmp = df.as_matrix()
    for i in [0, 2, 3]:
        plt.scatter( tmp[:,i], tmp[:,1] )
        plt.show()


# let's predict X2: antelope size
X = np.vstack( [ df.X1, df.X3, df.X4, np.ones(N) ] ).T
Y = np.array( df.X2 )

# set 1/4 of data to test data and rest of them are train data
trainX = X[:-2, :]
testX  = X[-2:, :]
trainY = Y[:-2]
testY  = Y[-2:]

# simple linear regression
w = np.linalg.solve( trainX.T.dot(trainX), trainX.T.dot(trainY) )

# calculate R-Squared value
def calcR2( Y, Yhat ):
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    R2 = 1 - d1.dot(d1) / d2.dot(d2)
    return R2

# apply to all data
Yhat = X.dot( w )
R2_all = calcR2( Y, Yhat )

# apply to train data
R2_train = calcR2( Y[:-2], Yhat[:-2] )

# apply to test
R2_test  = calcR2( Y[-2:], Yhat[-2:] )

# # try to use PCA
# # standardize the explanatory variables X
# X = X[:,:-1]    # delete bias column because standardized X doesn't need bias (data is centered to its mean)
# tmpX = X - X.mean(0)
# s = np.sqrt( np.diag( tmpX.T.dot( tmpX ) ) )
# stX = ( X - X.mean(0) ) / mlib.repmat( s.reshape(1,D), len(X), 1 )     # stX is standardized X

# eval, evec = np.linalg.eig( stX.T.dot( stX ) )

#
# polynomial fitting
#
# try 2nd order polynomial
polyX = np.vstack( [X.T, X[:,0]*X[:,1], X[:,0]*X[:,2], X[:,1]*X[:,2], X[:,0]**2, X[:,1]**2, X[:,2]**2] ).T
ptrainX = polyX[:-2,:]
ptestX  = polyX[-2:,:]

# cause polyX is fat matrix (D > N)
# l1 regularization should be used
D_poly = polyX.shape[1]
w_poly = np.random.randn( D_poly ) / np.sqrt( D_poly )
costs = []
learning_rate = 0.00001
l1 = 10
for i in range(10000):
    Yhat   = ptrainX.dot( w_poly )
    delta  = Yhat - trainY
    w_poly = w_poly - learning_rate * ( ptrainX.T.dot( delta ) + l1 * np.sign( w_poly ) )
    mse    = delta.dot(delta)
    costs.append( mse )

Yhat = polyX.dot( w_poly )
R2_poly   = calcR2( Y, Yhat )
R2_ptrain = calcR2( Y[:-2], Yhat[:-2] )
R2_ptest  = calcR2( Y[-2:], Yhat[-2:] )
# R2_ptest is minus...


#
# Try to use single variable
#
R2train_single = np.zeros( 3 )
R2test_single  = np.zeros( 3 )
w_single    = np.zeros( (3,2) )
Yhat_single = np.zeros( (len(Y),3) )
for i in range(3):
    tmpX = np.vstack( [X[:,i], X[:,3]] ).T
    tmpTrainX = tmpX[:-2,:]
    tmpTestX  = tmpX[-2:,:]
    w_single[i,:] = np.linalg.solve( tmpTrainX.T.dot(tmpTrainX), tmpTrainX.T.dot(trainY) )
    Yhat_single[:,i] = tmpX.dot( w_single[i,:] )
    R2train_single[i] = calcR2( trainY, Yhat_single[:-2,i] )
    R2test_single[i]  = calcR2( testY,  Yhat_single[-2:,i] )

# interesting! Compariing R2train_single and R2test_signle,
# the 3rd explanatory variable is good for test data but bad for train data!

# choose 1st and 2nd variables for computing w
tmpX      = np.vstack([ X[:,0], X[:,1], X[:,-1] ]).T
tmpTrainX = tmpX[:-2,:]
tmpTestX  = tmpX[-2:,:]
w_double = np.linalg.solve( tmpTrainX.T.dot(tmpTrainX), tmpTrainX.T.dot(trainY) )
Yhat = tmpX.dot( w_double )
R2train_double = calcR2( trainY, Yhat[:-2] )
R2test_double  = calcR2( testY,  Yhat[-2:] )
# double R2 score is not good...

#
# Try polynomial for double variables
#
dpolyX   = np.vstack([ X[:,0], X[:,1], X[:,0]*X[:,1], X[:,0]**2, X[:,1]**2, X[:,-1] ]).T
dptrainX = dpolyX[:-2,:]
dptestX  = dpolyX[-2:,:]
D_dpoly = dpolyX.shape[1]
w_dpoly = np.random.randn( D_dpoly ) / np.sqrt( D_dpoly )
costs = []
learning_rate = 0.00001
l1 = 20
for i in range(10000):
    Yhat    = dptrainX.dot( w_dpoly )
    delta   = Yhat - trainY
    w_dpoly = w_dpoly - learning_rate * ( dptrainX.T.dot( delta ) + l1 * np.sign( w_dpoly ) )
    mse     = delta.dot(delta)
    costs.append( mse )

Yhat = dpolyX.dot( w_dpoly )
R2_dpoly   = calcR2( Y, Yhat )
R2_dptrain = calcR2( Y[:-2], Yhat[:-2] )
R2_dptest  = calcR2( Y[-2:], Yhat[-2:] )

# Ummm...



