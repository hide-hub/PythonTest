# Test Scores for General Psychology      -- mlr03.xls
#
# The data (X1, X2, X3, X4) are for each student.
# X1 = score on exam #1
# X2 = score on exam #2
# X3 = score on exam #3
# X4 = score on final exam

import numpy  as np
import pandas as pd
from matplotlib           import pyplot as plt
from numpy                import matlib as mlib
from mpl_toolkits.mplot3d import axis3d

df = pd.read_excel( 'DataFromCengage/mlr03.xls' )

N = df.shape[0]
D = df.shape[1] - 1

X = df.as_matrix()
Y = X[:, -1]
X = X[:,:-1]
#X = np.vstack([ X[:,0], X[:,1], X[:,2], np.ones(N) ]).T

# データのプロットテストをする場合はフラグを True にする
TryToShowPlot = False
if TryToShowPlot:
    for i in range( D ):
        for j in range( D ):
            if i >= j:
                continue
            fig = plt.figure()
            ax  = fig.add_subplot( 111, projection='3d' )
            ax.scatter( X[:,i], X[:,j], Y )
            plt.grid()
            plt.show()

TryToShowOneByOne = False
if TryToShowOneByOne:
    tmp = df.as_matrix()
    for i in range( D ):
        plt.scatter( X[:,i], Y )
        plt.grid()
        plt.show()

# split the data to train data and test data
trainX = X[:-5, :]
testX  = X[-5:, :]
trainY = Y[:-5]
testY  = Y[-5:]
trainN = N - 5
testN  = 5

#--------------------------------------------------
# define function
def calcR2( Y, Yhat ):
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    R2 = 1 - d1.dot(d1) / d2.dot(d2)
    return R2

#--------------------------------------------------
# use single variable
def calcSingleRegression( trainX, trainY, testX=0, testY=0 ):
    # assume trainX doesn't have bias term
    D = trainX.shape[1]
    w  = np.zeros( ( D, 2 ) )
    for i in range( D ):
        tmpX = np.vstack([ trainX[:,i], np.ones( trainX.shape[0] ) ]).T
        w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
    if np.isscalar( testX ) == True:
        return w
    else:
        R2_single = np.zeros( D )
        for i in range( D ):
            tmpX = np.vstack([ testX[:,i], np.ones( testX.shape[0] ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_single[i] = calcR2( testY, Yhat )
        return w, R2_single

w_single, R2_single = calcSingleRegression( trainX, trainY, testX, testY )

#--------------------------------------------------
# use double variables
import itertools
import math
def combinations_count( n, r ):
    # calculate the combination count of nCr
    r = min( r, n - r )
    return math.factorial( n ) // ( math.factorial( n-r ) * math.factorial( r ) )

def calcDoubleRegression( trainX, trainY, testX=0, testY=0 ):
    # assume trainX doesn't have bias term
    D = trainX.shape[1]
    w = np.zeros( ( combinations_count( D,2 ), 3 ) )
    i = 0
    for combi in itertools.combinations( list( range(D) ), 2 ):
        tmpX = np.vstack([ trainX[:, combi[0]], trainX[:, combi[1]], np.ones( trainX.shape[0] ) ]).T
        w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
        i += 1
    if np.isscalar( testX ) == True:
        return w
    else:
        R2_double = np.zeros( combinations_count( D,2 ) )
        i = 0
        for combi in itertools.combinations( list( range(D) ), 2 ):
            tmpX = np.vstack([ testX[:,combi[0]], testX[:,combi[1]], np.ones( testX.shape[0] ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_double[i] = calcR2( testY, Yhat )
            i += 1
        return w, R2_double

w_double, R2_double = calcDoubleRegression( trainX, trainY, testX, testY )

#--------------------------------------------------
# single variable 2nd order polynomial
def calcSinglePolyRegression( trainX, trainY, testX, testY ):
    # assume trainX doesn't have bias term
    D = trainX.shape[1]
    w  = np.zeros( ( D, 3 ) )
    for i in range( D ):
        tmpX = np.vstack([ trainX[:,i]**2, trainX[:,i], np.ones( trainX.shape[0] ) ]).T
        w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
    if np.isscalar( testX ) == True:
        return w
    else:
        R2_singlepoly = np.zeros( D )
        for i in range( D ):
            tmpX = np.vstack([ testX[:,i]**2, testX[:,i], np.ones( testX.shape[0] ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_singlepoly[i] = calcR2( testY, Yhat )
        return w, R2_singlepoly

w_singpoly, R2_singpoly = calcSinglePolyRegression( trainX, trainY, testX, testY )

#--------------------------------------------------
# double variables 2nd order polynomial
def calcDoublePolyRegression( trainX, trainY, testX, testY ):
    # assume trainX doesn't have bias term
    # the number of samples should be more than 6
    if trainX.shape[0] < 6:
        print( 'the number of samples should be more than 6!!' )
        return
    D = trainX.shape[1]
    w = np.zeros( ( combinations_count( D,2 ), 6 ) )
    i = 0
    for combi in itertools.combinations( list( range(D) ), 2 ):
        tX1 = trainX[:,combi[0]]
        tX2 = trainX[:,combi[1]]
        tmpX = np.vstack([ tX1**2, tX2**2, tX1*tX2, tX1, tX2, np.ones( trainX.shape[0] ) ]).T
        w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
        i += 1
    if np.isscalar( testX ) == True:
        return w
    else:
        R2_doubpoly = np.zeros( combinations_count( D,2 ) )
        i = 0
        for combi in itertools.combinations( list( range(D) ), 2 ):
            tX1 = testX[:,combi[0]]
            tX2 = testX[:,combi[1]]
            tmpX = np.vstack([ tX1**2, tX2**2, tX1*tX2, tX1, tX2, np.ones( testX.shape[0] ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_doubpoly[i] = calcR2( testY, Yhat )
            i += 1
        return w, R2_doubpoly

w_doubpoly, R2_doubpoly = calcDoublePolyRegression( trainX, trainY, testX, testY )

#--------------------------------------------------
# find the best R2 score
bestR2 = 0
bestSol = []
for i in range( len( R2_single ) ):
    if bestR2 < R2_single[i]:
        bestR2  = R2_single[i]
        bestSol = 'single regression from variable: {}'.format(i+1)

combilist = list( itertools.combinations( list(range(X.shape[1])), 2 ) ) + 1
for i in range( len( R2_double ) ):
    if bestR2 < R2_double[i]:
        bestR2  = R2_double[i]
        bestSol = 'double regression from variables: {} and {}'.format(combilist[i][0], combilist[i][1] )

for i in range( len( R2_singpoly ) ):
    if bestR2 < R2_singpoly[i]:
        bestR2  = R2_singpoly[i]
        bestSol = 'single polynomial regression from variable: {}'.format(i+1)

for i in range( len( R2_doubpoly ) ):
    if bestR2 < R2_doubpoly[i]:
        bestR2  = R2_doubpoly[i]
        bestSol = 'double polynomial regression from variables: {} and {}'.format(combilist[i][0], combilist[i][1] )

bestR2
bestSol
# fortunately the best prediction of final EXAM is estimated
# from the results of EXAM2 and EXAM3

