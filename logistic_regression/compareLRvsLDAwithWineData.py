# Try to identify the logistic regression by two ways
# one is minimization of the cross entropy itteratively
# the other is using within variance-covariance matrices (LDA)
# following codes are based on the Udemy class
# "Deep Learning Prerequisite: logistic regression" provided by LazyProgrammer.com
# wine data use and LDA codes are based on following web site
# https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2

import numpy             as np
import numpy.matlib      as mlib
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine


wine = load_wine()

X = pd.DataFrame( wine.data, columns=wine.feature_names )
y = pd.Categorical.from_codes( wine.target, wine.target_names )

df = X.join( pd.Series( y, name='class' ) )

ones = np.ones( (X.shape[0], 1) )
Xb = np.concatenate( (X.values, ones), axis=1 ) # add bias column

# normalization
# there are several types of normalization
# this normalization is just shift the center point and modify the scale
xbar = np.mean( Xb, 0)
xbar[-1] = 0   # final dimension is for bias term so normalization isn't needed
nXb  = Xb - mlib.repmat( xbar, Xb.shape[0], 1 )
vcov = nXb.T.dot( nXb ) / nXb.shape[0]
stdX = np.sqrt( np.diag( vcov ) )
stdX[xbar == 0] = 1
nXb  = nXb / mlib.repmat( stdX, nXb.shape[0], 1 )



# following logistic regression code separates the 3 classes as 2 classes
# because logistic regression is used for binary classification problem (in my understandings)
# so regression is calculated three times,
# - class0 or other classes
# - class1 or other classes
# - class2 or other classes (is this necessary?)

# from compareLogisticRegressionVSLDA import sigmoid
# from compareLogisticRegressionVSLDA import cross_entropy
def sigmoid( a ):
    return 1 / ( 1 + np.exp(-a) )

def cross_entropy( T, Y ):
    E = 0
    for i in range( len(T) ):
        if T[i] == 1:
            E -= np.log( Y[i] )
        else:
            E -= np.log( 1 - Y[i] )
    return E

# list for class_0 or others, class_1 or others, and class_2 or others
w_list = []
learning_rate = 0.001
l2_regularization = 0.1

#
# Logistic Regression for each class
#
for i in range(3): # there are 3 classes in wine data
    class_name = 'class_' + str(i)
    w = np.random.randn( nXb.shape[1] )
    T = np.int8( y==class_name )
    z = sigmoid( nXb.dot(w) )
    cost_hist = [ cross_entropy( T, z ) ]
    for i in range(10000):
        if i % 100 == 0:
            print( cross_entropy( T, z ) )
        w -= learning_rate * ( nXb.T.dot( z - T ) - l2_regularization*w )
        z  = sigmoid( nXb.dot(w) )
        cost_hist.append( cross_entropy( T, z ) )
    w_list.append( w )

T[ y=='class_0' ] = 0
T[ y=='class_1' ] = 1
T[ y=='class_2' ] = 2
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter( nXb.dot(w_list[0]), nXb.dot(w_list[1]), nXb.dot(w_list[2]), c=T )
plt.show()



#
# Linear Discriminant Analysis (LDA)
#

# cut bias term (the final column)
nXb = nXb[:,:-1]

class_feature_mean = np.zeros( (3, nXb.shape[1]) )
for i in range( nXb.shape[0] ):
    class_feature_mean[T[i]] += nXb[i,:]

for i in range( 3 ):
    class_feature_mean[i] = class_feature_mean[i] / np.sum(T==i)

# sum of scatter matrix in each class
within_class_scatter_matrix = np.zeros( (nXb.shape[1], nXb.shape[1]) )
for i in range( nXb.shape[0] ):
    tmp = nXb[i, :] - class_feature_mean[ T[i] ]
    tmp = tmp.reshape( len(tmp), 1 )
    within_class_scatter_matrix += tmp.dot(tmp.T)

between_class_scatter_matrix = np.zeros( (nXb.shape[1], nXb.shape[1]) )
total_mean = np.mean( nXb, 0 )
for i in range( np.max(T)+1 ):
    tmp = class_feature_mean[T[i]] - total_mean
    tmp = tmp.reshape( len(tmp), 1 )
    between_class_scatter_matrix += tmp.dot(tmp.T) * np.sum(T==i)

eval, evec = np.linalg.eig( \
    np.linalg.inv( within_class_scatter_matrix ).dot( \
    between_class_scatter_matrix ) )

# sort the eigen values with corresponding eigen vector
pairs = [ (np.abs( eval[i] ), evec[:,i]) for i in range(len(eval)) ]
pairs = sorted( pairs, key=lambda x: x[0], reverse=True )

for pair in pairs:
    print( pair[0] )

eval_sum = np.sum( eval )

w_matrix = np.hstack( (pairs[0][1].reshape(13,1), \
                        pairs[1][1].reshape(13,1), \
                        pairs[2][1].reshape(13,1)) ).real

X_lda = np.array( nXb.dot( w_matrix ) )


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter( X_lda[:,0], X_lda[:,1], X_lda[:,2], c=T )
plt.show()

