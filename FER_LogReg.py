p# facial expression recognition test
# facial data is provided by kaggle challenge project
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
# the code for recognition is for udemy lazyprogrammer's project

# this is a just test and practice program for data science
# The 2nd try is Logistic Regression

import csv
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy      import matlib as mlib
from utils_fer  import read_dataset
from utils_fer  import sigmoid
from utils_fer  import cross_entropy

##################################################
# program body
##################################################
X, Y, Usage = read_dataset( 'fer2013/fer2013.csv' )

# normalize the value from range X(0-255) to 0-1
X = X / 255

Xt = X[Usage==0,:]
Yt = Y[Usage==0]

# check the mean of each category
catmeanX = []
catNum   = np.max(Yt) + 1   # number of categories
for cat in range( catNum ):
    tmp = Xt[Yt==cat, :]
    catmeanX.append( np.mean(tmp,0) )

catmeanX = np.array( catmeanX )

# check distance between each category
# from category 0 to other categories
catdist = []
for cat in range( catNum ):
    diff_center = catmeanX - mlib.repmat( catmeanX[cat, :], catNum, 1 )
    distances = np.sqrt( np.diag( diff_center.dot(diff_center.T) ) )
    catdist.append( distances )

catdist = np.array( catdist )

# display the distance table
tmp = np.round(catdist*1000)/1000
print( "distance from each centers to others" )
print( tmp )

# check variance covariance in each category
vcovM = []
for cat in range( catNum ):
    tmpX  = Xt[Yt==cat,:]
    diffX = tmpX - mlib.repmat( catmeanX[cat,:], tmpX.shape[0], 1 )
    vcov  = diffX.T.dot( diffX ) / tmpX.shape[0]
    vcovM.append( vcov )

# # make class to 0 or other (replace the value to True==1)
# tmpY = Yt != 0

# # add one dimension to Xt for bias term
# Xt = np.concatenate( ( Xt, np.ones( (Xt.shape[0],1) ) ), axis=1 )

## tmp try
# make boundary between 2 classes
# so the boundaries should be 21 = 2 combination of 7

tmpY = np.concatenate( (Yt[Yt==4],   Yt[Yt==5]) )
tmpX = np.concatenate( (Xt[Yt==4,:], Xt[Yt==5,:]), axis=0 )

# set label (tmpY) to 0 or 1 because this is 2 categories classification
tmpY[tmpY==4] = 0
tmpY[tmpY==5] = 1

## change code not to use bias term (calculate bias separately)
# # add dimension for bias term
# tmpX = np.concatenate( (tmpX, np.ones( (tmpX.shape[0],1) )), axis=1 )

# random weight
w = np.random.randn( tmpX.shape[1] ) / np.sqrt( tmpX.shape[1] )
b = 0

# initial prediction
Yp = sigmoid( tmpX.dot( w ) + b )

costhist = []
learning_rate     = 1*(10**-6)
l1_regularization = 0.1
l2_regularization = 1
for i in range( 5000 ):
    if i % 100 == 0:
        print( cross_entropy( tmpY, Yp ) )
    costhist.append( cross_entropy( tmpY, Yp ) )
    w -= learning_rate * ( tmpX.T.dot( Yp - tmpY ) \
         - l1_regularization*np.sign( w ) \
         - l2_regularization*( w ) )
    b -= learning_rate * np.sum( Yp - tmpY )
    Yp = sigmoid( tmpX.dot( w ) + b )
    costhist.append( cross_entropy( tmpY, Yp ) )

plt.plot(costhist)
plt.show()

result = tmpX.dot(w) + b
plt.scatter( range( len(result) ), result, c=tmpY, cmap='rainbow', alpha=0.2, edgecolor='k' )
plt.show()

plt.scatter( range( len(result) ), sigmoid(result), c=tmpY, cmap='rainbow', alpha=0.2, edgecolors='k' )
plt.show()


## end of tmp try
