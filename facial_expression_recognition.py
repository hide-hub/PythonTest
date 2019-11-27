# facial expression recognition test
# facial data is provided by kaggle challenge project
# the code for recognition is for udemy lazyprogrammer's project

import csv
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy      import matlib as mlib


##################################################
# definition of function
###################################################

##### read dataset function #####
def read_dataset( csv_file_name ):
    '''This function is specified to kaggle competition, Facial Expression Recognition.
    input is csv file name which contains facial expression data in particular format
    outputs are 3 arrays
    1st is image data
    2nd is expression (0:anger, 1:Happy)
    3rd is usage data (0:for training, 1:for public test, 2:for privat test)
    '''
    with open( csv_file_name ) as f:
        reader = csv.reader( f )
        l = [row for row in reader]

    # create empty lists
    X = []
    Y = []
    Usage = []
    l.pop(0) # drop first column
    for row in l:
        # separate the expression result (7 categories)
        Y.append( int(row[0]) )
        # eDataMatrixract image data
        X.append([int(val) for val in row[1].split()])
        # tag for training or test
        if   row[2] == 'Training':
            Usage.append( 0 )
        elif row[2] == 'PublicTest':
            Usage.append( 1 )
        else:
            Usage.append( 2 )

    # change list to array
    X = np.array(X)
    Y = np.array(Y)
    Usage = np.array(Usage)

    return X, Y, Usage


##### LDA function #####
def LDA( DataMatrix, Label ):
    '''LDA is Linear Discriminant Analysis
    LDA makes several axes for re-measuring features
    input:
        DataMatrix : data matrix containing training data
                     the shape of this matrix should be N x M which
                     N is number of samples, and M is dimension of feature space
        Label : the category label for input DataMatrix
                each values are corresponding to each row of DataMatrix
                the label should be integer
    output:
        output is list of pairs, eigen values and eigen vectors in LDA
        the pairs are sorted so that the first eigen vector can be reached to index 0
    '''

    # make within class scatter matrix
    wcsMatrix = np.zeros( (DataMatrix.shape[1], DataMatrix.shape[1]) )
    for cat in range( np.max( Label ) + 1 ):
        tmpX  = DataMatrix[ Label == cat ]
        meanX = np.mean( tmpX, 0 )

        tmpX = tmpX - mlib.repmat( meanX, tmpX.shape[0], 1 )
        wcsMatrix += tmpX.T.dot( tmpX )

    # make between class scatter matrix
    bcsMatrix = np.zeros( (DataMatrix.shape[1], DataMatrix.shape[1]) )
    wholeMean = np.mean( DataMatrix, 0 )
    for cat in range( np.max( Label ) + 1 ):
        tmpX  = DataMatrix[ Label == cat ]
        meanX = np.mean( tmpX, 0 )

        tmpX  = ( wholeMean - meanX ).reshape( len(meanX), 1 )
        bcsMatrix += tmpX.dot( tmpX.T ) * np.sum( DataMatrix==cat )


    # calculate eigen values and vectors
    eval, evec = np.linalg.eig( np.linalg.inv( wcsMatrix ).dot( bcsMatrix ) )

    # sort eigen values with corresponding eigen vectors
    pairs = [ ( np.abs(eval[i]), evec[:,i] ) for i in range( len(eval) ) ]
    pairs = sorted( pairs, key=lambda x: x[0], reverse=True )
    return pairs


##################################################
# program body
##################################################

X, Y, Usage = read_dataset( 'fer2013/fer2013.csv' )

# normalize the value range X (0-255) to 0-1
X = X / 255

Xt = X[Usage==0,:]
Yt = Y[Usage==0]

# try Linear Discriminant Analysis for reducing big data dimensions
pairs = LDA( Xt, Yt )
eval = np.zeros( (Xt.shape[1], 1) )
evec = np.zeros( (Xt.shape[1], Xt.shape[1]) )
for i in range( len(pairs) ):
    eval[i]   = pairs[i][0]
    evec[:,i] = pairs[i][1]

ax1 = evec[:,0].T.dot( Xt.T ).real
ax2 = evec[:,1].T.dot( Xt.T ).real
ax3 = evec[:,2].T.dot( Xt.T ).real

# plt.scatter( ax1, ax2, c=Yt, cmap='rainbow', alpha=0.7, edgecolors='b' )
# plt.show()
# ↑全然だめだった。データがバラけることもない
# いくら何でも軸 2 つは無理があったか

# ということで 3 本目の軸をダメ元でやってみる
from mpl_toolkits import mplot3d as Axis3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D( ax1[:100]*100, ax2[:100]*100, ax3[:100]*100, \
            c=Yt[:100], cmap='rainbow', alpha=0.7, edgecolors='b' )
plt.show()
# 表示点数を 10 まで少なくしても表示されなかった (どこかに間違いがあるのか？)



