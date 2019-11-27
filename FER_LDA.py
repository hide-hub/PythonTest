# facial expression recognition test
# facial data is provided by kaggle challenge project
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
# the code for recognition is for udemy lazyprogrammer's project

# this is a just test and practice program for data science
# The 1st try is LDA (Linear Discriminant Analysis)

import csv
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy      import matlib as mlib
from utils_fer   import read_dataset
from utils_fer   import LDA

##################################################
# program body
##################################################

X, Y, Usage = read_dataset( 'fer2013/fer2013.csv' )

# normalize the value range X (0-255) to 0-1
X = X / 255

Xt = X[Usage==0,:]
Yt = Y[Usage==0]

## 1st try
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
ax.scatter3D( ax1[:200], ax2[:200], ax3[:200], \
            c=Yt[:200], cmap='rainbow', alpha=0.3, edgecolors='b' )
plt.show()
# ↑全く分離できていない。結論として LDA は基本的に 2 クラス分類の手法だと考えている。
# tmpY = Yt!=0 を実行して 2 クラス分類としてやってみたが 3 次元程度では分けるのが難しいのか、
# 目立った効果は感じられなかった


