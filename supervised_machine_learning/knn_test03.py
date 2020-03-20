# KNN method for xor data

import numpy  as np
import pandas as pd
from matplotlib  import pyplot as plt
from numpy       import matlib as mlib
from utils_minst import get_xor
from mpl_toolkits.mplot3d import Axes3D

from knn_test01  import KNN
from utils_minst import get_xor
from utils_minst import get_donut

if __name__ == '__main__':
  X, Y = get_xor()

  plt.scatter( X[:,0], X[:,1], s=100, c=Y, alpha=0.5 )
  plt.show()

  model = KNN(3)
  model.fit( X, Y )

  print( '[xor data] Train accuracy :', model.score( X, Y ) )
  # train accuracy is pretty good

  X, Y = get_donut()

  plt.scatter( X[:,0], X[:,1], s=100, c=Y, alpha=0.5 )
  plt.show()

  model = KNN(3)
  model.fit( X, Y )

  print( '[donut data] Train accuracy :', model.score( X, Y ) )
  # train accuracy is pretty good


