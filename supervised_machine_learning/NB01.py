# naive bayse estimation for minst dataset

from utils_minst import get_data
from utils_minst import get_test_data

import numpy             as np
import matplotlib.pyplot as plt

(X,Y) = get_data()

# organized data which has each component having categorized image data
D = []
for i in range(10):
  D.append([])

for i in range( len(Y) ):
  D[Y[i]].append( X[ i, : ] )

for i in range(10):
  D[i] = np.array( D[i] )


# calculate mean of the category
M = []
for i in range(10):
  M.append( D[i].mean(0) )

M = np.array( M )

# calculate variance of each pixel
var = []
for i in range( 10 ):
  diff = D[i] - M[i,:]
  var.append( np.diag( diff.T.dot(diff) ) / ( diff.shape[0] - 1 ) + 10**-3 )

var = np.array( var )

# mean is estimation of each pixel value
p = M









