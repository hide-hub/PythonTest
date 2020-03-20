# initial test program for K-nearest neighberk

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import numpy.matlib      as mlib
import scipy.spatial     as ss
from future.utils     import iteritems
from sortedcontainers import SortedList
from utils_minst      import get_data
from datetime         import datetime

class KNN( object ):
	def __init__( self, k ):
		self.k = k
	
	def fit( self, X, y ):
		self.X = X
		self.y = y
		self.tree = ss.KDTree( self.X, leafsize=10 )
	
	def predict( self, X ):
		y = np.zeros( len(X) )
		for i, x in enumerate( X ):
			sl = SortedList()
			for j, xt in enumerate( self.X ):
				diff = x - xt
				d    = diff.dot( diff )
				if len( sl ) < self.k:
					sl.add( (d, self.y[j]) )
				else:
					if d < sl[-1][0]:
						del sl[-1]
						sl.add( (d, self.y[j]) )
			
			votes = {}
			for _, v in sl:
				votes[v] = votes.get( v, 0 ) + 1
			max_votes = 0
			max_votes_class = -1
			for v, count in iteritems( votes ):
				if count > max_votes:
					max_votes       = count
					max_votes_class = v
			y[i] = max_votes_class
		return y
	
	def score( self, X, Y ):
		P = self.predict( X )
		return np.mean( P==Y )


# improve test using linear algebra
class KNN_lin( object ):
	def __init__( self, k ):
		self.k = k
	
	def fit( self, X, y ):
		self.X = X
		self.y = y
		self.tree = ss.KDTree( self.X, leafsize=10 )
	
	def predict( self, X ):
		y = np.zeros( len(X) )
		for i, x in enumerate( X ):
			dist, idx = self.tree.query( x, self.k )
			
			sl = SortedList()
			for a in range( len(idx) ):
				sl.add( (dist[a], self.y[ idx[a] ]) )
			
			votes = {}
			for _, v in sl:
				votes[v] = votes.get( v, 0 ) + 1
			
			max_votes = 0
			max_votes_class = -1
			for v, count in iteritems( votes ):
				if count > max_votes:
					max_votes       = count
					max_votes_class = v
			y[i] = max_votes_class
			# votes = np.histogram( self.y[idx], list( range( int(self.y[idx].max()) + 2 ) ) )
			# y[i]  = np.argmax( votes[0] )
		return y
	
	def score( self, X, Y ):
		P = self.predict( X )
		return np.mean( P==Y )


if __name__ == '__main__':
	X, Y   = get_data( 2000 )
	Ntrain = 1000
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest,  Ytest  = X[Ntrain:], Y[Ntrain:]
	for k in (1,2,3,4,5):
		knn = KNN( k )
		t0 = datetime.now()
		knn.fit( Xtrain, Ytrain )
		print( 'Training Time :', (datetime.now() - t0) )

		t0 = datetime.now()
		print( 'Train accuracy :', knn.score( Xtrain, Ytrain ) )
		print( 'Time to compute train accuracy :', (datetime.now() - t0) )
		print( 'Train size :', len( Ytrain ) )

		t0 = datetime.now()
		print( 'Test accuracy :', knn.score( Xtest, Ytest ) )
		print( 'Time to compute test accuracy :', (datetime.now() - t0) )
		print( 'Train size :', len( Ytest ) )


# アウトプット↓
# Training Time : 0:00:00.000010
# Train accuracy : 1.0
# Time to compute train accuracy : 0:00:05.064430
# Train size : 1000

# Test accuracy : 0.0
# Time to compute test accuracy : 0:03:28.596044
# Train size : 1000

# Training Time : 0:00:00.000003
# Train accuracy : 1.0
# Time to compute train accuracy : 0:00:05.078711
# Train size : 1000

# Test accuracy : 0.0
# Time to compute test accuracy : 0:03:28.061883
# Train size : 1000

# Training Time : 0:00:00.000002
# Train accuracy : 0.957
# Time to compute train accuracy : 0:00:05.040962
# Train size : 1000

# Test accuracy : 0.0
# Time to compute test accuracy : 0:03:25.217660
# Train size : 1000

# Training Time : 0:00:00.000002
# Train accuracy : 0.956
# Time to compute train accuracy : 0:00:05.036537
# Train size : 1000

# Test accuracy : 0.0
# Time to compute test accuracy : 0:03:26.421376
# Train size : 1000

# Training Time : 0:00:00.000021
# Train accuracy : 0.934
# Time to compute train accuracy : 0:00:05.039429
# Train size : 1000

# Test accuracy : 0.0
# Time to compute test accuracy : 0:03:26.259713
# Train size : 1000




