
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib  as mlib
from utils_far import read_dataset
from datetime    import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
  def fit( self, X, Y, smoothing=10e-3 ):
    N, D = X.shape
    self.gaussians = dict()
    self.priors    = dict()
    labels = set( Y )
    for c in labels:
      current_x = X[ Y==c ]
      self.gaussians[c] = {
        'mean': current_x.mean( axis=0 ),
        'cov':  np.cov( current_x.T ) + np.eye(D) * smoothing
      }
      self.priors[c] = float( len( Y[Y==c] ) / len( Y ) )
  
  def score( self, X, Y ):
    P = self.predict( X )
    return np.mean( P==Y )
  
  def get_error_idx( self, X, Y ):
    P = self.predict( X )
    return ( P!=Y )
  
  def predict( self, X ):
    N, D = X.shape
    K = len( self.gaussians )
    P = np.zeros( (N, K) )
    for c, g in self.gaussians.items():
      mean, cov = g['mean'], g['cov']
      P[:,c] = mvn.logpdf( X, mean=mean, cov=cov ) + np.log( self.priors[c] )
    return np.argmax( P, axis=1 )


if __name__ == "__main__":
    X, Y, Usage = read_dataset( 'fer2013/fer2013.csv' )

    Xtrain = X[ Usage==0, : ]
    Ytrain = Y[ Usage==0 ]

    Xtest = X[ Usage==1,: ]
    Ytest = Y[ Usage==1 ]

    model = Bayes()
    model.fit( Xtrain, Ytrain )

    # check the correlation of each class and each pixel
    label = set(Y)
    dim = len( Xtrain[0,:] )
    corrX = np.zeros( (len(label), dim, dim )
    for c in label:
      tmpX = X[ Y==c ]
      covX = np.cov( tmpX.T )
      d = np.sqrt( np.diag( covX ) )
      corrX[c, :,:] = covX / d.dot( d.T )
    
    for c in label:
      plt.imshow( corrX[c] )
      plt.show()






