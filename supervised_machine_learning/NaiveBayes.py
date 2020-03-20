
import numpy as np
from utils_minst import get_data
from datetime    import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):
  def fit( self, X, Y, smoothing=10e-3 ):
    self.gaussians = dict()
    self.priors    = dict()
    labels = set( Y )
    for c in labels:
      current_x = X[ Y==c ]
      self.gaussians[c] = {
        'mean': current_x.mean( axis=0 ),
        'var':  current_x.var( axis=0 ) + smoothing
      }
      self.priors[c] = float( len( Y[Y==c] ) / len( Y ) )
  
  def score( self, X, Y ):
    P = self.predict( X )
    return np.mean( P==Y )
  
  def predict( self, X ):
    N, D = X.shape
    K = len( self.gaussians )
    P = np.zeros( (N, K) )
    for c, g in self.gaussians.items():
      mean, var = g['mean'], g['var']
      P[:,c] = mvn.logpdf( X, mean=mean, cov=var ) + np.log( self.priors[c] )
    return np.argmax( P, axis=1 )


if __name__ == "__main__":
    X, Y = get_data()
    Ntrain = int( len(Y) / 2 )
    Xtrain = X[:Ntrain, :]
    Ytrain = Y[:Ntrain]
    Xtest  = X[Ntrain:, :]
    Ytest  = Y[Ntrain:]

    model = NaiveBayes()
    t0 = datetime.now()
    model.fit( Xtrain, Ytrain )
    print( 'Training Time : {}'.format( (datetime.now() - t0 ) ) )

    t0 = datetime.now()
    print( 'Train accuracy : {}'.format( model.score( Xtrain, Ytrain ) ) )
    print( 'Time to compute train accuracy : {}'.format( datetime.now() - t0 ) )
    print( 'Train size : {}'.format( len( Ytrain ) ) )

    t0 = datetime.now()
    print( 'Test accuracy : {}'.format( model.score( Xtest, Ytest ) ) )
    print( 'Time to compute test accuracy : {}'.format( datetime.now() - t0 ) )
    print( 'Test size : {}'.format( len( Ytest ) ) )



