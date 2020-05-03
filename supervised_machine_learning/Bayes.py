
import numpy as np
import matplotlib.pyplot as plt
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
  
  def get_error_idx( self, X, Y ):
    P = self.predict( X )
    return ( P!=Y )
  
  def predict( self, X ):
    N, D = X.shape
    K = len( self.gaussians )
    P = np.zeros( (N, K) )
    for c, g in self.gaussians.items():
      mean, var = g['mean'], g['var']
      P[:,c] = mvn.logpdf( X, mean=mean, cov=var ) + np.log( self.priors[c] )
    return np.argmax( P, axis=1 )

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


# test program
# this program class comppare the each category vs else categories
# that is, each category probability is calculated normally
# but the else categories (ex. class0 vs class1-9 in minst case)
class testBayes(object):
  def fit( self, X, Y, smoothing=10e-3 ):
    N, D = X.shape
    self.gaussians = dict()
    self.others_gaussians = dict()
    self.priors       = dict()
    self.other_priors = dict()
    labels = set( Y )
    for c in labels:
      current_x = X[ Y==c ]
      others    = X[ Y!=c ]
      self.gaussians[c] = {
        'mean': current_x.mean( axis=0 ),
        'cov':  np.cov( current_x.T ) + np.eye( D ) * smoothing
      }
      self.others_gaussians[c] = {
        'mean': others.mean( axis=0 ),
        'cov':  np.cov( others.T ) + np.eye( D ) * smoothing
      }
      self.priors[c]       = float( len( Y[Y==c] ) / len(Y) )
      self.other_priors[c] = 1 - self.priors[c]
    
  def score( self, X, Y ):
    P = self.predict( X )
    return np.mean( P==Y )
  
  def predict( self, X ):
    N, D = X.shape
    K = len( self.gaussians )
    P  = np.zeros( (N, K) )
    oP = np.zeros( (N, K) )
    for c in range( K ):
      g  = self.gaussians[c]
      og = self.others_gaussians[c]
      mean,  cov  = g['mean'],  g['cov']
      omean, ocov = og['mean'], og['cov']
      P[:,c]  = mvn.logpdf( X, mean=mean,  cov=cov )  + np.log( self.priors[c] )
      oP[:,c] = mvn.logpdf( X, mean=omean, cov=ocov ) + np.log( self.other_priors[c] )
    print( P )
    print( oP )
    return np.argmax( P, axis=1 ), np.argmax( oP, axis=1 ), P, oP



if __name__ == "__main__":
  X, Y = get_data( 10000 )
  Ntrain = int( len(Y) / 2 )
  Xtrain = X[:Ntrain, :]
  Ytrain = Y[:Ntrain]
  Xtest  = X[Ntrain:, :]
  Ytest  = Y[Ntrain:]

  naive_model = NaiveBayes()
  nonnaive_model = Bayes()
  t0 = datetime.now()
  naive_model.fit( Xtrain, Ytrain )
  print( 'Training Time : {}'.format( (datetime.now() - t0 ) ) )

  t0 = datetime.now()
  print( 'Train accuracy : {}'.format( naive_model.score( Xtrain, Ytrain ) ) )
  print( 'Time to compute train accuracy : {}'.format( datetime.now() - t0 ) )
  print( 'Train size : {}'.format( len( Ytrain ) ) )

  t0 = datetime.now()
  print( 'Test accuracy : {}'.format( naive_model.score( Xtest, Ytest ) ) )
  print( 'Time to compute test accuracy : {}'.format( datetime.now() - t0 ) )
  print( 'Test size : {}'.format( len( Ytest ) ) )


  t0 = datetime.now()
  nonnaive_model.fit( Xtrain, Ytrain )
  print( 'Training Time : {}'.format( (datetime.now() - t0 ) ) )

  t0 = datetime.now()
  print( 'Train accuracy : {}'.format( nonnaive_model.score( Xtrain, Ytrain ) ) )
  print( 'Time to compute train accuracy : {}'.format( datetime.now() - t0 ) )
  print( 'Train size : {}'.format( len( Ytrain ) ) )

  t0 = datetime.now()
  print( 'Test accuracy : {}'.format( nonnaive_model.score( Xtest, Ytest ) ) )
  print( 'Time to compute test accuracy : {}'.format( datetime.now() - t0 ) )
  print( 'Test size : {}'.format( len( Ytest ) ) )



