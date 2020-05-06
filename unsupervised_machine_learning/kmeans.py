
import numpy             as np
import numpy.matlib      as mlib
import matplotlib.pyplot as plt

def d( u, v, sigma ):
  diff = u - v
  if np.size( sigma ) == 1:
    inv_sigma = sigma
  else:
    inv_sigma = np.linalg.inv( sigma )
  return diff.dot( inv_sigma ).dot( diff )

def cost( X, R, M, sigma=1.0 ):
  cost = 0
  for k in range( len(M) ):
    for n in range( len(X) ):
      cost += R[n,k] * d( M[k], X[n], sigma )
  return cost

# pattern 1 for k-means clustering fails
def donut():
  N = 1000
  D = 2
  Nhalf = int( N/2 )
  #
  R_inner = 5
  R_outer = 10
  #
  R1 = np.random.randn( Nhalf ) + R_inner
  theta = 2*np.pi * np.random.randn( Nhalf )
  Xinner = np.concatenate([ [R1 * np.cos( theta )], [R1 * np.sin( theta )] ]).T
  #
  R2 = np.random.randn( Nhalf ) + R_outer
  theta = 2*np.pi * np.random.randn( Nhalf )
  Xouter = np.concatenate([ [R2 * np.cos( theta )], [R2 * np.sin( theta )] ]).T
  #
  X = np.concatenate([ Xinner, Xouter ])
  return X

# pattern 2 for k-means clustering fails
def highvariance2class():
  N = 1000
  D = 2
  Nhalf = int( N/2 )
  #
  X = np.zeros( (N, D) )
  X[:Nhalf, :] = np.random.multivariate_normal( [0,0], [[1,0], [0,20]], Nhalf )
  X[Nhalf:, :] = np.random.multivariate_normal( [5,0], [[1,0], [0,20]], Nhalf )
  #
  return X

# pattern 3 for k-means clustering fails
def unbalance2class():
  N = 1000
  D = 2
  Nhalf = int( 1000/20 * 19 )
  #
  X = np.zeros( (N, D) )
  X[:Nhalf, :] = np.array([0, 0]) + np.random.randn( Nhalf, 2 )
  X[Nhalf:, :] = np.array([5, 0]) + np.random.randn( N-Nhalf, 2 )
  #
  return X

def plot_k_means( X, K, max_iter=20, sigma=1.0 ):
  #
  N, D = X.shape
  M = np.zeros( ( K, D ) )
  R = np.zeros( ( N, K ) )
  #
  if np.size( sigma ) == 1:
    sigma = np.eye( D ) * sigma
  else:
    if np.sqrt( np.size( sigma ) ) != D:
      print( 'sigma sould be {}x{} matrix'.format( D ) )
      return 0
  #
  grid_width  = 5
  grid_height = max_iter / grid_width
  random_colors = np.random.random( (K, 3) )
  plt.figure()
  #
  for k in range( K ):
    M[k] = X[ np.random.choice(N), : ]
  #
  costs = np.zeros( max_iter )
  for i in range( max_iter ):
    for k in range( K ):
      for n in range( N ):
        R[n,k] = np.exp( -1*d(M[k,:], X[n,:], sigma) ) / np.sum( np.exp( -1*d(M[j,:], X[n,:], sigma) ) for j in range(K) )
    #
    colors = R.dot( random_colors )
    plt.subplot( grid_width, grid_height, i+1 )
    plt.scatter( X[:,0], X[:,1], c=colors )
    plt.scatter( M[:,0], M[:,1], marker='x' )
    print( cost( X, R, M ) )
    #
    for k in range( K ):
      M[k,:] = R[:,k].T.dot( X ) / R[:,k].sum()
    #
    costs[i] = cost( X, R, M )
    if i > 0:
      if np.abs( costs[i] - costs[i-1] ) < 0.1:
        break
  #
  plt.show()  # show detail of conversion
  #
  plt.plot( costs )
  plt.title( 'Costs' )
  plt.show()
  #
  colors = R.dot( random_colors  )
  plt.scatter( X[:,0], X[:,1], c=colors )
  plt.scatter( M[:,0], M[:,1], marker='x' )
  plt.show()


def main():
  X1 = donut()
  X1 = np.concatenate([ X1.T, [np.diag( X1.dot(X1.T) ) ]]).T
  plot_k_means( X1, 2 )
  return

  D = 2
  s = 4
  mu1 = np.array( [0,0] )
  mu2 = np.array( [s,s] )
  mu3 = np.array( [0,s] )
  #
  N = 900
  X = np.zeros( (N,D) )
  X[:300, :]    = np.random.randn( 300, D ) + mu1
  X[300:600, :] = np.random.randn( 300, D ) + mu2
  X[600:, :]    = np.random.randn( 300, D ) + mu3
  #
  plt.scatter( X[:,0], X[:,1] )
  plt.show()
  #
  K = 3
  plot_k_means( X, K )
  #
  K = 5
  plot_k_means( X, K, max_iter=30 )
  #
  K = 5
  plot_k_means( X, K, max_iter=30, sigma=0.3 )


if __name__ == "__main__":
    main()



