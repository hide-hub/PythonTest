# this utility file contains plot functions normally provided by R

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import numpy.matlib      as mlib


# # this normalization is not completed
# # I forgot something important
# def normalize( X ):
#   '''
#   input :
#       Data for normalization
#       Data sould be N x M matrix and N > M
#       which means each row is each record of data
#   output :
#       normalize matrix (B) which change the Data to normalized data
#       the normalized data (X) has the attribute of X.T * X is nearly identity matrix
#       the normalized matrix X is calculated as Data * B
#   memo :
#       this function is useful for principal compornent analysis
#       identity matrix is really stable for eigen value decomposition
#   '''
#   eval, evec = np.linalg.eig( X.T.dot(X) )
#   eval = np.diag( eval )
#   return evec.dot( np.linalg.inv( np.sqrt(eval) ) )


def Show_Residuals_vs_Fitted( X, Y, b ):
  """ plot function for linear regression
  this function shows residual vs fitted value for all records in X and Y

  Arguments:
      X {ndarray} -- the data matrix
      Y {ndarray} -- the target values
      b {ndarray} -- linear regrassion coefficients
  """
  FittedValue = X.dot( b )
  Residuals   = Y - FittedValue
  plt.scatter( FittedValue, Residuals )
  plt.grid( b=None, which='both', axis='both' )
  plt.show()


def Show_NormalQ_Q( X, Y, b, step=10**5 ):
  """ plot function for Q-Q plot
  this function shows Q-Q plot
  it's difficult to explain detail of Q-Q plot shortly
  for detail of the Q-Q plot, following web site is easy to understand
  https://qiita.com/kenmatsu4/items/59605dc745707e8701e0

  Arguments:
      X {ndarray} -- the data matrix
      Y {ndarray} -- the target values
      b {ndarray} -- linear regrassion coefficients
      step {int}  -- step for cumulative normal distribution
  """
  # create theoretical normal distribution
  mag  = step
  w    = 6
  sample  = np.array( list( range(-w*mag, w*mag, round(w*2*mag/step)) ) ) / mag
  nDist   = 1/np.sqrt( 2*np.pi ) * np.exp( sample**2/-2 )
  norCum  = np.cumsum( nDist * w*2/step )

  # calcurate Hat matrix
  # Hat marix is projection matrix from Y to Y_hat
  # Xb = Y means b = inv(X.T * X) * X.T
  # so substituting inv(X.T*X)*X.T to b
  # X * inv(X.T*X) * X.T * Y = Y_hat (estimation of Y)
  # which is Hat matrix (project Y to Y_hat) is X * inv(X.T*X) * X.T
  # Hat * Y = Y_hat
  tmp = X.values
  Hat = tmp.dot( np.linalg.inv( tmp.T.dot(tmp) ) ).dot( tmp.T )
  h   = np.diag( Hat )  # diagonal values in Hat is important for standardization

  # standardized residual
  Residuals = Y - X.dot( b )
  stdRed    = Residuals / ( np.sqrt( np.var(Residuals, ddof=1) ) * np.sqrt( 1 - h ) )

  # sorted data and its indices
  N = len( stdRed )  # number of data records
  sortedStdRed = sorted( stdRed )
  idx = np.array( list(range( 0, N+2 )) ) / (N+1)
  idx = idx[1:-1]

  # x and y for Q-Q plot
  diff = mlib.repmat( norCum, len(idx), 1 ) - mlib.repmat( idx, len(norCum), 1 ).T
  x = sample[ np.argmin( diff**2, 1 ) ]
  y = sortedStdRed

  plt.scatter( x, y )
  plt.grid( b=None, which='major', axis='both' )
  plt.show()



if __name__ == "__main__":
  Data = pd.read_csv( 'medicalfee_vs_lifetime.csv' )
  Data['bias'] = np.ones( len(Data) )
  X = Data[['bias', 'Fee']]
  Y = Data['Time']

  Q = np.linalg.inv( X.T.dot(X) ).dot( X.T )
  b = Q.dot( Y )
  residual = Y - X.dot( b )



