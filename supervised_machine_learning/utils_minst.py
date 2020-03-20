# this program is utility functions for minst data

import numpy  as np
import pandas as pd


def get_data( limit=None ):
  print( 'Reading in and transforming data...' )
  df   = pd.read_csv( 'Data/train.csv' )
  data = df.as_matrix()
  np.random.shuffle( data )
  X = data[:, 1:] / 255
  Y = data[:, 0]
  if limit is not None:
    X, Y = X[:limit, :], Y[:limit]
  return X, Y

def get_test_data( limit=None ):
  print( 'Reading in and transforming data...' )
  df   = pd.read_csv( 'Data/test.csv' )
  data = df.as_matrix()
  np.random.shuffle( data )
  X = data/255
  if limit is not None:
    X = X[:limit, :]
  return X

def get_xor():
  X = np.zeros( (200, 2) )
  X[:50]     = np.random.random( (50, 2) ) / 2 + 0.5                  # (0.5-1, 0.5-1)
  X[50:100]  = np.random.random( (50, 2) ) / 2                        # (0-0.5, 0-0.5)
  X[100:150] = np.random.random( (50, 2) ) / 2 + np.array([ 0, 0.5 ]) # (0-0.5, 0.5-1)
  X[150:]    = np.random.random( (50, 2) ) / 2 + np.array([ 0.5, 0 ]) # (0.5-1, 0-0.5)
  Y = np.array( [0]*100 + [1]*100 )
  return X, Y


def get_donut():
  halfN = 100
  N = halfN * 2
  R_inner = 5
  R_outer = 10

  # distance from origiin is radius + random normal
  # angle theta is uniformly distribution between (0, 2pi)
  R1 = np.random.randn( halfN ) + R_inner
  theta = 2*np.pi * np.random.random( halfN )
  X_inner = np.concatenate([ [R1 * np.cos( theta )], [R1 * np.sin( theta )] ]).T

  R2 = np.random.randn( halfN ) + R_outer
  theta = 2*np.pi * np.random.random( halfN )
  X_outer = np.concatenate([ [R2 * np.cos( theta )], [R2 * np.sin( theta )] ]).T

  X = np.concatenate([ X_inner, X_outer ])
  Y = np.array( [0]*(halfN) + [1]*(halfN) )

  return X, Y


