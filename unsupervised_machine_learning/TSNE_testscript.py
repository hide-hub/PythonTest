# t-SNE test script
# most of following code is copied from t_sne.py in the package skearn
# this code is for easy to understand the concept or algorithm of t-SNE

import numpy             as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold     import TSNE
from sklearn.manifold     import _utils
from sklearn.neighbors    import NearestNeighbors
from sklearn.utils        import check_random_state
from scipy.sparse         import csr_matrix


# flag for checking data distribution
CHECK_DATA = True

# create 3D data points
D = 3
N = 900
s = 4

mu1 = np.array( [ 0, 0, 0 ] )
mu2 = np.array( [ s, s, 0 ] )
mu3 = np.array( [ 0, 0, s ] )

X = np.zeros( (N, D) )
X[ :300, : ]    = np.random.randn( 300, 3 ) + mu1
X[ 300:600, : ] = np.random.randn( 300, 3 ) + mu2
X[ 600:, : ]    = np.random.randn( 300, 3 ) + mu3

col = np.random.random( (3,3) )

if CHECK_DATA:
  fig = plt.figure()
  ax  = fig.add_subplot( 111, projection='3d' )
  for i in range( 3 ):
    ax.scatter( X[300*i:300*(i+1), 0], X[300*i:300*(i+1), 1], X[300*i:300*(i+1), 2], c=col[i,:] )
  plt.show()



# prepare t-SNE parameters (default for TSNE class)
n_components            = 2
perplexity              = 30.0
early_exaggeration      = 12.0
learning_rate           = 200.0
n_iter                  = 1000
n_iter_without_progress = 300
min_grad_norm           = 1e-7
metric                  = "euclidean"
init                    = "random"
verbose                 = 0
random_state            = None
method                  = 'barnes_hut'
angle                   = 0.5

n_samples    = X.shape[0]
neighbors_nn = None
k = min(n_samples - 1, int(3. * perplexity + 1))
knn = NearestNeighbors( algorithm='auto', n_neighbors=k, metric=metric )
knn.fit( X )
distances_nn, neighbors_nn = knn.kneighbors( None, n_neighbors=k )
del knn
distances_nn **= 2


# in the function _joint_probabilities_nn ----from here
distances = distances_nn.astype( np.float32, copy=True )
neighbors = neighbors_nn.astype( np.int64, copy=True )
conditional_P = _utils._binary_search_perplexity( distances, neighbors, perplexity, verbose )
P = csr_matrix( (conditional_P.ravel(), neighbors.ravel(), range(0, n_samples * k + 1, k) ), shape=( n_samples, n_samples ) )
P = P + P.T
MACHINE_EPSILON = np.finfo(np.double).eps
sum_P = np.maximum( P.sum(), MACHINE_EPSILON )
P /= sum_P
# in the function _joint_probabilities_nn ----till here


# simplified _binary_search_perplexity() from github ---- from here
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/_utils.pyx
sqdistances = distances
desired_entropy = np.log( perplexity )
PERPLEXITY_TOLEARANCE = 1e-5
Pi = np.zeros( sqdistances.shape )
beta_hist = []
beta_sum = 0
for i in range( n_samples ):
  beta = 1. # beta is inverse of 1/2σ^2
  beta_min = -np.NINF
  beta_max =  np.NINF
  n_steps = 100
  for l in range( n_steps ):
    Pi[i,:] = np.exp( -sqdistances[ i,: ] * beta )
    sum_Pi = np.sum( Pi[ i,: ] )
    Pi[i,:] /= sum_Pi
    sum_disti_Pi = np.sum( sqdistances[ i,: ] * Pi[ i,: ] )
    entropy = np.log( sum_Pi ) + beta * sum_disti_Pi
    entropy_diff = entropy - desired_entropy
    if entropy_diff > 0.0:
      beta_min = beta
      if beta_max == np.NINF:
        beta *= 2.0
      else:
        beta = ( beta + beta_max ) / 2.0
    else:
      beta_max = beta
      if beta_min == -np.NINF:
        beta /= 2.0
      else:
        beta = ( beta + beta_min ) / 2.0
  beta_sum += beta
  beta_hist.append( beta )
# simplified _binary_search_perplexity() from github ---- till here

random_state = check_random_state(random_state)

X_embedded = 1e-4 * random_state.randn( n_samples, n_components ).astype( np.float32 )

params = X_embedded.ravel()

_N_ITER_CHECK       = 50
min_grad_norm       = 1e-7
skip_num_points     = 0
degrees_of_freedom  = max(n_components - 1, 1)
_EXPLORATION_N_ITER = 250
opt_args = {
    "it": 0,
    "n_iter_check": _N_ITER_CHECK,
    "min_grad_norm": min_grad_norm,
    "learning_rate": learning_rate,
    "verbose": verbose,
    "kwargs": dict(skip_num_points=skip_num_points),
    "args": [P, degrees_of_freedom, n_samples, n_components],
    "n_iter_without_progress": _EXPLORATION_N_ITER,
    "n_iter": _EXPLORATION_N_ITER,
    "momentum": 0.5,
  }

obj_func = _kl_divergence_bh
opt_args['kwargs']['angle'] = angle
# Repeat verbose argument for _kl_divergence_bh
opt_args['kwargs']['verbose'] = verbose


# iteration for convergence ---- from here

# Learning schedule (part 1): do 250 iteration with lower momentum but
# higher learning rate controlled via the early exageration parameter
P *= early_exaggeration
params, kl_divergence, it = _gradient_descent(obj_func, params,
                                              **opt_args)


# Learning schedule (part 2): disable early exaggeration and finish
# optimization with a higher momentum at 0.8
P /= early_exaggeration
remaining = n_iter - _EXPLORATION_N_ITER
if it < _EXPLORATION_N_ITER or remaining > 0:
  opt_args['n_iter']   = n_iter
  opt_args['it']       = it + 1
  opt_args['momentum'] = 0.8
  opt_args['n_iter_without_progress'] = n_iter_without_progress
  params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                **opt_args)
# iteration for convergence ---- till here

# Save the final number of iterations
n_iter_ = it
X_embedded = params.reshape(n_samples, self.n_components)
kl_divergence_ = kl_divergence



def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True):
  """t-SNE objective function: KL divergence of p_ijs and q_ijs.
  #
  Uses Barnes-Hut tree methods to calculate the gradient that
  runs in O(NlogN) instead of O(N^2)
  #
  Parameters
  ----------
  params : array, shape (n_params,)
      Unraveled embedding.
  #
  P : csr sparse matrix, shape (n_samples, n_sample)
      Sparse approximate joint probability matrix, computed only for the
      k nearest-neighbors and symmetrized.
  #
  degrees_of_freedom : int
      Degrees of freedom of the Student's-t distribution.
  #
  n_samples : int
      Number of samples.
  #
  n_components : int
      Dimension of the embedded space.
  #
  angle : float (default: 0.5)
      This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
      'angle' is the angular size (referred to as theta in [3]) of a distant
      node as measured from a point. If this size is below 'angle' then it is
      used as a summary node of all points contained within it.
      This method is not very sensitive to changes in this parameter
      in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
      computation time and angle greater 0.8 has quickly increasing error.
  #
  skip_num_points : int (optional, default:0)
      This does not compute the gradient for points with indices below
      `skip_num_points`. This is useful when computing transforms of new
      data where you'd like to keep the old data fixed.
  #
  verbose : int
      Verbosity level.
  #
  compute_error: bool (optional, default:True)
      If False, the kl_divergence is not computed and returns NaN.
  #
  Returns
  -------
  kl_divergence : float
      Kullback-Leibler divergence of p_ij and q_ij.
  #
  grad : array, shape (n_params,)
      Unraveled gradient of the Kullback-Leibler divergence with respect to
      the embedding.
  """
  params = params.astype(np.float32, copy=False)
  X_embedded = params.reshape(n_samples, n_components)
  #
  val_P = P.data.astype(np.float32, copy=False)
  neighbors = P.indices.astype(np.int64, copy=False)
  indptr = P.indptr.astype(np.int64, copy=False)
  #
  grad = np.zeros(X_embedded.shape, dtype=np.float32)
  error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                    grad, angle, n_components, verbose,
                                    dof=degrees_of_freedom,
                                    compute_error=compute_error)
  c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
  grad = grad.ravel()
  grad *= c
  #
  return error, grad






def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
  """Batch gradient descent with momentum and individual gains.
  #
  Parameters
  ----------
  objective : function or callable
      Should return a tuple of cost and gradient for a given parameter
      vector. When expensive to compute, the cost can optionally
      be None and can be computed every n_iter_check steps using
      the objective_error function.
  #
  p0 : array-like, shape (n_params,)
      Initial parameter vector.
  #
  it : int
      Current number of iterations (this function will be called more than
      once during the optimization).
  #
  n_iter : int
      Maximum number of gradient descent iterations.
  #
  n_iter_check : int
      Number of iterations before evaluating the global error. If the error
      is sufficiently low, we abort the optimization.
  #
  n_iter_without_progress : int, optional (default: 300)
      Maximum number of iterations without progress before we abort the
      optimization.
  #
  momentum : float, within (0.0, 1.0), optional (default: 0.8)
      The momentum generates a weight for previous gradients that decays
      exponentially.
  #
  learning_rate : float, optional (default: 200.0)
      The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
      the learning rate is too high, the data may look like a 'ball' with any
      point approximately equidistant from its nearest neighbours. If the
      learning rate is too low, most points may look compressed in a dense
      cloud with few outliers.
  #
  min_gain : float, optional (default: 0.01)
      Minimum individual gain for each parameter.
  #
  min_grad_norm : float, optional (default: 1e-7)
      If the gradient norm is below this threshold, the optimization will
      be aborted.
  #
  verbose : int, optional (default: 0)
      Verbosity level.
  #
  args : sequence
      Arguments to pass to objective function.
  #
  kwargs : dict
      Keyword arguments to pass to objective function.
  #
  Returns
  -------
  p : array, shape (n_params,)
      Optimum parameters.
  #
  error : float
      Optimum.
  #
  i : int
      Last iteration.
  """
  if args is None:
      args = []
  if kwargs is None:
      kwargs = {}
  #
  p = p0.copy().ravel()
  update = np.zeros_like(p)
  gains = np.ones_like(p)
  error = np.finfo(np.float).max
  best_error = np.finfo(np.float).max
  best_iter = i = it
  #
  tic = time()
  for i in range(it, n_iter):
      check_convergence = (i + 1) % n_iter_check == 0
      # only compute the error when needed
      kwargs['compute_error'] = check_convergence or i == n_iter - 1
      #
      error, grad = objective(p, *args, **kwargs)
      grad_norm = linalg.norm(grad)
      #
      inc = update * grad < 0.0
      dec = np.invert(inc)
      gains[inc] += 0.2
      gains[dec] *= 0.8
      np.clip(gains, min_gain, np.inf, out=gains)
      grad *= gains
      update = momentum * update - learning_rate * grad
      p += update
      #
      if check_convergence:
          toc = time()
          duration = toc - tic
          tic = toc
          #
          if verbose >= 2:
              print("[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration))
          #
          if error < best_error:
              best_error = error
              best_iter = i
          elif i - best_iter > n_iter_without_progress:
              if verbose >= 2:
                  print("[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress))
              break
          if grad_norm <= min_grad_norm:
              if verbose >= 2:
                  print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm))
              break
  #
  return p, error, i

