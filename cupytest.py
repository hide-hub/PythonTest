# test script for cpu vs gpu

import numpy as np
import cupy  as cp
from tictoc import tic, toc


# multiplicatioin test
def get_w_np( x, t ):
    xx     = np.dot( x, x.T )
    xx_inv = np.linalg.inv( xx )
    xt     = np.dot( x.T, t )
    w      = np.dot( xx_inv, xt )
    return w

def get_w_cp( x, t ):
    xx     = cp.dot( x, x.T )
    xx_inv = cp.linalg.inv( xx )
    xt     = cp.dot( x.T, t )
    w      = cp.dot( xx_inv, xt )
    return w

for N in [10, 100, 1000, 10000]:
    np.random.seed(0)
    x_cpu = np.random.randn( N, N )
    t_cpu = np.random.randn( N, 1 )
    x_gpu = cp.asarray( x_cpu )
    t_gpu = cp.asarray( t_cpu )
    tic()
    w = get_w_np( x_cpu, t_cpu )
    cputime = toc()
    tic()
    w = get_w_cp( x_gpu, t_gpu )
    gputime = toc()
    print( 'for {} by N matrix'.format(N, N) )
    print( 'cpu time is : {} [sec]'.format( cputime ) )
    print( 'gpu time is : {} [sec]'.format( gputime ) )
    print( '\n\n' )
## above code cannot executed to final N=10000
## the cupy reject the step at cp.dot() process
## maybe it's a limit of my gpu but I have to search the reason


# creation and multiplicatioin
def test(xp):
    a = xp.arange(1000000).reshape(1000,-1)
    return xp.dot( a, a.T )

## https://www.slideshare.net/ryokuta/cupy
## above site says following code makes cupy up its speed
#cp.cuda.set_allocator( cp.cuda.MemoryPool().malloc )

tic()
for i in range( 1000 ):
    b = test(np)

cputime = toc()

tic()
for i in range( 1000 ):
    b = test(cp)

gputime = toc()

print( 'cpu time is {} [sec]'.format( cputime ) )
print( 'gpu time is {} [sec]'.format( gputime ) )
