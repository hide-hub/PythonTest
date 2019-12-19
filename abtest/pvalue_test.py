# p value test program lectured in Udemy course, Baysian Machine Learning.

import numpy as np
from scipy import stats

N = 10
a = np.random.randn( N ) + 2.0    # center is 2.0
b = np.random.randn( N )        # center is 0

# variance of each group
var_a = a.var( ddof=1 )
var_b = b.var( ddof=1 )

# pooled variance (cause both group are same size)
s = np.sqrt( ( var_a + var_b ) / 2.0 )

# t test
t = ( a.mean() - b.mean() ) / ( s * np.sqrt( 2.0/N ) )

# degrees of freedom
df = 2*N - 2

# p value
p = 1 - stats.t.cdf( t, df=df )

print( "t:\t", t, "p:\t", 2*p )

# built-in staticstics function contained in scipy
t2, p2 = stats.ttest_ind( a, b )
print( "t2:\t", t2, "p2:\t", p2 )   # p value is already doubled


