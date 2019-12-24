# practice program for a/b testing

import csv
import numpy as np
from scipy import stats

a = []
b = []
with open( "advertisement_clicks.csv" ) as f:
    reader = csv.reader( f )
    for row in reader:
        if row[0] == "A":
            a.append( int( row[1] ) )
        elif row[0] == "B":
            b.append( int( row[1] ) )

a = np.array( a )
b = np.array( b )

# variance of each group
a_var = a.var( ddof=1 )
b_var = b.var( ddof=1 )

# pooled variance
s = np.sqrt( ( a_var + b_var ) / 2 )

# t value
# caution : the size of a and b are same
t = ( a.mean() - b.mean() ) / ( s * np.sqrt( 2 / a.size ) )

# degrees of freedom
# caution : the size of a and b are same
df = a.size * 2 - 2

# p value
# caution : the size of a and b are same
p = 1 - stats.t.cdf( np.abs( t ), df=df )

print( "t:\t", t, "p:\t", p*2 )


t2, p2 = stats.ttest_ind( a, b )

print( "t2:\t", t2, "p2:\t", p2 )

# in this case, type B advertisement is attractful obviously
# so one side testing might be appried



