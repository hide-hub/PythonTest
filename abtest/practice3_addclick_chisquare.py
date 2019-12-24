# this is practice program for chi square testing
# the data file is advertisement.clicks.csv again

import csv
import numpy        as np
import pandas       as pd
import scipy.stats  as stats
from scipy.stats import chi2, chi2_contingency

data = pd.read_csv( "advertisement_clicks.csv" )

a = data[ data['advertisement_id']=='A' ]
b = data[ data['advertisement_id']=='B' ]
a = a['action']
b = b['action']

print( 'a.mean() : ', a.mean() )
print( 'b.mean() : ', b.mean() )

T = np.zeros( (2,2) )
T[0] = [ sum( a ), len(a)-sum(a) ]
T[1] = [ sum( b ), len(b)-sum(b) ]

# chi square testing
det = np.linalg.det( T )
c2 = float( det ) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
p_value = 1 - chi2.cdf( x=c2, df=1 )





