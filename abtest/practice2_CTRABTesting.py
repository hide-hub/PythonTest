# this is practice program for Udemy course, A/B Testing
# the click through rate for advertisement A and B can be
# compaired by Chi-square distribution

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, chi2_contingency

class DataGenerator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def next(self):
        click1 = 1 if ( np.random.random() < self.p1 ) else 0
        click2 = 1 if ( np.random.random() < self.p2 ) else 0
        return click1, click2

def get_p_value(T): # T is 2x2 matrix
    det = T[0,0]*T[1,1] - T[1,0]*T[0,1]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
    p = 1 - chi2.cdf( x=c2, df=1 )
    return p

def run_experiment(p1, p2, N):
    data = DataGenerator(p1, p2)
    p_values = np.empty(N)
    T = np.zeros( (2,2) ).astype( float )
    for i in range(N):
        c1, c2 = data.next()
        T[0,c1] += 1
        T[1,c2] += 1
        if i < 10:  # because there is some 0 cells in the table T during initial steps
            p_values[i] = None
        else:
            p_values[i] = get_p_value(T)
    #print( T )
    plt.plot( p_values )
    plt.plot( np.ones(N)*0.05 )
    plt.show()

# run following code several tiimes
# so that you can see the sample size 20,000 is not enough for
# detecting significant difference between two groups only different 1% probability
# I checked the sample size 50,000 is goot for them (experimentaly)
run_experiment( 0.1, 0.11, 20000 )

# rule of thumb for sample size is
# 16 * sigma^2 / delta^2
# where sigma^2 is variance for sample data, and delta is difference you want to detect

# in this case
N = 500
data_sample = np.zeros( (2,N) )
data = DataGenerator( 0.1, 0.11 ) # the difference is 0.01 
for i in range( N ):
    for j in range( 500 ):
        c1, c2 = data.next()
        data_sample[0,i] += c1
        data_sample[1,i] += c2

ssize = 16 * np.var( data_sample[0] ) / ( 0.01**2 )
print( "sample size for this case is ", ssize )

# the answer is over 7 million
# I'm not sure if this sample size is required
# (anway, this is just rule of thumb)
