# this is practice program for exercise in Udemy course Bayesian A/B Testing
# just calculate probability of advertisements A and B
# the data is provided by advertisement_clicks.csv

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from scipy.stats         import beta

data = pd.read_csv( "advertisement_clicks.csv" )

A = data[ data['advertisement_id']=='A' ]
B = data[ data['advertisement_id']=='B' ]
A = A['action']
B = B['action']

total_clicks_A = A.sum()
total_clicks_B = B.sum()

total_chance_A = A.size
total_chance_B = B.size

total_through_A = total_chance_A - total_clicks_A
total_through_B = total_chance_B - total_clicks_B

x = np.linspace( 0, 1, 200 )
A_dist = beta.pdf( x, a=total_clicks_A+1, b=total_through_A+1 )
B_dist = beta.pdf( x, a=total_clicks_B+1, b=total_through_B+1 )

plt.plot( x, A_dist, label="CTR of A" )
plt.plot( x, B_dist, label="CTR_of B" )
plt.legend()
plt.show()

