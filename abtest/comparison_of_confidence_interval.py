import matplotlib.pyplot as plt
import numpy             as np
from scipy.stats import beta, norm

T = 501
true_ctr = 0.5 # true click through rate
a, b = 1, 1 # parameter for beta distribution (ones are uniform distribution)

plot_indices = [10, 20, 30, 50, 100, 200, 300, 500]
data = np.empty( T )

for i in range( T ):
    x = 1 if np.random.random() < true_ctr else 0
    data[i] = x

    a += x
    b += 1 - x

    if i in plot_indices:
        p = data[:i].mean()
        n = i + 1
        std = np.sqrt( p*(1-p)/n )

        x = np.linspace( 0, 1, 200 )
        g = norm.pdf( x, loc=p, scale=std )
        plt.plot( x, g, label="Gaussian Approximation" )

        posterior = beta.pdf( x, a=a, b=b )
        plt.plot( x, posterior, label="Beta Posterior" )

        plt.legend()
        plt.title( "N = %s" % n )
        plt.show()

