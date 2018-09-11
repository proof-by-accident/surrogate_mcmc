import itertools as it
import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy as spla

from samplers import shrinking_bullseye as sb

np.set_printoptions(linewidth=200)

import line_profiler as lp

chains = 4
starts = np.random.multivariate_normal(np.zeros(2) , 10.*np.eye( 2 ), chains )

true_like = lambda x: sp.stats.multivariate_normal.pdf(x, 1.4*np.ones(2), 1.*np.eye( 2 ) )
prior = lambda x: sp.stats.multivariate_normal.pdf(x, np.zeros(2) , 10.*np.eye( 2 ) )

S = np.random.multivariate_normal(np.zeros(2) , 10.*np.eye( 2 ), 20 )
fS = np.array([ true_like( s ) for s in S ])

kwargs = { 'S_seed' : S,
           'fS_seed' : fS,
           'train' : 10 }


print 'init sampler...'
foo = sb.Sampler( prior, true_like, starts, chains, kwargs )
print 'done'

print 'burn-in...'
burn_var = 5*np.eye(2)
foo.burn( burn_var, N=20 )
print 'done'

print 'sampling...'
prop_var = .1*np.eye(2)
for i in range( int( 1e3 ) ):
    print i
    foo.update( prop_var )
print 'done'


