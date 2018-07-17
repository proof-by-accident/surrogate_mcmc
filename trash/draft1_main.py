import sys
import pickle
import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt

import epidemic
import sampler
import prog_bar

import time

time_start = time.time()
# generate synthetic data using the `epidemic` object defined in epidemic.py
n_mems = int(1e2)

prob_inf = 10. / n_mems
init_probs = [ 1. - prob_inf, prob_inf, 0. ]

beta_true = beta = .2
gamma_true = gamma = 2.
rho = .4

print('Generating data...')
epi = epidemic.epidemic( n_mems, beta, gamma, init_probs, 1.)

t_start = 0
delta_t = 1e-3

T = 100

t_points = list(np.arange(t_start, delta_t*T, delta_t))
traj = []
for t in t_points:
    epi.run2( t )
    traj.append( epi.observe_true() )
y = [npr.binomial(obs[1],rho) for obs in traj]

test = sampler.da_sampler( y, t_points, n_mems, beta, gamma, init_probs, rho)
print('Initializing...')
test.initialize()

draws = int(1e3)
like_samps = []

print('Sampling...')
bar = prog_bar.percent_bar( draws )
bar.set()
for subj in npr.choice(range(n_mems), draws, replace=True):
    test.update(subj)
    
    like_samps.append( test.full_likelihood() )

    bar.step()

like_dict = {(beta,gamma):like_samps}

#handle = open('like_dict.pickle','wb')
with open('like_dict.pickle','wb') as handle:
    pickle.dump( like_dict, handle )

with open('sampler.pickle','wb') as handle:
    pickle.dump( test, handle )

def time_autocorr(x,lag):
    curr_mean = np.mean( x[lag:] )
    lagg_mean = np.mean( x[:-lag] )
    rand_var = zip(x[:-lag],x[lag:])
    rand_var = [ (r[0]-lagg_mean)*(r[1]-curr_mean) for r in rand_var ]

    return np.mean( rand_var )
time_end = time.time()
print( 'run time was ' + str(time_end-time_start) )
