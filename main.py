import sys
import pickle
import numpy as np
import numpy.random as npr

import epidemic
import sampler

import prog_bar
import time


n_mems = int(1e2)

prob_inf = 10. / n_mems
init_probs = [ 1. - prob_inf, prob_inf, 0. ]

beta_true = beta = .2
gamma_true = gamma = 2.
rho = .4

t_start = 0
delta_t = 1e-3

t_steps = 100

def generateSyntheticData( n_mems, parms, time_data ):

    beta, gamma, rho, init_probs = parms
    t_start, delta_t, t_steps = time_data
    
    epi = epidemic.epidemic( n_mems, beta, gamma, init_probs, 1.)
    
    t_points = list(np.arange(t_start, delta_t*T, delta_t))
    traj = []
    for t in t_points:
        epi.run2( t )
        traj.append( epi.observe_true() )
        y = [npr.binomial(obs[1],rho) for obs in traj]

    return (t_points, y)

def expectedLikelihood( obs, n_mems, parms, aug_data_n_draws ):
    t_points, y = obs                      
    beta, gamma, init_probs, rho = parms
    draws = int( aug_data_n_draws )

    chain = sampler.da_sampler( y, t_points, n_mems, beta, gamma, init_probs, rho)
    chain.initialize()

    like_samps = 

for subj in npr.choice(range(0,10), draws, replace=True):
    test.hmm_step( subj )
    test.discrete_time_step( subj )    
    test.event_time_step( subj )

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
