import sys
import time
import pickle
import numpy as np
import numpy.random as npr
from numba import vectorize
from numba import float64

import prog_bar
import epidemic
import sampler


def generateSyntheticData( n_mems, parms, time_data ):

    beta, gamma, init_probs, rho = parms
    t_start, delta_t, t_steps = time_data
    
    epi = epidemic.epidemic( n_mems, beta, gamma, init_probs, 1.)
    
    t_points = list(np.arange(t_start, delta_t*t_steps, delta_t))
    traj = []
    for t in t_points:
        epi.run2( t )
        traj.append( epi.observe_true() )
        y = [npr.binomial(obs[1],rho) for obs in traj]

    return (t_points, y)

def expectedLikelihood( beta, gamma  ):
#    prob_inf = 10. / n_mems
#    init_probs = [ 1. - prob_inf, prob_inf, 0. ]
#    
#    t_points, y = obs                      
#    beta, gamma, init_probs, rho = parms
#    draws = int( aug_data_n_draws )
#
    chain = sampler.da_sampler( y, t_points, n_mems, beta, gamma, init_probs, rho)
    print('init...')
    chain.initialize()
    print('...completed')

    like_samps = np.ones( draws, dtype=np.float64 )
    
    #bar = prog_bar.percent_bar( draws )
    #bar.set()
    for i in range(draws):
        try:
            chain.update( npr.choice( range(n_mems) ) )

        except AssertionError:
            print 'Internal Assertion Error'
            break
        
        like_samps[i] = chain.full_likelihood()
        #bar.step()

    return np.mean( like_samps )


def main():
    time_start = time.time()

    global n_mems
    n_mems = int(1e2)

    global prob_inf
    global init_probs
    prob_inf = 10. / n_mems
    init_probs = [ 1. - prob_inf, prob_inf, 0. ]
    
    beta_true = beta = .2
    gamma_true = gamma = 2.

    global rho
    rho = .4
    
    t_start = 0
    delta_t = 1e-3
    t_steps = 100

    global t_points
    global y
    t_points, y = generateSyntheticData( n_mems, [beta, gamma, init_probs, rho], [t_start, delta_t, t_steps] )

    global draws
    draws = 1000

    global exp_like

    betas = np.linspace(0,10,1000)
    exp_like_surface = np.array([[0,0,0]]*len(betas))

    for i in range(len(betas)):
        exp_like = expectedLikelihood( betas[i], gamma )
        exp_like_surface[i] = [betas[i],gamma,exp_like]
        sys.stdout.write(str(i))
        sys.stdout.flush()

        if i % 20 == 0:
            print i
            with open('like_surface.pickle','wb') as handle:
                pickle.dump( exp_like_surface, handle ) 

        else:
            pass
    
    time_end = time.time()
    print( 'run time was ' + str(time_end-time_start) )


if __name__=='__main__':
    main()
    

# some handy auxillary funcs that I only use sometimes
def time_autocorr(x,lag):
    curr_mean = np.mean( x[lag:] )
    lagg_mean = np.mean( x[:-lag] )
    rand_var = zip(x[:-lag],x[lag:])
    rand_var = [ (r[0]-lagg_mean)*(r[1]-curr_mean) for r in rand_var ]

    return np.mean( rand_var )

#!!!!!!!!!!!!!!!!!CRUFT!!!!!!!!!!!!!!!!!#
#    #handle = open('like_dict.pickle','wb')
#    with open('like_dict.pickle','wb') as handle:
#        pickle.dump( like_dict, handle )
#        
#        with open('sampler.pickle','wb') as handle:
#            pickle.dump( test, handle )
