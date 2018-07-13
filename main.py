import sys
import time
import pickle
import math as m
import numpy as np
import numpy.random as npr
import jug

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

def expectedLikelihood( n_mems, obs, parms, aug_data_n_draws  ):
    prob_inf = 10. / n_mems
    init_probs = [ 1. - prob_inf, prob_inf, 0. ]
    
    t_points, y = obs                      
    beta, gamma, init_probs, rho = parms
    draws = int( aug_data_n_draws )

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


def main(beta_in, gamma_in):

    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    ind = tx + ty*bw

    if ind < exp_like_out.size:
        parms = [beta[ind], gamma[ind], init_probs, rho]
        exp_like_out[ind] = expectedLikelihood( n_mems, obs, parms, draws )
    

if __name__=='__main__':
    beta_true = .2
    gamma_true = 2.

    global n_mems
    global prob_inf
    global init_probs
    global rho
    global draws
    global obs

    n_mems = int(1e2)
    prob_inf = 10. / n_mems
    init_probs = [ 1. - prob_inf, prob_inf, 0. ]
    rho = .4
    draws = 1000
    
    t_start = 0
    delta_t = 1e-3
    t_steps = 100

    obs = generateSyntheticData( n_mems, [beta_true, gamma_true, init_probs, rho], [t_start, delta_t, t_steps] )

    n_axis_ticks = 1000
    betas = np.linspace(0,np.float64(10),n_axis_ticks)
    gammas = np.linspace(0,np.float64(10),n_axis_ticks)

    Betas, Gammas= np.meshgrid(betas, gammas)
    Betas = np.array([ b for b_list in Betas for b in b_list ])
    Gammas = np.array([ g for g_list in Gamma for g in g_list ])

    n_grid_points = Betas.size
    exp_like_out = np.ones(n_grid_points, dtype='float64')

    threads_per_block = 512
    blocks_per_grid = m.ceil( float(n_grid_points) / threads_per_block )

    main[blocks_per_grid, threads_per_block](Betas, Gammas, exp_like_out)

           

# some handy auxillary funcs that get used for post hoc analysis
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


#    if i % 20 == 0:
#        print i
#        with open('like_surface.pickle','wb') as handle:
#            pickle.dump( exp_like_surface, handle ) 
#            
#    else:
#        pass
#    
#    print( 'run time was ' + str(time_end-time_start) )
#    
#    time_start = time.time()
