import sys
import time
import pickle
import math as m
import numpy as np
import numpy.random as npr
from mpi4py import MPI

import prog_bar
import epidemic
import sampler


def generateSyntheticData( n_mems, parms, time_data ):

    beta, gamma, init_prob_inf, rho = parms
    init_probs = [ 1. - init_prob_inf, init_prob_inf, 0. ]    
    t_start, delta_t, t_steps = time_data
    
    epi = epidemic.epidemic( n_mems, beta, gamma, init_probs, 1.)
    
    t_points = list(np.arange(t_start, delta_t*t_steps, delta_t))
    traj = []
    for t in t_points:
        epi.run2( t )
        traj.append( epi.observe_true() )
        y = [npr.binomial(obs[1],rho) for obs in traj]

    return (t_points, y)

def initSampler(n_mems, obs, parms):
    t_points, y = [list( obj ) for obj in obs]
    beta, gamma, init_prob_inf, rho = parms
    init_probs = [ 1. - init_prob_inf, init_prob_inf, 0. ]    

    chain = sampler.da_sampler( y, t_points, n_mems, beta, gamma, init_probs, rho)

    return chain

def expectedLikelihood( sampler, parms, aug_data_n_draws  ):   
    sampler.beta, sampler.gamma, init_prob_inf, sampler.rho = parms
    sampler.p = y[ 1. - init_prob_inf, init_prob_inf, 0. ]

    like_samps = np.ones( draws, dtype=np.float64 )
    draws = int( aug_data_n_draws )    
    for i in range(draws):
        try:
            chain.update( npr.choice( range(n_mems) ) )

        except AssertionError:
            like_samps = 'Internal Assertion Error'
            break
        
        like_samps[i] = chain.full_likelihood()

    if type( like_samps ) != str:
        return np.mean( like_samps )

    else:
        return like_samps


def main( rank, size, comm ):
    # set parms for synthetic data
    beta_true = .2
    gamma_true = 2.

    n_mems = int(1e2)
    init_prob_inf = 10. / n_mems
    rho = .4
   
    t_start = 0
    delta_t = 1e-3
    t_steps = 100

    # initialize empty arrays for data sync across nodes
    n_axis_ticks = size #1000
    n_grid_pts = n_axis_ticks**2

    obs = np.empty( [2, t_steps], dtype=np.float64 )
    Betas = Gammas = np.empty( n_grid_pts, dtype=np.float64 )
    beta = gamma = np.empty(size, dtype=np.float64)
    sampler = None

    # start the show
    if rank == 0:
        obs = np.array( generateSyntheticData( n_mems, [beta_true, gamma_true, init_prob_inf, rho], [t_start, delta_t, t_steps] ), dtype=np.float64 )

        betas = np.linspace(0, 10, n_axis_ticks, dtype=np.float64)
        gammas = np.linspace(0, 10, n_axis_ticks, dtype=np.float64)

        Betas, Gammas= np.meshgrid(betas, gammas)

        print 'call sampler...'
        sampler = initSampler( n_mems, obs, [ Betas[0,0], Gammas[0,0], init_prob_inf, rho ] )
        print '...complete'
        print 'init sampler...'
        sampler.initialize(stop=100)
        print '...complete'
        print ' begin data sync...'

    comm.Bcast( [obs, MPI.DOUBLE], root=0 )
    comm.Scatter(Betas, beta, root=0)
    comm.Scatter(Gammas, gamma, root=0)
    sampler = comm.bcast( sampler, root=0 )

    comm.Barrier()
    if rank == 0:
        print '...complete'

    else:
        parms_array = np.array([ [b, g, init_prob_inf, rho] for b,g in zip(beta,gamma) ])
        draws = 1000

        #foo = expectedLikelihood( n_mems, obs, parms_array[0], draws )
        
if __name__=='__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    main(rank, size, comm)          

    MPI.Finalize()

# some handy auxillary funcs that get used for post hoc analysis
#def time_autocorr(x,lag):
#    curr_mean = np.mean( x[lag:] )
#    lagg_mean = np.mean( x[:-lag] )
#    rand_var = zip(x[:-lag],x[lag:])
#    rand_var = [ (r[0]-lagg_mean)*(r[1]-curr_mean) for r in rand_var ]
#
#    return np.mean( rand_var )
#
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
