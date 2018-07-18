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
    sampler.p = [ 1. - init_prob_inf, init_prob_inf, 0. ]

    draws = int( aug_data_n_draws )    
    like_samps = np.ones( draws, dtype=np.float64 )
    for i in range(draws):
        try:
            sampler.update( npr.choice( range( sampler.n ) ) )

        except AssertionError:
            with open('./saves/rank'+str(rank)+'.pl','wb') as file:
                pickle.dump( sampler, file )

            like_samps = 'Internal Assertion Error on draw ' + str(i)
            return like_samps
        
        like_samps[i] = sampler.full_likelihood()

    return np.mean( like_samps )


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
    n_axis_ticks = size
    n_grid_pts = n_axis_ticks**2

    obs = np.empty( [2, t_steps], dtype=np.float64 )
    Betas = Gammas = np.empty( n_grid_pts, dtype=np.float64 )
    beta = gamma = np.empty(size, dtype=np.float64)
    sampler = None

    exp_like_row = np.empty(size, dtype=np.float64)
    exp_like_out = np.empty([size,size], dtype=np.float64)

    # start the show
    if rank == 0:
        obs = np.array( generateSyntheticData( n_mems, [beta_true, gamma_true, init_prob_inf, rho], [t_start, delta_t, t_steps] ), dtype=np.float64 )

        betas = np.linspace(.001, 10, n_axis_ticks, dtype=np.float64)
        gammas = np.linspace(.001, 10, n_axis_ticks, dtype=np.float64)

        Betas, Gammas= np.meshgrid(betas, gammas)

        print 'call sampler...'
        sampler = initSampler( n_mems, obs, [ beta_true, gamma_true, init_prob_inf, rho ] )
        print '...complete'
        print 'init sampler...'
        sampler.initialize()
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
        print 'node ',rank,' beginning evals...'
        for i in range(len(parms_array)):
            ex_lk = expectedLikelihood( sampler, parms_array[i], draws )
            exp_like_row[i] = ex_lk           
            
            if i%20 == 0:
                print '...node ',rank,' checking in on run ',i,'...'
                with open('./saves/rank'+str(rank)+'_exlk.pl','wb') as f:
                    save = [parms_array, exp_like_row]
                    pickle.dump( save, f)

        print '...node ', rank, ' complete'

    comm.Gather( exp_like_row, exp_like_out, root=0 )
    if rank == 0:
        save = [Betas,Gammas,exp_like_out]
        with open('./exp_like.pl','wb') as f:
            pickle.dump( save, f )

    else:
        pass

        
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
