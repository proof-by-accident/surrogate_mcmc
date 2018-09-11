import copy
import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as scla
from scipy import stats as st
from scipy.integrate import ode
import itertools as it
import sys
import matplotlib.pyplot as plt
import time

import line_profiler as lp

class Sampler(list):
    def __init__(self, prior, like, starts, chains, kwargs={} ):
        self.n_chains = int(chains)

        assert len(starts ) == self.n_chains
        for start in starts:
            self.append(
                ShrinkingBullseyeChain_NormProp( prior = prior,
                                                 true_like = like,
                                                 start = start,
                                                 **kwargs )
            )

    def burn(self, prop_var, N=1 ):
        for chain in self:
            chain.burn( prop_var, N )

    def update(self, var ):
        for chain in self:
            chain.update(var)

class ShrinkingBullseyeChain_NormProp:
    #@profile
    def __init__(self, prior, true_like, start, S_seed, fS_seed, train = None, size=int(1e9)):
        self.prior = prior
        self.true_like = true_like
        self.log_like = lambda x: np.log( true_like( x ) )

        self.shape = [size,np.shape(start)[0]]
        self.dim = np.shape(start)[0]
        self.samps = np.zeros(self.shape)


        if self.dim == 1:
            self.Ndef = int((self.dim+1)*(self.dim+2)/2)

            self.N = 2*self.Ndef

        else:
            self.Ndef = int((self.dim+1)*(self.dim+2)/2)
            self.N = int(np.ceil(np.sqrt(self.dim)*self.Ndef))
        
        self.samps[0,:] = start
        self.curr = self.samps[0]

        self.refine_tracker = []

        if self.dim == 1:
            self.S = np.reshape(S_seed, [np.shape(S_seed)[0],1] )
            self.fS = np.reshape(fS_seed, [np.shape(fS_seed)[0],1] )

        else:
            self.S = S_seed
            self.fS = fS_seed

        self.t = 0
        self.accept_freq = 0

        if train == None:
            train = len(S)

        else:
            pass
        
        self.seed_grow( train )

    #@profile
    def seed_grow( self, train_size ):
        orig_S = copy.copy( self.S )
        for s in orig_S:
            
            S_spacing = np.mean( [ npla.norm( s1 - s2 ) for s1,s2 in it.combinations_with_replacement( self.S, 2 ) ] )

            for junk_index in range( train_size):
                refine_center = s + np.random.multivariate_normal( np.zeros( self.dim ) , S_spacing * np.eye( self.dim ) )
                refine_range = 2 * S_spacing * np.random.uniform()

                self.refine( refine_center, refine_range )
            
                                   
    #@profile
    def rad_calc(self, theta, N ):
        radii = np.sort(npla.norm( self.S- theta, ord=2, axis=1 ))

        try:
            Rdef = radii[N+1]

        except IndexError:
            print 'Sample set S is too small, consider increasing'
            return np.max( radii )
                    
        return Rdef

    #@profile
    def weight_func(self, theta, b, Rdef, R):
        r = npla.norm( theta - b )

        if r <= Rdef:
            return 1.

        elif ( Rdef <= r ) and ( r <= R ):            
            w = (1. - ( (r-Rdef)/(R-Rdef) )**3.)**3.
            
            return w

        else:
            return 0.
            

    # function calculates regression coefficients for LQR model of likelihood surface
    #@profile
    def regress(self, theta, B, fB, Rdef, R, i = None):        
        N = B.shape[0]

        if len(B.shape) == 1:
            B = np.reshape(B, (N, self.dim) )


        phi = np.ones([ N, (self.dim + 2)*(self.dim + 1)/2 ])

        try:
            assert phi.shape[0] > phi.shape[1]

        except AssertionError:
            print 'insufficient sample points, inference may be problematic...'

        scale = np.sqrt(np.var(theta))
        theta_hat = (B - theta)/scale
        
        w = np.array([ self.weight_func( np.zeros(self.dim) , b, Rdef/scale , R/scale )  for b in theta_hat ])
        
        phi[:,1:(self.dim+1)] = theta_hat

        theta_hat_squared = np.array([ [a*b for a,b in it.combinations_with_replacement(row,2)] for row in theta_hat ])
        
        phi[:,(self.dim+1):] = theta_hat_squared
       
        q,r = npla.qr( np.dot( phi.T, np.dot(np.diag(w),phi) ), mode='complete')        

        Z = np.dot( npla.inv(r), q.T )            
        Z = np.dot( Z, np.dot( phi.T, w * fB) )

        return Z, q, r, w

    #@profile
    def regress_predict(self, theta, Z):        
        return np.exp( Z[0] )

    #@profile
    def post_approx( self, theta ): 
        Rdef = self.rad_calc( theta, self.Ndef )
        R  = self.rad_calc( theta, self.N )
        
        B = self.S[ npla.norm(self.S - theta, axis=1 ) < R ]
        fB = np.array(self.fS)[ npla.norm(self.S - theta, axis=1 ) < R ]
        
        Z, q, r, w = self.regress( theta, B, fB, Rdef, R)               
            
        post = self.prior( theta ) * self.regress_predict( theta, Z )

        return post, Z, q, r, w, R, B, fB


    #Functions to cross validate the regression at the candidate and current points
    #@profile
    def cross_val(self, theta, post_theta_prime, a , B, fB, q, r, W, cand_flag = 0, eps = 1e-2):
        N = B.shape[0]

        a_list = np.zeros(N)

        scale = np.sqrt(np.var(theta))
            
        for i in range(0,N):            
            W_up = np.delete( W, i )
            B_up = np.delete( B, i, axis=0 )
            fB_up = np.delete( fB, i )

            phi_up = np.ones([ N-1, (self.dim + 2)*(self.dim + 1)/2 ])

            theta_hat = (B_up - theta)/scale
            
            phi_up[:,1:(self.dim+1)] = theta_hat
            
            theta_hat_squared = np.array([ [x*y for x,y in it.combinations_with_replacement(row,2)] for row in theta_hat ])
        
            phi_up[:,(self.dim+1):] = theta_hat_squared

            phi_down = np.concatenate([ [1], (B[i]-theta)/scale, [x*y for x,y in it.combinations_with_replacement( (B[i]-theta)/scale, 2 ) ] ])

            q_up, r_up = scla.qr_update( q, r, -phi_down, phi_down )
            
            Z_up = np.dot( npla.inv(r_up), q_up.T )
            Z_up = np.dot( Z_up, np.dot( phi_up.T, W_up * fB_up ) )

            post_up = self.prior( theta ) * self.regress_predict( theta, Z_up )

            try:
                assert post_up >= 0.

            except AssertionError:
                post_up = abs( post_up )

            if cand_flag:
                a_list[i] = self.a_calc( post_up, post_theta_prime )

            else:
                a_list[i] = self.a_calc( post_theta_prime, post_up )

        err_list = np.abs(a-a_list) + np.abs(min(1.,1./a)-np.array([min(1.,1./a_up) for a_up in a_list]))
        err = np.max(err_list)
        if err >= eps:
            return 1
        else:
            return 0

    #@profile
    def propose(self, theta, var):
        if self.dim == 1:
            return np.random.normal( theta, var, 1)[0]

        else:
            return np.random.multivariate_normal( theta, var, 1 )[0]

    #Functions to refine parameter samples
    #@profile
    def refine(self,theta,R):
        # require that the refinement is within .9*R, rather than just R, bc otherwise the sol tends to be slightly outside R due to numerical error 
        cons = [{'type' : 'ineq', 'fun': lambda x: .9*R - npla.norm(x - theta)},
                {'type' : 'ineq', 'fun': lambda x: self.prior(x)*self.true_like(x) } ]            
        
        penalty = lambda x: -1*np.min( npla.norm( x - self.S, axis=1 ) )
        
        sol = sp.optimize.minimize( penalty, theta, constraints=cons, tol=1e-1 ) #, options = {'maxiter' : 10000})

        self.refine_tracker.append( self.t )

        update = sol['x']
                    
        if not any( np.isnan(update) ) and npla.norm( theta - update ) <= R  :
            self.S = np.vstack([ self.S, update])
            self.fS = np.concatenate([ self.fS, [self.log_like(update)] ])


        else:
            #print 'Suitable refinement not found, adding random point'
            
            update = np.random.multivariate_normal(theta, (R/3)*np.eye(self.dim ))

            while self.prior( update ) < 0:
                update = np.random.multivariate_normal(theta, (R/3)*np.eye(self.dim ))

            self.S = np.vstack([ self.S, update])
            self.fS = np.concatenate([ self.fS, [self.log_like(update)] ])
            

    #@profile        
    def a_calc( self, p_cand, p_curr ):
        if p_curr == 0.0:
            return 1.

        elif np.isnan( p_cand ):
            return 0.

        else:            
            return min( 1. , p_cand/p_curr )


    # function that runs whenever a refinement is triggered (either by cross val or random refinement)
    #@profile
    def refine_routine(self):
        self.refine( self.cand, self.cand_R )
        self.cand_post, self.cand_Z, self.cand_q, self.cand_r, self.cand_w, self.cand_R, self.cand_B, self.cand_fB = self.post_approx( self.cand )
            
        if self.t==0:
            self.curr_post, self.curr_Z, self.curr_q, self.curr_r, self.curr_w, self.curr_R, self.curr_B, self.curr_fB = self.post_approx( self.curr )
                
        else:
            pass
            
        self.a = self.a_calc( self.cand_post, self.curr_post )
        

        
    #Update routine
    #@profile
    def update(self, var ):
        eps = 0.1*(self.t+1)**(-0.1)
        rand_refine = 0.01*(self.t+1)**(-0.2)

        self.cand = self.propose( self.curr, var )


        self.cand_post, self.cand_Z, self.cand_q, self.cand_r, self.cand_w, self.cand_R, self.cand_B, self.cand_fB = self.post_approx( self.cand )
        
        if self.t==0:
            self.curr_post, self.curr_Z, self.curr_q, self.curr_r, self.curr_w, self.curr_R, self.curr_B, self.curr_fB = self.post_approx( self.curr )

        else:
            pass

        self.a = self.a_calc( self.cand_post, self.curr_post )


        if np.random.binomial(1, rand_refine):
            self.refine_routine()

        else:
            pass



        while self.cross_val( self.cand, self.curr_post, self.a, self.cand_B, self.cand_fB, self.cand_q, self.cand_r, self.cand_w, cand_flag = 1, eps=eps ):            
            self.refine_routine()

        accept_flag = np.random.binomial( 1, self.a )

        if self.t > self.samps.shape[0]:
            self.samps = np.vstack([ self.samps, np.zeros( self.dim ) ])
            
        else:
            pass
        
        if accept_flag:
            self.samps[ self.t ] = self.cand
            self.curr = self.cand
            
            self.curr_post, self.curr_Z, self.curr_q, self.curr_r, self.curr_w, self.curr_R, self.curr_B, self.curr_fB = ( self.cand_post,
                                                                                                                           self.cand_Z,
                                                                                                                           self.cand_q,
                                                                                                                           self.cand_r,
                                                                                                                           self.cand_w,
                                                                                                                           self.cand_R,
                                                                                                                           self.cand_B,
                                                                                                                           self.cand_fB )
            
            self.accept_freq += 1
            
        else:
            self.samps[ self.t ] = self.curr
            
        self.t += 1

    def burn(self,var,N=1):
        for i in range(0,N):
            self.update( var )
            self.t = 0
            self.accept_freq = 0
