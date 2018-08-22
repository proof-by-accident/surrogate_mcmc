import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.linalg as spla

class epidemic:
    def __init__(self, num_members, inf_rate, recovery_rate, init_proportions, obs_prop):
        self.n = num_members
        self.beta = inf_rate
        self.gamma = recovery_rate

        assert sum(init_proportions) == 1.
        assert type(init_proportions) == list
        assert len(init_proportions) == 3
        self.p = init_proportions
        self.rho = obs_prop

        self.t = 0
        self.pop = ['s']*self.n

        ind = 0
        while sum( [a == 'i' for a in self.pop] ) < self.n*self.p[1]:
            self.pop[ind] = 'i'
            ind += 1

        while sum( [a == 'r' for a in self.pop] ) < self.n*self.p[2]:
            self.pop[ind] = 'r'
            ind += 1

      
    def run2(self, end_t, start_t = None):
        if start_t == None:
            curr_t = self.t

        else:
            pass

        if curr_t > end_t:
            return(None)

        else:
            assert curr_t <= end_t

        while curr_t < end_t:           
            self.S = np.sum([a == 's' for a in self.pop])
            self.I = np.sum([a == 'i' for a in self.pop])
            self.R = np.sum([a == 'r' for a in self.pop])

            rate = self.beta*self.S*self.I + self.gamma*self.I

            self.rate = rate
            
            if rate == 0:
                p_infec = 0
                infec_flag = -1
                next_t = end_t - curr_t
                

            else:
                p_infec = self.beta*self.S*self.I/rate
                next_t = np.random.exponential( 1/rate)
                infec_flag = np.random.binomial(1, p_infec)

            if infec_flag==1:
                self.pop[ self.pop.index('s') ] = 'i'

            elif infec_flag==0:
                self.pop[ self.pop.index('i') ] = 'r'

            else:
                pass
            
            
            curr_t = curr_t + next_t
            self.t = curr_t


            
    def run(self, end_t, start_t = None):
        
        if start_t == None:
            curr_t = self.t

        else:
            curr_t = start_t

        assert curr_t != None

        if curr_t > end_t:
            return(None)

        else:        
            assert curr_t <= end_t
        
        while curr_t <= end_t:
            self.times = []
            
            self.S = np.sum([a == 's' for a in self.pop])
            self.I = np.sum([a == 'i' for a in self.pop])
            self.R = np.sum([a == 'r' for a in self.pop])
            
            for agent in self.pop:
                if agent == 's':
                    rate = self.beta * self.I
                    
                elif agent == 'i':
                    rate = self.gamma
                    
                else:
                    rate = np.inf
                    
                self.times.append( np.random.exponential(1/rate) )
                    
                next_t = np.argmin(self.times)
                
                assert self.pop[next_t] != 'r'
                
                if self.pop[next_t] == 's':
                    self.pop[next_t] = 'i'
                    
                else:
                    self.pop[next_t] == 'r'
                    
                    curr_t = curr_t + self.times[next_t]
                    
                    self.t = curr_t
                        
    def observe_true(self):
        self.S = sum([a == 's' for a in self.pop])
        self.I = sum([a == 'i' for a in self.pop])
        self.R = sum([a == 'r' for a in self.pop])

        return [self.S, self.I, self.R]
    
    def observe_error(self, error = None ):
        if error == None:
            error = [self.rho]*3

        else:
            pass
        
        assert np.max( error ) <= 1.0
        assert len( error ) == 3
        assert type(error) == list
        
        self.S = np.sum([a == 's' for a in self.pop])
        self.I = np.sum([a == 'i' for a in self.pop])
        self.R = np.sum([a == 'r' for a in self.pop])

        self.obs_S = np.random.binomial( self.S, error[0] )
        self.obs_I = np.random.binomial( self.I, error[1] )
        self.obs_R = np.random.binomial( self.R, error[2] )

        return [self.obs_S, self.obs_I, self.obs_R]

