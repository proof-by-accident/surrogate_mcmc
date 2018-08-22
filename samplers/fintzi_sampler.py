#Embedded file name: /home/peter/Desktop/Surrogate_MCMC/Code/Back_to_Basics/sampler.py
import pickle
from random import shuffle
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as sps
import scipy.linalg as spla
import math as m
import bisect

class da_sampler:

    def __init__(self, obs_infec, obs_times, n_mems, inf_rate, recovery_rate, init_probs, obs_prob):
        self.n = n_mems
        self.y = obs_infec
        self.obs_times = obs_times
        assert len(self.y) == len(self.obs_times)
        self.beta = inf_rate
        self.gamma = recovery_rate
        self.p = init_probs
        self.rho = obs_prob

        self.subj = None

    def state_traj_calc(self, exclude = None):
        trans_times_flat = {}
        for i in range(len(self.trans_times)):
            if i == exclude:
                pass
            else:
                indiv = self.trans_times[i]
                if len(indiv) > 0 and indiv[0] != self.obs_times[0]:
                    trans_times_flat[indiv[0]] = 'I'

                else:
                    pass

                if len(indiv) == 2 and indiv[1] != self.obs_times[0]:
                    trans_times_flat[indiv[1]] = 'R'

                else:
                    pass

        S, I, R = self.state_count(self.obs_times[0], exclude)
        self.state_traj = {self.obs_times[0]: (S, I, R)}
        for key in sorted(trans_times_flat.keys()):
            if trans_times_flat[key] == 'I':
                S -= 1
                I += 1
            else:
                I -= 1
                R += 1
            self.state_traj[key] = (S, I, R)

    def state_count(self, time, exclude = None, refresh = True, trans_times = None):
        if trans_times == None:
            trans_times = self.trans_times

        else:
            pass
        
        
        if refresh:
            S, I, R = (0, 0, 0)
            for i in range(0, len(trans_times)):
                if i == exclude:
                    pass
                else:
                    indiv = trans_times[i]
                    if len(indiv) == 0:
                        S += 1
                    elif len(indiv) == 1:
                        if indiv[0] <= time:
                            I += 1
                        else:
                            S += 1
                    elif indiv[0] <= time and indiv[1] <= time:
                        R += 1
                    elif indiv[0] <= time and indiv[1] >= time:
                        I += 1
                    else:
                        S += 1

        else:
            key_match = max([ t for t in self.state_traj.keys() if t <= time ])
            S, I, R = self.state_traj[key_match]
        return (S, I, R)

    def validAugDataChecker(self, exclude=None):
        valid_tpts_mask = []
        for tpoint in self.obs_times:
            curr_infected = self.state_count(tpoint, exclude=None)[1]
            valid_tpts_mask.append(curr_infected >= self.y[self.obs_times.index(tpoint)])

        return all(valid_tpts_mask)

    def initialize(self, stop=None):
        self.trans_times = [[]] * self.n
        stop_q = 0
        while not self.validAugDataChecker():
            self.trans_times = [[]] * self.n
            stop_q += 1
            t_curr = self.obs_times[0]
            init_counts = npr.multinomial(self.n, self.p)
            S = [[]] * init_counts[0]
            I = [[t_curr]] * init_counts[1]
            R = [[t_curr, t_curr]] * init_counts[2]

            while t_curr <= max(self.obs_times):
                rate = self.beta * len(S) * len(I) + self.gamma * len(I)
                if rate == 0:
                    infec_flag = -1
                    t_next = 0
                else:
                    t_next = np.random.exponential(1 / rate)
                    p_infec = self.beta * len(S) * len(I) / rate
                    infec_flag = np.random.binomial(1, p_infec)
                t_curr += t_next
                if infec_flag == 1:
                    try:
                        agent = S.pop()
                        I.append(agent + [t_curr])
                        shuffle(I)
                    except IndexError:
                        pass

                elif infec_flag == 0:
                    try:
                        agent = I.pop()
                        R.append(agent + [t_curr])
                    except IndexError:
                        pass

                else:
                    break

            if stop_q == stop:
                with open('./init_breakpt_save.pickle','wb') as file:
                    pickle.dump( self, file )

                return(None)

            else:
                pass

            self.trans_times = S + I + R
            shuffle(self.trans_times)

    def rate_matrix(self, time, exclude = None, refresh = True, trans_times = None):
        self.I = self.state_count(time, exclude=exclude, refresh=refresh, trans_times = trans_times)[1]
        return np.array([[-self.beta * self.I, self.beta * self.I, 0], [0, -self.gamma, self.gamma], [0, 0, 0]])

    def binom_pmf(self, n, rho, y):           
        if n >= y:
            return m.factorial(n) / (m.factorial(n - y) * m.factorial(y)) * pow(rho, y) * pow(1 - rho, n - y)

        else:
            return 0.

    def hmm_step(self, subj):
        n = subj
        self.state_traj_calc(exclude=n)        
        I = self.state_count(self.obs_times[0], exclude=n, refresh=False)[1]
        F = [float(self.binom_pmf(I, self.rho, self.y[0])),
             float(self.binom_pmf(I + 1, self.rho, self.y[0])),
             float(self.binom_pmf(I, self.rho, self.y[0]))]
        f = [ p*phi for p,phi in zip(self.p,F) ]
        f_vecs = [f]
        P_mats = []
        F_vecs = []
        
        #forward step

        for t_left, t_right in zip(self.obs_times[:-1], self.obs_times[1:]):
            f = f/np.sum(f)
            L = np.array(self.rate_matrix(t_left, exclude=n, refresh=False))
            P = spla.expm((t_right - t_left) * L)
            I = self.state_count(t_right, exclude=n, refresh=False)[1]
            F = [float(self.binom_pmf(I, self.rho, self.y[self.obs_times.index(t_right)])),
                 float(self.binom_pmf(I + 1, self.rho, self.y[self.obs_times.index(t_right)])),
                 float(self.binom_pmf(I, self.rho, self.y[self.obs_times.index(t_right)])) ]

            P = (f * P.T).T * F
            f = np.sum(P, 0)
            
            assert sum(f) != 0

            f_vecs.append((f / sum(f), np.sum(f)))
            P_mats.append(P / np.sum(P))

        self.f_vecs = f_vecs
        self.hmm_skeleton = []
        self.b_vecs = []
        #backward step
        for t in self.obs_times[::-1]:
            if t == self.obs_times[-1]:
                b = f_vecs.pop()[0]
                P = P_mats.pop
                state = npr.choice(['S', 'I', 'R'], p=b)
                self.hmm_skeleton.insert(0, state)
                self.b_vecs.insert(0, b)
            else:
                state_ind = ['S', 'I', 'R'].index(state)
                P = P_mats.pop()
                b = P[:, state_ind]
                assert sum(b) > 0
                b = b / sum(b)
                state = npr.choice(['S', 'I', 'R'], p=b)
                self.hmm_skeleton.insert(0, state)
                self.b_vecs.insert(0, b)

        for i in range(1, len(self.hmm_skeleton)):
            curr = ['S', 'I', 'R'].index(self.hmm_skeleton[i])
            prev = ['S', 'I', 'R'].index(self.hmm_skeleton[i - 1])
            assert curr >= prev

    def discrete_time_step(self, subj):
        n = subj
        self.dt_skeleton = []
        for i in range(1, len(self.obs_times)):
            left_endpt, right_endpt = self.obs_times[i - 1], self.obs_times[i]
            left_state, right_state = self.hmm_skeleton[i - 1], self.hmm_skeleton[i]
            right_state_ind = ['S', 'I', 'R'].index(right_state)
            self.dt_skeleton.append([left_state, left_endpt])
            interval_change_times = [ t for agent in self.trans_times for t in agent if left_endpt <= t and t <= right_endpt ]
            interval_change_times.append(right_endpt)
            interval_change_times.insert(0, left_endpt)
            interval_change_times = np.sort( interval_change_times )
            self.ict = interval_change_times
            
            Q = [ (t_right - t_left) * self.rate_matrix(t_left, n, refresh=False) for t_left, t_right in zip(interval_change_times[:-1], interval_change_times[1:]) ]
            P = [ spla.expm(q) for q in Q ]
            P_fwd = []
            p = np.eye(3)
            for m in P[::-1]:
                p = np.matmul(p, m)
                P_fwd.insert(0, p)

            state = left_state
            self.debug_P = P
            self.debug_P_fwd = P_fwd 
            for j in range(1, len(interval_change_times) - 1):
                t_left, t_right = interval_change_times[j - 1], interval_change_times[j]
                state_ind = ['S', 'I', 'R'].index(state)
                p1 = P[j - 1][state_ind, :]
                p2 = P_fwd[j][:, right_state_ind]
                p3 = P_fwd[j - 1][state_ind, right_state_ind]
                pi = np.array(p1*p2/p3)/np.sum(p1*p2/p3)
                
                state = ['S', 'I', 'R'][list(npr.multinomial(1, pi)).index(1)]

                self.dt_skeleton.append([state, t_right])

        for i in range(1, len(self.dt_skeleton)):
            curr = ['S', 'I', 'R'].index(self.dt_skeleton[i][0])
            prev = ['S', 'I', 'R'].index(self.dt_skeleton[i - 1][0])
            assert curr >= prev

    def event_time_step(self, subj):
        n = subj
        bracket_times = []
        for state_left, state_right in zip(self.dt_skeleton[:-1], self.dt_skeleton[1:]):
            if state_left[0] != state_right[0]:
                bracket_times.append(state_left)
                bracket_times.append(state_right)

        assert len(bracket_times) in (0, 2, 4)
        self.bracket_times = bracket_times
        new_trans_times = []
        if len(bracket_times) == 4:
            for unused_index in [0, 1]:
                state_left = bracket_times.pop(0)
                state_ind = ['S', 'I', 'R'].index(state_left[0])
                state_right = bracket_times.pop(0)
                L = self.rate_matrix(state_left[1], n)[state_ind, state_ind]
                T = state_right[1] - state_left[1]
                u = npr.uniform()
                new_trans_times.append(state_left[1] - np.log(1 - u * (1 - np.exp(-T * L))) / L)

        elif len(bracket_times) == 2:
            state_left = bracket_times.pop(0)
            state_ind = ['S', 'I', 'R'].index(state_left[0])
            state_right = bracket_times.pop(0)
            L = self.rate_matrix(state_left[1], n)[state_ind, state_ind]
            T = state_right[1] - state_left[1]
            u = npr.uniform()
            new_trans_times.append(state_left[1] - np.log(1 - u * (1 - np.exp(-T * L))) / L)
            if state_left[0] == 'S' and state_right[0] == 'R':
                state_left = ('I', new_trans_times[0])
                state_ind = ['S', 'I', 'R'].index(state_left[0])
                L = self.rate_matrix(state_left[1], n)[state_ind, state_ind]
                T = state_right[1] - state_left[1]
                u = npr.uniform()
                new_trans_times.append(state_left[1] - np.log(1 - u * (1 - np.exp(-T * L))) / L)

            elif state_left[0] == 'I' and state_right[0] == 'R':
                new_trans_times.insert(0, self.obs_times[0])

            else:
                pass

        elif self.hmm_skeleton[0] == 'S':
            pass
        elif self.hmm_skeleton[0] == 'I':
            new_trans_times = new_trans_times + [self.obs_times[0]]
        else:
            new_trans_times = new_trans_times + [self.obs_times[0]] * 2

        self.cand_trans_times = new_trans_times


    def met_hast_step(self, subj):
        n = subj
        a = self.accept_prob(n)

        accept_flag = npr.binomial(1, a)

        if accept_flag == 1:
            self.trans_times[n] = self.cand_trans_times

        else:
            pass
            

    def accept_prob(self, subj):
        n = subj
        
        # calculate the probability of current and proposed AD trajectories given parameter values beta and gamma

        #################
        # calc curr probs
        trans_times = self.trans_times
        trans_times_flat = []
        for indiv in trans_times:
            for t in indiv:
                if t != self.obs_times[0]:
                    trans_times_flat.append([ ['I','R'][ indiv.index(t) ], t ])

                else:
                    pass

        if len( self.trans_times[n] ) == 0:
            p_traj_curr = self.p[0]

        elif self.trans_times[n][0] > self.obs_times[0]:
            p_traj_curr = self.p[0]

        elif len(self.trans_times[n]) == 2 and self.trans_times[n][1] > self.obs_times[0]:
            p_traj_curr = self.p[1]

        else:
            p_traj_curr = self.p[2]

                
        trans_times_flat = [ trans_times_flat[i] for i in np.argsort([ t[1] for t in trans_times_flat ]) ]
        S, I, R = self.state_count(self.obs_times[0], trans_times = trans_times)
        p_x_curr = self.p[0]**S * self.p[1]**I * self.p[2]**R
        
        for t_left, t_right in zip(trans_times_flat[:-1], trans_times_flat[1:]):
            L = self.rate_matrix(self, t_left[1], trans_times = trans_times)
            P = spla.expm( (t_right[1]-t_left[1])*L )

            left_state_ind = bisect.bisect( trans_times[n], t_left[1] )
            right_state_ind = bisect.bisect( trans_times[n], t_right[1] )

            p_traj_curr *= P[ left_state_ind, right_state_ind ]

            if t_right[0] == 'I':
                S, I, R = self.state_count(t_right[1])
                p_x_curr *= self.beta * I * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))
            else:
                S, I, R = self.state_count(t_right[1])
                p_x_curr *= self.gamma * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))        

        ######################
        # calc candidate probs
        trans_times[n] = self.cand_trans_times
        trans_times_flat = []
        for indiv in trans_times:
            for t in indiv:
                if t != self.obs_times[0]:
                    trans_times_flat.append([ ['I','R'][ indiv.index(t) ], t ])

                else:
                    pass

        if len(self.cand_trans_times) == 0:
            p_traj_cand = self.p[0]        
                
        elif self.cand_trans_times[0] > self.obs_times[0]:
            p_traj_cand = self.p[0]

        elif len(self.cand_trans_times) == 2 and self.cand_trans_times[1] > self.obs_times[0]:
            p_traj_cand = self.p[1]

        else:
            p_traj_cand = self.p[2]
                
        trans_times_flat = [ trans_times_flat[i] for i in np.argsort([ t[1] for t in trans_times_flat ]) ]
        S, I, R = self.state_count(self.obs_times[0], trans_times = trans_times)
        p_x_cand = self.p[0]**S * self.p[1]**I * self.p[2]**R
        for t_left, t_right in zip(trans_times_flat[:-1], trans_times_flat[1:]):
            L = self.rate_matrix(self, t_left[1], trans_times = trans_times)
            P = spla.expm( (t_right[1]-t_left[1])*L )

            left_state_ind = bisect.bisect( trans_times[n], t_left[1] )
            right_state_ind = bisect.bisect( trans_times[n], t_right[1] )

            p_traj_cand *= P[ left_state_ind, right_state_ind ]
            
            if t_right[0] == 'I':
                S, I, R = self.state_count(t_right[1], trans_times = trans_times)
                p_x_cand *= self.beta * I * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))
            else:
                S, I, R = self.state_count(t_right[1], trans_times = trans_times)
                p_x_cand *= self.gamma * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))              

        if p_x_cand * p_traj_curr == 0:
            a = 0.

        elif p_x_curr * p_traj_cand == 0:
            a = 1.

        else:
            a = ( p_x_cand * p_traj_curr ) / ( p_x_curr * p_traj_cand )

        if not np.isnan(a):
            return min(a,1.)

        else:
            return 1
                
                        
                    
                
    def full_likelihood(self):
        # from the augmented data (AD) calculate I(t) for all observation times
        ad_infec_traj = np.array([ self.state_count(t)[1] for t in self.obs_times ])

        # calculate observation probabilities given AD 
        p1 = np.prod(np.array([ self.binom_pmf(i, self.rho, obs) for i, obs in zip(ad_infec_traj, self.y) ]))

        # calculate prob of AD at initialization time given init_prob_inf
        p2 = np.prod([ p ** s for p, s in zip(self.p, self.state_count(self.obs_times[0])) ])

        # calculate the probability of AD trajectory given parameter values beta and gamma
        trans_times_flat = []
        for indiv in self.trans_times:
            for t in indiv:
                if t != self.obs_times[0]:
                    trans_times_flat.append([ ['I','R'][ indiv.index(t) ], t ])

                else:
                    pass

        trans_times_flat = [ trans_times_flat[i] for i in np.argsort([ t[1] for t in trans_times_flat ]) ]
        p3 = 1
        for t_left, t_right in zip(trans_times_flat[:-1], trans_times_flat[1:]):
            if t_right[0] == 'I':
                S, I, R = self.state_count(t_right[1])
                p3 *= self.beta * I * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))
            else:
                S, I, R = self.state_count(t_right[1])
                p3 *= self.gamma * np.exp(-(t_right[1] - t_left[1]) * (self.beta * S * I + self.gamma * I))

        self.like_comps = [p1, p2, p3]
        return p1 * p2 * p3

    def update(self, subj):

        self.subj = subj
        self.hmm_step(subj)
        self.discrete_time_step(subj)
        self.event_time_step(subj)
        self.met_hast_step(subj)
        assert self.validAugDataChecker()
        self.prev_subj = subj

