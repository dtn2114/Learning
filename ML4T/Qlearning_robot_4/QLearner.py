"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import timeit
class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.epochs = 0
        #initialize Q-Table
        self.Q = np.random.uniform(-1,1, size=(num_states, num_actions))

        #initialize T Model and R Model
        if self.dyna != 0:
            self.Tc = np.ones((self.num_states, self.num_actions, self.num_states))/1000000
            self.T = self.Tc/self.Tc.sum(axis=2,keepdims=True)
            self.R = np.ndarray(shape=(num_states,num_actions))
            self.R.fill(-1.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # randomly choose an action
        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s,:])
        #action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.Q[self.s,self.a] = (((1-self.alpha) * self.Q[self.s, self.a]) + self.alpha*\
                (r + self.gamma*self.Q[s_prime,np.argmax(self.Q[s_prime,:])]))
        #self.s = s_prime
       ########################################################################
        #################         START OF DYNA                     ############

        if self.dyna>0:
            self.Tc[self.s, self.a, s_prime] +=1
            self.T[self.s, self.a,:] = self.Tc[self.s,self.a,:]/self.Tc[self.s,self.a,:].sum()
            self.R[self.s,self.a] = (1-self.alpha)*self.R[self.s,self.a] + self.alpha*r

            if self.epochs > 1:
                dyna_S = np.random.randint(0,self.num_states,self.dyna)
                dyna_A = np.random.randint(0,self.num_actions, self.dyna)
                dyna_s_prim = [np.random.multinomial(1,self.T[dyna_S[i],dyna_A[i],:]).argmax() for i in range(self.dyna)]
                for i in range(0,self.dyna):
                    dyna_a = dyna_A[i]
                    dyna_s = dyna_S[i]
                    #dyna_s_prime = np.argmax(np.random.multinomial(1, self.T[dyna_s, dyna_a,:]))
                    r = self.R[dyna_s, dyna_a]
                    self.Q[dyna_s,dyna_a] = (1-self.alpha)*self.Q[dyna_s, dyna_a] + self.alpha * \
                                     (r +  self.gamma*np.max(self.Q[dyna_s_prim[i],:]))

        ##################        END OF DYNA                       ############
        ########################################################################
            #self.Q[dyna_s,dyna_a] = (1-self.alpha)*self.Q[dyna_s, dyna_a] + self.alpha * \
                            #         (r +  self.gamma*self.Q[dyna_s_prime,np.argmax(self.Q[dyna_s_prime,:])])


        # Randomly choosing an action.
        if rand.random() <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s_prime,:])
        self.rar = self.rar * self.radr
        self.epochs += 1
        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def author(self):
        return 'dnguyen333'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
