import numpy as np


class DynamicProgramming:
    def __init__(self, MDP):
        self.R = MDP.R  # |A|x|S|
        self.P = MDP.P  # |A|x|S|x|S|
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    ####Helpers####
    def extractRpi(self, pi):
        return self.R[pi, np.arange(len(self.R[0]))]

    def extractPpi(self, pi):
        return self.P[pi, np.arange(len(self.P[0]))]

    ####Value Iteration###
    def computeVfromQ(self, Q, pi):
        row_idx = np.arange(0, 17)
        pi_tuple = tuple([row_idx, pi])
        return Q[pi_tuple]

    def computeQfromV(self, V):
        return (self.R).T + self.discount * (self.P @ V).T

    def extractMaxPifromQ(self, Q):
        return np.argmax(Q, axis=1)

    def extractMaxPifromV(self, V):
        return np.argmax(self.P @ V, axis=0) 

    def valueIterationStep(self, Q):
        Qmax = np.max(Q, axis=1)
        return (self.R).T + self.discount * (self.P @ Qmax).T

    def valueIteration(self, initialQ, tolerance=0.01):
        pi = np.zeros(self.nStates)  
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = np.inf
        Q = initialQ

        while(epsilon > tolerance):
            Q_t1 = self.valueIterationStep(Q)
            epsilon = np.max(np.abs(Q_t1 - Q))
            Q = Q_t1
            iterId += 1
        
        pi = np.argmax(Q, axis=1)
        V = self.computeVfromQ(Q, pi)

        return pi, V, iterId, epsilon

    ### EXACT POLICY EVALUATION  ###
    def exactPolicyEvaluation(self, pi):
        vector_r = self.extractRpi(pi)
        vector_p = self.extractPpi(pi)
        I = np.eye(self.nStates)
        return np.linalg.solve((I - self.discount * vector_p), vector_r)


    ### APPROXIMATE POLICY EVALUATION ###
    def approxPolicyEvaluation(self, pi, tolerance=0.01):
        V = np.zeros(self.nStates)
        epsilon = np.inf
        n_iters = 0

        vector_r = self.extractRpi(pi)
        vector_p = self.extractPpi(pi)

        while(epsilon > tolerance):
            V_t1 = vector_r + self.discount * vector_p @ V
            epsilon = np.max(np.abs(V_t1 - V))
            V = V_t1
            n_iters += 1

        return V, n_iters, epsilon

    def policyIterationStep(self, pi, exact):
        if(exact):
            V = self.exactPolicyEvaluation(pi)
        else:
            V, _, _ =  self.approxPolicyEvaluation(pi)
        return np.argmax((self.R) + self.discount * (self.P @ V), axis=0)

    def policyIteration(self, initial_pi, exact):
        iterId = 0
        pi = np.zeros(self.nStates)
        V = np.zeros(self.nStates)

        pi = initial_pi
        while True:
            iterId += 1
            pi_t1 = self.policyIterationStep(pi, exact)
            if(np.all(pi_t1 == pi)): break
            pi = pi_t1
        
        if(exact):
            V = self.exactPolicyEvaluation(pi)
        else:
            V, _, _ =  self.approxPolicyEvaluation(pi)

        return pi, V, iterId
