import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        #TODO
        ns = s_star.shape[0]
        na = a_star.shape[0]
        f = self.f(s_star, a_star)
        c = self.c(s_star, a_star)
        x = np.concatenate((s_star, a_star), axis=0)

        fA_lambda = lambda s: self.f(s, np.array([0]))
        fB_lambda = lambda a: self.f(np.zeros(ns), a)
        cH_lambda = lambda x: self.c(np.array(x[:ns]), np.array([x[ns]]))
        cq_lambda = lambda s: self.c(s, np.array([0]))
        cr_lambda = lambda a: self.c(np.zeros(ns), a)

        A = jacobian(fA_lambda, s_star)
        B = jacobian(fB_lambda, a_star)
        H = hessian(cH_lambda, x)
        H_ = np.zeros((ns + na, ns + na))
        w, v = np.linalg.eig(H)
        lbd = 1e-4
        # if(np.all(w > 0) == False):
        for i, wi in enumerate(w):
            v_ = v[i].reshape((5, 1))
            H_ += (max(0, wi) + lbd) * (v_ @ v_.T)
        H = H_
        Q = H[0:ns, 0:ns]
        M = H[0:ns, ns:(ns+na)]
        R = H[ns:(ns+na), ns:(ns+na)]
        q = gradient(cq_lambda, s_star)
        q = q.reshape((ns, 1))
        r = gradient(cr_lambda, a_star)
        r = r.reshape((na, 1))
        b = self.c(s_star, a_star) + (0.25 * (s_star.T) @ Q @ s_star) + (0.5 * (a_star.T) @ R @ a_star) + (s_star.T @ M @ a_star) - (q.T @ s_star) - (r.T @ a_star)
        m = self.f(s_star, a_star) - A @ s_star - B @ a_star
        m = m.reshape((ns, 1))

        ret = lqr(A, B, m, Q, R, M, q, r, b, T)

        return [(policy[0], policy[1]) for policy in ret]
        #return [(np.zeros((1,4)), np.zeros(1)) for _ in range(T)]

class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a



