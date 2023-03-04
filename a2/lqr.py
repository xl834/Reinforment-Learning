import numpy as np




def compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p):
    """
    Compute the Q function parameters for time step t.
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
        R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
        M (2d numpy array): A numpy array with shape (n_s, n_a)
        q (2d numpy array): A numpy array with shape (n_s, 1)
        r (2d numpy array): A numpy array with shape (n_a, 1)
        b (1d numpy array): A numpy array with shape (1,)
        P (2d numpy array): A numpy array with shape (n_s, n_s). This is the quadratic term of the
            value function equation from time step t+1. P is PSD.
        y (2d numpy array): A numpy array with shape (n_s, 1).  This is the linear term
            of the value function equation from time step t+1
        p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
            value function equation from time step t+1
    Returns:
        C (2d numpy array): A numpy array with shape (n_s, n_s)
        D (2d numpy array): A numpy array with shape (n_s, n_a)
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s,1)
        g (2d numpy array): A numpy array with shape (n_a,1)
        h (1d numpy array): A numpy array with shape (1,)

        where the following equation should hold
        Q_t^*(s) = s^T C s + a^T D s + s^T E a + f^T s  + g^T a + h

    """
    # TODO
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert Q.shape == (n_s, n_s)
    assert R.shape == (n_a, n_a)
    assert M.shape == (n_s, n_a)
    assert q.shape == (n_s, 1)
    assert r.shape == (n_a, 1)
    assert b.shape == (1, )
    assert P.shape == (n_s, n_s)
    assert y.shape == (n_s, 1)
    assert p.shape == (1, )
    C = np.zeros((n_s, n_s))
    D = np.zeros((n_s, n_a))
    E = np.zeros((n_s, n_a))
    f = np.zeros((n_s, 1))
    g = np.zeros((n_a, 1))
    h = np.zeros(1)

    C = Q + A.T @ P @ A
    D = R + B.T @ P @ B
    E = M + 2 * A.T @ P @ B
    f = (q.T + 2 * m.T @ P @ A + y.T @ A).T
    g = (r.T + 2 * m.T @ P @ B + y.T @ B).T
    h = b + m.T @ P @ m + y.T @ m + p

    if(C.dtype != np.float64): C.astype(np.float64)
    if(D.dtype != np.float64): D.astype(np.float64)
    if(E.dtype != np.float64): E.astype(np.float64)
    if(f.dtype != np.float64): f.astype(np.float64)
    if(g.dtype != np.float64): g.astype(np.float64)
    if(h.dtype != np.float64): h.astype(np.float64) 

    return C, D, E, f, g, h


def compute_policy(A, B, m, C, D, E, f, g, h):
    """
    Compute the optimal policy at the current time step t
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)


    Let Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a  + h
    Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s, 1)
        g (2d numpy array): A numpy array with shape (n_a, 1)
        h (1d numpy array): A numpy array with shape (1, )
    Returns:
        K_t (2d numpy array): A numpy array with shape (n_a, n_s)
        k_t (2d numpy array): A numpy array with shape (n_a, 1)

        where the following holds
        \pi*_t(s) = K_t s + k_t
    """
    #TODO
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert C.shape == (n_s, n_s)
    assert D.shape == (n_a, n_a)
    assert E.shape == (n_s, n_a)
    assert f.shape == (n_s, 1)
    assert g.shape == (n_a, 1)
    assert h.shape == (1, )

    n_s, n_a = B.shape
    K_t = np.zeros((n_a, n_s))
    k_t = np.zeros(n_a)

    I = np.eye(n_a)
    D_inv = np.linalg.solve(D, I)

    K_t = -0.5 * D_inv @ (E.T)
    k_t = -0.5 * D_inv @ g
    k_t = k_t.flatten()

    if(K_t.dtype != np.float64): K_t.astype(np.float64)
    if(k_t.dtype != np.float64): k_t.astype(np.float64)

    return K_t, k_t


def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    """
    Compute the V function parameters for the next time step
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let V_t^*(s) = s^TP_ts + y_t^Ts + p_t
    Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s, 1)
        g (2d numpy array): A numpy array with shape (n_a, 1)
        h (1d numpy array): A numpy array with shape (1, )
        K (2d numpy array): A numpy array with shape (n_a, n_s)
        k (2d numpy array): A numpy array with shape (n_a, 1)

    Returns:
        P_h (2d numpy array): A numpy array with shape (n_s, n_s)
        y_h (2d numpy array): A numpy array with shape (n_s, 1)
        p_h (1d numpy array): A numpy array with shape (1,)
    """
    #TODO
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert C.shape == (n_s, n_s)
    assert D.shape == (n_a, n_a)
    assert E.shape == (n_s, n_a)
    assert f.shape == (n_s, 1)
    assert g.shape == (n_a, 1)
    assert h.shape == (1, )
    assert K.shape == (n_a, n_s)
    #assert k.shape == (n_a, 1)
    #assert k.shape == (n_a, )

    P_h = np.zeros((n_s, n_s))
    y_h = np.zeros((n_s, 1))
    p_h = np.zeros(1)

    P_h = C + K.T @ D @ K + E @ K
    y_h = (f.T + 2 * k.T @ D @ K + k.T @ E.T + g.T @ K).T
    p_h = k.T @ D @ k + g.T @ k + h

    if(P_h.dtype != np.float64): P_h.astype(np.float64)
    if(y_h.dtype != np.float64): y_h.astype(np.float64)
    if(p_h.dtype != np.float64): p_h.astype(np.float64)

    return P_h, y_h, p_h


def lqr(A, B, m, Q, R, M, q, r, b, T):
    """
    Compute optimal policies by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} s_t^T Q s_t + a_t^T R a_t + s_t^T M a_t + q^T s_t + r^T a_t
    subject to s_{t+1} = A s_t + B a_t + m, a_t = \pi_t(s_t)

    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let optimal \pi*_t(s) = K_t s + k_t

    Parameters:
    A (2d numpy array): A numpy array with shape (n_s, n_s)
    B (2d numpy array): A numpy array with shape (n_s, n_a)
    m (2d numpy array): A numpy array with shape (n_s, 1)
    Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD.
    R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
    M (2d numpy array): A numpy array with shape (n_s, n_a)
    q (2d numpy array): A numpy array with shape (n_s, 1)
    r (2d numpy array): A numpy array with shape (n_a, 1)
    b (1d numpy array): A numpy array with shape (1,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (n_a, n_s), the shape of k_t is (n_a,)
    """
    n_s, n_a = B.shape

    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert Q.shape == (n_s, n_s)
    assert R.shape == (n_a, n_a)
    assert M.shape == (n_s, n_a)
    assert q.shape == (n_s, 1)
    assert r.shape == (n_a, 1)
    assert b.shape == (1, )
    # TODO
    ret = []

    P = np.zeros((n_a, n_a))
    y = np.zeros((n_s, 1))

    R_inv = np.linalg.solve(R, np.eye(n_a))
    K_t_1 = -0.5 * R_inv @ M.T
    k_t_1 = -0.5 * R_inv @ r
    k_t_1 = k_t_1.flatten()
    ret.insert(0, (K_t_1, k_t_1))
    T_ = T
    while(T_ > 1):
        # compute Q params
        if(T_ == T):
            P = Q - 0.25 * M @ R_inv @ M.T
            y = (q.T - 0.5 * r.T @ R_inv @ M.T).T
            p = b - 0.25 * r.T @ R_inv @ r
        else:
            P, y, p = compute_V_params(A, B, m, C, D, E, f, g, h, K_t, k_t)
        p = p.reshape(1)
        C, D, E, f, g, h = compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p)
        h = h.reshape(1)
        K_t, k_t = compute_policy(A, B, m, C, D, E, f, g, h)
        k_t = k_t.flatten()
        if(K_t.dtype != np.float64): K_t.astype(np.float64)
        if(k_t.dtype != np.float64): k_t.astype(np.float64)
        ret.insert(0, (K_t, k_t))
        T_ = T_ - 1
    return ret