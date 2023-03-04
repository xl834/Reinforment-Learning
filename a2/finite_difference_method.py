import numpy as np


def gradient(f, x, delta=1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    #TODO
    n, = x.shape
    gradient = np.zeros(n).astype('float64')
    # x_mat = np.tile(x, (n, 1))
    delta_diag = np.eye(n) * delta
    # gradient = ((f(x_mat + delta_diag) - f(x_mat - delta_diag)) / (2*delta))[0]
    for i in range(n):
        gradient[i] = (f(x + delta_diag[i]) - f(x - delta_diag[i])) / (2 * delta)

    if(gradient.dtype != np.float64): gradient.astype(np.float64)
    return gradient


def jacobian(f, x, delta=1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    #TODO
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input. 
    gradient = np.zeros((m, n)).astype('float64')
    
    #x_mat = np.tile(x, (n, 1))
    #delta_diag = np.eye(n) * delta
    #gradient = (f(x_mat + delta_diag) - f(x_mat - delta_diag)) / (2*delta)

    # for i in range(n):
    #     delta_v = np.zeros(n)
    #     delta_v[i] = delta
    #     gradient[i] = (f(x + delta_v) - f(x - delta_v)) / (2*delta)
    for i in range(n):
        x_p = x.copy()
        x_p[i] += delta
        f_p = f(x_p)
        x_m = x.copy()
        x_m[i] -= delta
        f_m = f(x_m)
        gradient[:, i] = (f_p - f_m) / (2*delta)
    if(gradient.dtype != np.float64): gradient.astype(np.float64)
    return gradient



def hessian(f, x, delta=1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    #TODO
    n, = x.shape
    hessian = np.zeros((n, n)).astype('float64')

    #x_mat = np.tile(x, (n, 1))
    delta_diag = np.eye(n)
    
    for i in range(n):
        hessian[i][i] = (f(x + delta*delta_diag[i]) - 2*f(x) + f(x - delta*delta_diag[i])) / (delta**2)
    
    for i in range(n):
        for j in range(i+1, n):
            fpp = f(x + delta * (delta_diag[i] + delta_diag[j]))
            fpm = f(x + delta * (delta_diag[i] - delta_diag[j]))
            fmp = f(x - delta * (delta_diag[i] - delta_diag[j]))
            fmm = f(x - delta * (delta_diag[i] + delta_diag[j]))
            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * delta**2)
            hessian[j, i] = hessian[i, j]

    if(hessian.dtype != np.float64): hessian.astype(np.float64)
    return hessian