import numpy as np

# def f2(x):
#     return (x - x_s) @ Q @ (x - x_s) + q @ (x - x_s) + 1

# Q = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
# q = np.array([10, 11, 12], dtype=np.float64)
# x_s = np.array([0.1, 0.2, 0.3], dtype=np.float64)
# delta = 1e-5
# delta_diag = np.eye(3) * delta

# a1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

# x_0 = x_s + delta_diag[0]
# x_0_ = x_s - delta_diag[0]

# print(a1*a1, a1@a1)

gradient = np.zeros((4, 1)).astype('float64')


print(gradient.shape)
