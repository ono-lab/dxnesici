import math
import numpy as np
from dxnesici import DXNESICI

def main():
    # problem setting
    dim = 80
    dim_int = dim // 2
    dim_co = dim // 2
    domain_int = np.tile(np.arange(-10, 11, 1), (dim_int, 1))
    def reversed_ellipsoid_int(x):
        np.round(x[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients[dim_co:] * x[:dim_co])**2) + np.sum((coefficients[:dim_co] * x[dim_co:])**2)

    # the other inputs
    mean = np.ones([dim, 1]) * 2.
    sigma = 1.0
    lamb = 18 # note that lamb (population size) should be even number
    margin = 1.0 / (dim * lamb)

    dxnesicmi = DXNESICI(dim_co, domain_int, reversed_ellipsoid_int, mean, sigma, lamb, margin)
    _, f_best, x_best = dxnesicmi.optimize(dim * 1e4, 1e-10, print_progress = True)

    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
