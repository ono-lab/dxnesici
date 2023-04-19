import numpy as np
from dxnesici import DXNESICI

def main():
    # problem setting
    dim = 40
    dim_int = dim // 2
    dim_co = dim // 2
    domain_int = np.tile(np.arange(-10, 11, 1), (dim_int, 1))
    def n_int_tablet(x):
        x[:dim_co] *= 100
        np.round(x[dim_co:])
        return np.sum(x**2)

    # the other inputs
    mean = np.ones([dim, 1]) * 2.
    sigma = 1.0
    lamb = 8 # note that lamb (population size) should be even number
    margin = 1.0 / (dim * lamb)

    dxnesicmi = DXNESICI(dim_co, domain_int, n_int_tablet, mean, sigma, lamb, margin)
    _, f_best, x_best = dxnesicmi.optimize(dim * 1e4, 1e-10, print_progress = True)

    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
