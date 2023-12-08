import numpy as np
from dxnesici import DXNESICI

def main():
    # Problem setting
    dim = 20
    dim_co = dim // 2
    dim_int = dim // 2
    domain_int = [list(range(-10, 11)) for _ in range(dim_int)]
    def n_int_tablet(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        xbar[:dim_co] *= 100
        return np.sum(xbar**2)

    # The other inputs
    mean = np.ones([dim, 1]) * 2.
    sigma = 1.0
    lamb = 6
    margin = 1.0 / (dim * lamb)

    # Running DX-NES-ICI
    dxnesici = DXNESICI(dim_co, domain_int, n_int_tablet, mean, sigma, lamb, margin)
    success, x_best, f_best = dxnesici.optimize(dim * 1e4, 1e-10)

    if success:
        print("x_best:{}, f_best:{}".format(x_best, f_best))
    else:
        print("Failure: The number of evaluations exceeded the max number!")
        print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
