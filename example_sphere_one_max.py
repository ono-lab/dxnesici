import numpy as np
from dxnesici import DXNESICI

def main():
    # Problem setting
    dim = 40
    dim_int = dim // 2
    dim_co = dim // 2
    domain_int = [[0,1] for _ in range(dim_int)]
    def sphere_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        return np.sum(xbar[:dim_co]**2) + dim_int - np.sum(xbar[dim_co:])

    # The other inputs
    mean = np.ones([dim, 1])
    mean[:dim_co] *= 2.
    mean[dim_co:] *= 0.5
    sigma = 1.0
    lamb = 10
    margin = 1.0 / (dim * lamb)

    # Running DX-NES-ICI
    for i in range(10):
        dxnesici = DXNESICI(dim_co, domain_int, sphere_one_max, mean, sigma, lamb, margin)
    success, x_best, f_best = dxnesici.optimize(dim * 1e4, 1e-10)

    if success:
        print("x_best:{}, f_best:{}".format(x_best, f_best))
    else:
        print("Failure: The number of evaluations exceeded the max number!")
        print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
