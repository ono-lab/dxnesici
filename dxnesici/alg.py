import math
from bisect import bisect_left

import numpy as np
from scipy.linalg import eigh, eigvalsh, expm
from scipy.stats import norm


def calculate_h_inv(dim):
    a = 1.
    for _ in range(10000):
        a -= ((1. + a * a) * math.exp(a * a / 2.) / 0.24 - 10. - dim) / (
            (2. * a * math.exp(a * a / 2.) + a * (1. + a * a) * math.exp(a * a / 2.)) / 0.24)
    return a

class DXNESICI:
    def __init__(self, dim_co, domain_int, f, m, sigma, lamb, margin, minimal_eigval = 1e-30, maximal_cond_num = 1e14, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        self.dim_co = dim_co
        self.dim_int = len(domain_int)
        self.dim = dim_co + self.dim_int
        self.domain_int = domain_int
        for i in range(self.dim_int):
            self.domain_int[i].sort()
            assert (len(domain_int[i]) >= 2), f"The number of elements of the domain in an integer variable must be greater than or equal to 2"
        self.lim = [[(domain_int[i][j] + domain_int[i][j + 1]) / 2. for j in range(len(domain_int[i]) - 1)] for i in range(self.dim_int)]
        self.norm_ci = norm.ppf(1. - margin)

        self.f = f
        self.m = m
        self.sigma = sigma
        self.b = np.eye(self.dim)
        self.bbT = self.b @ self.b.T
        self.p_sigma = np.zeros([self.dim, 1])
        self.lamb = lamb
        assert (lamb > 0 and lamb % 2 == 0), f"The value of 'lamb' must be an even, positive integer greater than 0"
        self.margin = margin
        self.c_gamma = 1. / (3. * (self.dim - 1.))
        self.d_gamma = min(1., self.dim / lamb)
        self.chiN = np.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.gamma = np.ones(self.dim)
        self.tau = np.zeros(self.dim)
        self.threshold_over = 5
        self.no_of_over = 0
        self.eye = np.eye(self.dim)

        self.w_rank_hat = (np.log(self.lamb / 2. + 1.) - np.log(np.arange(1, self.lamb + 1))).reshape(self.lamb, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0.)] = 0.
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.lamb)
        self.mu_eff = 1. / ((self.w_rank + (1. / self.lamb)).T @ (self.w_rank + (1. / self.lamb)))[0][0]
        self.c_sigma = ((self.mu_eff + 2.) / (self.dim + self.mu_eff + 5.)) / (2. * np.log(self.dim + 1.))
        self.h_inv = calculate_h_inv(self.dim)
        self.alpha = self.h_inv * min(1., math.sqrt(self.lamb / self.dim))
        self.w_dist_hat = lambda zi: math.exp(self.alpha * np.linalg.norm(zi))

        self.state_move = 0
        self.state_stag = 1
        self.state_conv = 2
        self.eta_sigmas = [
            1.,
            math.tanh((0.024 * self.lamb + 0.7 * self.dim + 20.) / (self.dim + 12.)),
            2. * math.tanh((0.025 * self.lamb + 0.75 * self.dim + 10.) / (self.dim + 4.))
        ]
        self.eta_bs = [
            1.5 * 120. * self.dim / (47. * self.dim * self.dim + 6400.) * np.tanh(0.02 * self.lamb),
            1.4 * 120. * self.dim / (47. * self.dim * self.dim + 6400.) * np.tanh(0.02 * self.lamb),
            0.1 * 120. * self.dim / (47. * self.dim * self.dim + 6400.) * np.tanh(0.02 * self.lamb)
        ]

        self.g = 0
        self.no_of_evals = 0
        self.minimal_eigval = minimal_eigval
        self.maximal_cond_num = maximal_cond_num

    def optimize(self, max_no_of_evals, target_eval, print_progress = False):
        success = False
        while self.no_of_evals < max_no_of_evals:
            x_best, f_best = self.next_generation()
            if print_progress:
                print("{} #Eval:{} f_best:{}".format(self.g, self.no_of_evals, f_best))
            if f_best < target_eval:
                success = True
                print("#Itr:{} #Eval:{}".format(self.g, self.no_of_evals))
                break
            eigvals = eigvalsh(self.bbT)
            if np.min(eigvals) < 1e-30:
                print("Failure: The smallest eigenvalue became lower than a limit value!")
                break
            elif np.max(eigvals) / np.min(eigvals) > 1e14:
                print("Failure: The condition number became higher than a limit value!")
                break
        return success, x_best, f_best


    def next_generation(self):
        dim_co = self.dim_co
        dim_int = self.dim_int
        dim = self.dim
        lamb = self.lamb
        bbT_cur = self.bbT
        m_cur = np.copy(self.m)

        # sample offspring
        z_positive = np.random.randn(dim, lamb//2)
        z = np.zeros([dim, lamb])
        z[:, :lamb//2] = z_positive
        z[:, lamb//2:] = -z_positive
        x = m_cur + self.sigma * self.b @ z
        xbar = np.array([[
            x[i][j] if i < dim_co else
            self.domain_int[i - dim_co][bisect_left(self.lim[i - dim_co], x[i][j])]
            for j in range(lamb)] for i in range(dim)])
        evals = np.array([self.f(np.array(xbar[:, i].reshape(dim, 1))) for i in range(lamb)])
        sorted_indices = np.argsort(evals)
        z = z[:, sorted_indices]
        x = x[:, sorted_indices]
        xbar = xbar[:, sorted_indices]
        # update evolution path
        self.p_sigma = (1. - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2. - self.c_sigma) * self.mu_eff) * (z @ self.w_rank)
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        # determine search phase
        self.no_of_over = self.no_of_over + 1 if p_sigma_norm >= self.chiN else 0
        state = self.state_move if p_sigma_norm >= self.chiN and self.no_of_over >= self.threshold_over else self.state_stag if p_sigma_norm >= 0.1 * self.chiN else self.state_conv
        # calculate weights
        w_tmp = np.array([self.w_rank_hat[i] * self.w_dist_hat(np.array(z[:, i].reshape(dim, 1))) for i in range(lamb)]).reshape(lamb, 1)
        w_dist = w_tmp / sum(w_tmp) - 1. / lamb
        weights = w_dist if state == self.state_move else self.w_rank
        # calculate learning rate of sigma and b
        eta_sigma = self.eta_sigmas[state]
        eta_b = self.eta_bs[state]
        # calculate gradients
        grad_M = np.array([weights[i] * (z[:, i].reshape(dim, 1) @ z[:, i].reshape(1, dim) - self.eye) for i in range(lamb)]).sum(axis=0)
        grad_sigma = np.trace(grad_M) / dim
        grad_b = grad_M - grad_sigma * self.eye
        grad_delta = z @ weights
        # calculate learning rate of m
        eta_m = np.ones([dim, 1])
        ci = (self.norm_ci * self.sigma * np.sqrt(np.diag(bbT_cur)))[dim_co:].reshape(dim_int,1)
        ci_up = m_cur[dim_co:] + ci
        ci_low = m_cur[dim_co:] - ci
        resolution = np.array([bisect_left(self.lim[i], ci_up[i]) - bisect_left(self.lim[i], ci_low[i]) for i in range(dim_int)])
        l_close = np.array([
            self.lim[i][min(len(self.lim[i]) - 1, max(0, bisect_left(self.domain_int[i], m_cur[dim_co + i]) - 1))]
            for i in range(dim_int)]).reshape(dim_int,1)
        condition_bias = (resolution <= 1) & ~(((self.b @ grad_delta)[dim_co:] < 0.0) ^ (m_cur[dim_co:] - l_close < 0.0)).reshape(dim_int)
        eta_m[dim_co:][np.where(condition_bias)] += 1.0
        # update parameters
        self.m += self.sigma * eta_m * self.b @ grad_delta
        self.sigma *= np.exp((eta_sigma / 2.) * grad_sigma)
        self.b = self.b @ expm((eta_b / 2.) * grad_b)
        # emphasize expansion
        eigvec = np.array(eigh(bbT_cur)[1])
        self.bbT = self.b @ self.b.T
        tau_tmp = np.array(
            [
                (eigvec[:, i] @ self.bbT @ eigvec[:, i])
                / (eigvec[:, i] @ bbT_cur @ eigvec[:, i])
                - 1.0
                for i in range(self.dim)
            ]
        )
        tau_index = np.argsort(tau_tmp)[::-1]
        self.tau = tau_tmp[tau_index]
        self.gamma = np.maximum(
            1.0,
            (1.0 - self.c_gamma) * self.gamma
            + self.c_gamma * np.sqrt(1.0 + self.d_gamma * self.tau),
        )
        if state == self.state_move:
            q = self.eye + (self.gamma[0] - 1.0) * np.array(
                [
                    eigvec[:, tau_index[i]].reshape(dim, 1)
                    @ eigvec[:, tau_index[i]].reshape(1, dim)
                    for i in range(self.dim)
                    if self.tau[i] > 0.0
                ]
            ).sum(axis=0)
            drt_det_q = math.pow(np.linalg.det(q), 1.0 / self.dim)
            self.sigma *= drt_det_q
            self.b = q @ self.b / drt_det_q
        self.bbT = self.b @ self.b.T
        # leap or correct m
        ci = (self.norm_ci * self.sigma * np.sqrt(np.diag(self.bbT)))[self.dim_co:].reshape(dim_int,1)
        ci_up = self.m[self.dim_co:] + ci
        ci_low = self.m[self.dim_co:] - ci
        resolution = np.array([bisect_left(self.lim[i], ci_up[i]) - bisect_left(self.lim[i], ci_low[i]) for i in range(self.dim_int)])
        self.m[dim_co:] = np.array([
            self.m[i + dim_co] if resolution[i] != 0 else
            self.lim[i][0] - ci[i] if self.m[i + dim_co] <= self.lim[i][0] else
            self.lim[i][-1] + ci[i] if self.lim[i][-1] < self.m[i + dim_co] else
            self.lim[i][bisect_left(self.lim[i], self.m[i + dim_co]) - 1] + ci[i] if self.m[i + dim_co] <= l_close[i] else
            self.lim[i][bisect_left(self.lim[i], self.m[i + dim_co])] - ci[i]
            for i in range(dim_int)])

        self.g += 1
        self.no_of_evals += lamb
        index_best = sorted_indices[0]
        x_best = xbar[:, 0]
        f_best = evals[index_best]
        return x_best, f_best
