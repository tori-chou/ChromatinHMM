"""
Unified HMM framework for chromatin state discovery with three pluggable
emission distributions: Bernoulli, Poisson, and Negative Binomial.

All forward/backward computation is in log space to avoid underflow.
"""

import numpy as np
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize_scalar
from numba import njit

@njit(cache=True)
def _forward_numba(emission, A, pi):
    """Scaled forward pass. Returns alpha_hat (T,K), scales (T,), log_Z."""
    T, K = emission.shape
    alpha = np.empty((T, K))
    scales = np.empty(T)

    alpha[0] = pi * emission[0]
    s = 0.0
    for k in range(K):
        s += alpha[0, k]
    if s == 0.0:
        s = 1e-300
    scales[0] = s
    for k in range(K):
        alpha[0, k] /= s

    for t in range(1, T):
        for k in range(K):
            v = 0.0
            for i in range(K):
                v += alpha[t - 1, i] * A[i, k]
            alpha[t, k] = v * emission[t, k]
        s = 0.0
        for k in range(K):
            s += alpha[t, k]
        if s == 0.0:
            s = 1e-300
        scales[t] = s
        for k in range(K):
            alpha[t, k] /= s

    log_Z = 0.0
    for t in range(T):
        log_Z += np.log(scales[t])
    return alpha, scales, log_Z


@njit(cache=True)
def _backward_numba(emission, A, scales):
    """Scaled backward pass. Returns beta_hat (T,K)."""
    T, K = emission.shape
    beta = np.empty((T, K))
    for k in range(K):
        beta[T - 1, k] = 1.0

    for t in range(T - 2, -1, -1):
        s = scales[t + 1]
        for k in range(K):
            v = 0.0
            for j in range(K):
                v += A[k, j] * emission[t + 1, j] * beta[t + 1, j]
            beta[t, k] = v / s
    return beta


class BernoulliEmission:
    """
    Independent Bernoulli emission for binarized ChIP-seq data.
    Each mark is modelled as Bernoulli(theta_km) in state k.
    """

    def __init__(self, K: int, M: int, rng: np.random.Generator):
        self.K, self.M = K, M
        self.theta = rng.uniform(0.2, 0.8, size=(K, M)) # (K, M)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        X: (T, M) float binary {0, 1}
        Returns (T, K) log emission log P(x_t | z_t = k).
        """
        t = np.clip(self.theta, 1e-6, 1.0 - 1e-6) # (K, M)
        return X @ np.log(t).T + (1.0 - X) @ np.log(1.0 - t).T

    def update(self, X_list: list, gamma_list: list) -> None:
        """M-step: theta_km = weighted mean of X over all sequences."""
        num = np.zeros((self.K, self.M))
        den = np.zeros(self.K)
        for X, gamma in zip(X_list, gamma_list):
            num += gamma.T @ X # (K, M)
            den += gamma.sum(axis=0) # (K,)
        self.theta = np.clip(num / den[:, None], 1e-6, 1.0 - 1e-6)


class PoissonEmission:
    """
    Independent Poisson emission for raw read count data.
    Each mark is modelled as Poisson(lambda_km) in state k.
    """

    def __init__(self, K: int, M: int, rng: np.random.Generator):
        self.K, self.M = K, M
        self.lam = rng.uniform(1.0, 10.0, size=(K, M)) # (K, M) rate

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        X: (T, M) non-negative integer counts
        Returns (T, K) log emission.
        """
        lam = np.clip(self.lam, 1e-6, None) # (K, M)
        # sum_m x_tm * log(lam_km) - lam_km - log(x_tm!)
        log_fact = gammaln(X + 1.0).sum(axis=1, keepdims=True) # (T, 1)
        return X @ np.log(lam).T - lam.sum(axis=1) - log_fact

    def update(self, X_list: list, gamma_list: list) -> None:
        """M-step: lambda_km = weighted mean count."""
        num = np.zeros((self.K, self.M))
        den = np.zeros(self.K)
        for X, gamma in zip(X_list, gamma_list):
            num += gamma.T @ X
            den += gamma.sum(axis=0)
        self.lam = np.clip(num / den[:, None], 1e-6, None)


class NBEmission:
    """
    Independent Negative Binomial emission for overdispersed count data.
    Each mark is modelled as NB(mu_km, r_km) in state k, where:
        P(x | r, mu) ∝ Gamma(x+r) / (Gamma(r) * x!) * (r/(r+mu))^r * (mu/(r+mu))^x

    mu is updated in closed form (weighted mean); r is fitted numerically.
    r is clamped to (0.1, 100) to prevent degeneracy.
    """

    def __init__(self, K: int, M: int, rng: np.random.Generator):
        self.K, self.M = K, M
        self.mu = rng.uniform(1.0, 10.0, size=(K, M)) # (K, M) mean
        self.r = np.full((K, M), 5.0) # (K, M) dispersion

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        X: (T, M) non-negative integer counts
        Returns (T, K) log emission.
        """
        r = np.clip(self.r,  0.1, 100.0)[None, :, :] # (1, K, M)
        mu = np.clip(self.mu, 1e-6, None)[None, :, :] # (1, K, M)
        x = X[:, None, :].astype(float) # (T, 1, M)
        lp = (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0)
              + r * np.log(r / (r + mu))
              + x * np.log(mu / (r + mu))) # (T, K, M)
        return lp.sum(axis=2) # (T, K)

    def update(self, X_list: list, gamma_list: list) -> None:
        """
        M-step:
          mu_km = weighted mean (closed form)
          r_km = MLE via minimize_scalar on neg log-likelihood (per state-mark pair)
        """
        # Closed-form mu update
        num = np.zeros((self.K, self.M))
        den = np.zeros(self.K)
        for X, gamma in zip(X_list, gamma_list):
            num += gamma.T @ X
            den += gamma.sum(axis=0)
        self.mu = np.clip(num / den[:, None], 1e-6, None)

        # Concatenate all data for r optimization
        X_all = np.concatenate(X_list,     axis=0).astype(float) # (N, M)
        gamma_all = np.concatenate(gamma_list, axis=0) # (N, K)
        lgf_all = gammaln(X_all + 1.0) # (N, M) — precomputed once

        for k in range(self.K):
            w = gamma_all[:, k]
            w = w / (w.sum() + 1e-300) # normalize weights
            for m in range(self.M):
                x_m = X_all[:, m]
                mu_km = float(self.mu[k, m])
                lgf = lgf_all[:, m]

                def _neg_ll(log_r: float, _x=x_m, _mu=mu_km, _lgf=lgf, _w=w) -> float:
                    r = np.exp(log_r)
                    return -float(np.dot(_w,
                        gammaln(_x + r) - gammaln(r) - _lgf
                        + r * np.log(r / (r + _mu))
                        + _x * np.log(_mu / (r + _mu))))

                # Optimise in log space; equivalent to searching r ∈ (0.1, 100)
                res = minimize_scalar(
                    _neg_ll,
                    bounds=(np.log(0.1), np.log(100.0)),
                    method='bounded',
                )
                self.r[k, m] = np.clip(np.exp(res.x), 0.1, 100.0)

class ChromatinHMM:
    """
    Hidden Markov Model for chromatin state discovery.

    Emission is pluggable: any object with
        log_prob(X) -> (T, K)
        update(X_list, gamma_list)

    Transition matrix and initial state use seed for reproducibility;
    emission parameters are initialized independently in each emission class.
    """

    def __init__(self, K: int, emission, n_iter: int = 100, seed: int = 42):
        self.K = K
        self.emission = emission
        self.n_iter = n_iter

        rng = np.random.default_rng(seed)
        self.log_A = np.log(rng.dirichlet(np.ones(K), size=K)) # (K, K) row-stochastic
        self.log_pi = np.log(np.full(K, 1.0 / K)) # (K,) uniform

    # Forward / backward
    # Each step is a numpy matmul (BLAS) rather than a logsumexp call to speed up training
    def _forward_scaled(self, emission: np.ndarray, A: np.ndarray, pi: np.ndarray):
        """Scaled forward algorithm via numba JIT. emission: (T,K) linear."""
        return _forward_numba(np.ascontiguousarray(emission), A, pi)

    def _backward_scaled(self, emission: np.ndarray, A: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Scaled backward algorithm via numba JIT. emission: (T,K) linear."""
        return _backward_numba(np.ascontiguousarray(emission), A,
                               np.ascontiguousarray(scales))

    def _e_step_one(self, X: np.ndarray):
        """
        Full E-step for a single observation sequence.
        Returns:
            log_Z float — sequence log-normalizer
            gamma (T, K) — state posteriors P(z_t=k | X)
            log_gamma_0 (K,) — log posterior at t=0
            log_xi_sum (K, K) — log of sum_t xi[t,i,j]
        """
        log_emission = self.emission.log_prob(X) # (T, K)
        emission = np.exp(log_emission) # linear for scaled fwd/bwd

        A = np.ascontiguousarray(np.exp(self.log_A))
        pi = np.ascontiguousarray(np.exp(self.log_pi))

        alpha, scales, log_Z = self._forward_scaled(emission, A, pi)
        beta = self._backward_scaled(emission, A, scales)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) # numerical safety

        log_gamma_0 = np.log(np.clip(gamma[0], 1e-300, None))

        # (the /c[t+1] keeps xi normalised; sum over i,j = 1 at each t)
        xi = (alpha[:-1, :, None] # (T-1, K, 1)
               * A[None, :, :] # (1,   K, K)
               * emission[1:, None, :] # (T-1, 1, K)
               * beta[1:, None, :] # (T-1, 1, K)
               / scales[1:, None, None]) # (T-1, 1, 1)
        xi_sum = xi.sum(axis=0) # (K, K)
        log_xi_sum = np.log(np.clip(xi_sum, 1e-300, None))

        return log_Z, gamma, log_gamma_0, log_xi_sum

    def fit(self, X_list: list, verbose: bool = True) -> list:
        """
        Baum-Welch EM on a list of observation sequences.
        X_list: list of (T_s, M) arrays (same M, variable T_s)
        Returns: (ll_history, converged) — list of per-iteration total
        log-likelihoods and a bool indicating whether the tolerance was met.
        """
        ll_history = []
        converged = False

        for iteration in range(self.n_iter):
            total_ll = 0.0
            log_xi_total = np.full((self.K, self.K), -np.inf)
            gamma_list = []
            log_gamma0_list = []

            for X in X_list:
                log_Z, gamma, log_gamma0, log_xi_sum = self._e_step_one(X)
                total_ll += log_Z
                gamma_list.append(gamma)
                log_gamma0_list.append(log_gamma0)
                log_xi_total = np.logaddexp(log_xi_total, log_xi_sum)

            # M-step: initial state (mean over sequences)
            self.log_pi = (logsumexp(np.array(log_gamma0_list), axis=0)
                           - np.log(len(X_list)))

            # M-step: transition matrix
            self.log_A = log_xi_total - logsumexp(log_xi_total, axis=1, keepdims=True)

            # M-step: emission parameters
            self.emission.update(X_list, gamma_list)

            ll_history.append(total_ll)
            if verbose:
                delta = total_ll - ll_history[-2] if iteration > 0 else float('nan')
                print(f" Iter {iteration + 1:3d}: LL = {total_ll:>14.2f} Δ = {delta:>12.6f}")

            if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < 0.1:
                converged = True
                if verbose:
                    print(f" Converged at iteration {iteration + 1}.")
                break

        if not converged and verbose:
            print(f" Warning: hit iteration limit ({self.n_iter}) without converging.")

        return ll_history, converged

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding.
        Returns 0-indexed state sequence of shape (T,).
        """
        log_emission = self.emission.log_prob(X)
        T = log_emission.shape[0]
        delta = np.empty((T, self.K))
        psi = np.zeros((T, self.K), dtype=np.int32)
        delta[0] = self.log_pi + log_emission[0]

        for t in range(1, T):
            v = delta[t - 1, :, None] + self.log_A # (K, K): v[i,j]
            psi[t] = v.argmax(axis=0)
            delta[t] = v.max(axis=0) + log_emission[t]

        states = np.empty(T, dtype=np.int32)
        states[-1] = int(delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def log_likelihood(self, X: np.ndarray) -> float:
        """Log-likelihood of a single observation sequence via the forward algorithm."""
        emission = np.exp(self.emission.log_prob(X))
        A = np.ascontiguousarray(np.exp(self.log_A))
        pi = np.ascontiguousarray(np.exp(self.log_pi))
        _, _, log_Z = self._forward_scaled(emission, A, pi)
        return log_Z