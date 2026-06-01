import warnings
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from numba import njit
from utils import logsumexp2, logsumexp4
from models.base import BaseTimeSeriesModel

warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL"
)

@njit
def EM(x, A, mu, var, max_iter=100000, tol=1e-6, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    T = len(x)

    # var_x
    mean_x = 0.0
    for t in range(T):
        mean_x += x[t] / T

    var_x = 0.0
    for t in range(T):
        var_x += (x[t] - mean_x) * (x[t] - mean_x) / (T - 1)

    A = np.asarray(A, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    prev_ll = -np.inf
    pi = np.zeros(2, dtype=np.float64)
    logA = np.zeros((2, 2), dtype=np.float64)
    logpi = np.zeros(2, dtype=np.float64)
    logB = np.zeros((T, 2), dtype=np.float64)
    log_alpha = np.zeros((T, 2), dtype=np.float64)
    log_beta = np.zeros((T, 2), dtype=np.float64)
    log_gamma = np.zeros((T, 2), dtype=np.float64)
    gamma = np.zeros((T, 2), dtype=np.float64)
    log_xi = np.zeros((T - 1, 2, 2), dtype=np.float64)
    xi = np.zeros((T - 1, 2, 2), dtype=np.float64)

    for iter in range(max_iter):
        p12 = A[0, 1]
        p21 = A[1, 0]

        pi[0] = p21 / (p21 + p12 + eps)
        pi[1] = p12 / (p21 + p12 + eps)

        logA[0, 0] = np.log(A[0, 0] + eps)
        logA[0, 1] = np.log(A[0, 1] + eps)
        logA[1, 0] = np.log(A[1, 0] + eps)
        logA[1, 1] = np.log(A[1, 1] + eps)

        logpi[0] = np.log(pi[0] + eps)
        logpi[1] = np.log(pi[1] + eps)

        for t in range(T):
            logB[t, 0] = -0.5 * np.log(2 * np.pi * var[0]) - (x[t] - mu[0])**2 / (2 * var[0])
            logB[t, 1] = -0.5 * np.log(2 * np.pi * var[1]) - (x[t] - mu[1])**2 / (2 * var[1])

        #forward
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            log_alpha[t, 0] = logB[t, 0] + logsumexp2(log_alpha[t-1, 0] + logA[0, 0], log_alpha[t-1, 1] + logA[1, 0])
            log_alpha[t, 1] = logB[t, 1] + logsumexp2(log_alpha[t-1, 0] + logA[0, 1], log_alpha[t-1, 1] + logA[1, 1])
            log_gamma[t, 0] = log_alpha[t, 0]
            log_gamma[t, 1] = log_alpha[t, 1]

        ll = logsumexp2(log_alpha[-1, 0], log_alpha[-1, 1])

        # backward
        for t in range(T - 2, -1, -1):
            log_beta[t, 0] = logsumexp2(logA[0, 0] + logB[t+1, 0] + log_beta[t+1, 0], logA[0, 1] + logB[t+1, 1] + log_beta[t+1, 1])
            log_beta[t, 1] = logsumexp2(logA[1, 0] + logB[t+1, 0] + log_beta[t+1, 0], logA[1, 1] + logB[t+1, 1] + log_beta[t+1, 1])
            log_gamma[t, 0] += log_beta[t, 0]
            log_gamma[t, 1] += log_beta[t, 1]

        for t in range(T):
            lse = logsumexp2(log_gamma[t, 0], log_gamma[t, 1])
            log_gamma[t, 0] -= lse
            log_gamma[t, 1] -= lse
            gamma[t, 0] = np.exp(log_gamma[t, 0])
            gamma[t, 1] = np.exp(log_gamma[t, 1])

        # xi
        for t in range(T - 1):
            log_xi[t, 0, 0] = log_alpha[t, 0] + logA[0, 0] + logB[t + 1, 0] + log_beta[t + 1, 0]
            log_xi[t, 0, 1] = log_alpha[t, 0] + logA[0, 1] + logB[t + 1, 1] + log_beta[t + 1, 1]
            log_xi[t, 1, 0] = log_alpha[t, 1] + logA[1, 0] + logB[t + 1, 0] + log_beta[t + 1, 0]
            log_xi[t, 1, 1] = log_alpha[t, 1] + logA[1, 1] + logB[t + 1, 1] + log_beta[t + 1, 1]       
            
            den = logsumexp4(log_xi[t, 0, 0], log_xi[t, 0, 1], log_xi[t, 1, 0], log_xi[t, 1, 1])
            log_xi[t, 0, 0] -= den
            log_xi[t, 0, 1] -= den
            log_xi[t, 1, 0] -= den
            log_xi[t, 1, 1] -= den

            xi[t, 0, 0] = np.exp(log_xi[t, 0, 0])
            xi[t, 0, 1] = np.exp(log_xi[t, 0, 1])
            xi[t, 1, 0] = np.exp(log_xi[t, 1, 0])
            xi[t, 1, 1] = np.exp(log_xi[t, 1, 1])

        # M-step
        # A update
        xi_sum = np.zeros((2, 2), dtype=np.float64)
        for t in range(T-1):
            xi_sum[0, 0] += xi[t, 0, 0]
            xi_sum[0, 1] += xi[t, 0, 1]
            xi_sum[1, 0] += xi[t, 1, 0]
            xi_sum[1, 1] += xi[t, 1, 1]

        A[0, 0] = xi_sum[0, 0] / (xi_sum[0, 0] + xi_sum[0, 1])
        A[0, 1] = xi_sum[0, 1] / (xi_sum[0, 0] + xi_sum[0, 1])
        A[1, 0] = xi_sum[1, 0] / (xi_sum[1, 0] + xi_sum[1, 1])
        A[1, 1] = xi_sum[1, 1] / (xi_sum[1, 0] + xi_sum[1, 1])

        # mu update
        mu = np.zeros(2, dtype=np.float64)
        den = np.zeros(2, dtype=np.float64)
        for t in range(T):
            mu[0] += gamma[t, 0] * x[t]
            den[0] += gamma[t, 0]
            mu[1] += gamma[t, 1] * x[t]
            den[1] += gamma[t, 1]
        
        mu[0] /= den[0]
        mu[1] /= den[1]

        # var update
        var = np.zeros(2, dtype=np.float64)
        for t in range(x.shape[0]):
            d = x[t] - mu[0]
            var[0] += gamma[t, 0] * d * d
            d = x[t] - mu[1]
            var[1] += gamma[t, 1] * d * d

        var[0] /= den[0]
        var[1] /= den[1]

        # clipping var
        if var[0] < var_x/9: var[0] = var_x/9
        if var[0] > var_x: var[0] = var_x
        if var[1] < var_x: var[1] = var_x
        if var[1] > 9*var_x: var[1] = 9*var_x

        if var[0] > var[1]:
            var[0], var[1] = var[1], var[0]
            mu[0], mu[1] = mu[1], mu[0]
            A[0, 0], A[1, 1] = A[1, 1], A[0, 0]
            A[0, 1], A[1, 0] = A[1, 0], A[0, 1]

        diff = np.abs(ll - prev_ll)
        if diff < tol: break
        prev_ll = ll

    return (A, mu, var, ll, diff, iter, gamma)

class MSM_2(BaseTimeSeriesModel):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.params = None
        self.bic = np.inf

    def _init_params(self, x, eps=1e-6, n_init=10):
        x = np.asarray(x).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init=n_init, random_state=self.random_state)
        labels = kmeans.fit_predict(x)

        A = np.array([[0.9, 0.1],[0.2, 0.8]])
        means = np.zeros(2)
        vars = np.zeros(2)
        for k in range(2):
            xk = x[labels == k].ravel()
            if len(xk) > 1:
                vars[k] = xk.var(ddof=1) + eps
                means[k] = xk.mean()
            elif len(xk) == 1:
                vars[k] = eps
                means[k] = xk.mean()
            else:
                vars[k] = eps
                means[k] = 0.0

        if vars[0] > vars[1]:
            vars = vars[::-1]
            means = means[::-1]
            A = A[::-1, ::-1]

        return [A, means, vars]
    
    def llh(self, data):
        x = np.asarray(data, dtype=float).ravel()
        P, means, vars = self.params

        p12 = P[0, 1]
        p21 = P[1, 0]
        pi = np.array([p21 / (p21 + p12), p12 / (p21 + p12)])

        logA = np.log(P)
        logpi = np.log(pi)

        logB = -0.5 * np.log(2 * np.pi * vars[None, :]) - (x[:, None] - means[None, :])**2 / (2 * vars[None, :])

        log_alpha = np.zeros((len(data), len(P)))
        log_alpha[0] = logpi + logB[0]

        for t in range(1, len(data)):
            log_alpha[t] = logB[t] + logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)

        return logsumexp(log_alpha[-1])
    
    def fit(self, x, init_p=None):
        params = self._init_params(x) if init_p is None else init_p
        A, mu, var = params
        sol = EM(x, A, mu, var)
        self.params = [sol[0], sol[1], sol[2]]
        self.bic = -2 * sol[3] + 6*np.log(len(x))
        return sol
    
    def fit_n(self, x, n_init=10, pbar=None):
        scores, params, sols = [], [], []
        for _ in range(n_init):
            self.random_state = np.random.randint(0, 1000000)
            A, mu, var = self._init_params(x)
            sol = EM(x, A, mu, var)
            scores.append(sol[3])
            params.append([sol[0], sol[1], sol[2]])
            sols.append(sol)

            if pbar is not None: pbar.update(1)

        idx = np.argmax(scores)
        self.params = params[idx]
        self.bic = -2 * scores[idx] + 5*np.log(len(x))
        return sols[idx]
    
    def cdf(self, x, history=None):
        A, mu, var = self.params[0], self.params[1], self.params[2]
    
        logA = np.log(A + 1e-10)
        std_dev = np.sqrt(var)

        # Прогрев на выборке history
        p12, p21 = A[0, 1], A[1, 0]
        pi = np.array([p21 / (p21 + p12 + 1e-10), p12 / (p21 + p12 + 1e-10)])
        logpi = np.log(pi + 1e-10)
        
        logB_hist_0 = norm.logpdf(history[0], loc=mu, scale=std_dev)
        log_alpha = logpi + logB_hist_0
        
        for x_h in history[1:]:
            logB = norm.logpdf(x_h, loc=mu, scale=std_dev)
            log_alpha_new = np.zeros(2)
            log_alpha_new[0] = logB[0] + logsumexp(log_alpha + logA[:, 0])
            log_alpha_new[1] = logB[1] + logsumexp(log_alpha + logA[:, 1])
            log_alpha = log_alpha_new

        # Прогноз и вычисление CDF для test
        cdf_results = np.zeros(len(x))
        
        for t, x in enumerate(x):
            filt_probs = np.exp(log_alpha - logsumexp(log_alpha))
            
            pred_probs = filt_probs @ A 
            
            cdf_0 = norm.cdf(x, loc=mu[0], scale=std_dev[0])
            cdf_1 = norm.cdf(x, loc=mu[1], scale=std_dev[1])
            cdf_results[t] = pred_probs[0] * cdf_0 + pred_probs[1] * cdf_1
            
            logB = norm.logpdf(x, loc=mu, scale=std_dev)
            log_alpha_new = np.zeros(2)
            log_alpha_new[0] = logB[0] + logsumexp(log_alpha + logA[:, 0])
            log_alpha_new[1] = logB[1] + logsumexp(log_alpha + logA[:, 1])
            log_alpha = log_alpha_new
            
        return cdf_results
    
    def params_get(self):
        return self.params