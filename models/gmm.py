import numpy as np
from numba import njit
from scipy.stats import norm
from scipy.special import logsumexp
from utils import logsumexp2
from sklearn.cluster import KMeans
from models.base import BaseTimeSeriesModel

@njit
def EM(x, pi, mu, var, max_iter=10000, tol=1e-6, eps=1e-6):
    x = np.asarray(x)
    n = len(x)

    #var_x
    mean_x = 0.0
    for t in range(n):
        mean_x += x[t] / n

    var_x = 0.0
    for t in range(n):
        var_x += (x[t] - mean_x) * (x[t] - mean_x) / (n - 1)

    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    
    prev_ll = -np.inf

    log_prob = np.zeros((n, 2), dtype=np.float64)

    gamma = np.zeros((n, 2), dtype=np.float64)

    for iter in range(max_iter):

        # ===== E-step =====
        ll = 0.0

        for i in range(n):
            log_prob[i, 0] = np.log(pi[0]) - 0.5 * np.log(2 * np.pi * var[0]) - 0.5 * (x[i] - mu[0]) ** 2 / var[0]
            log_prob[i, 1] = np.log(pi[1]) - 0.5 * np.log(2 * np.pi * var[1]) - 0.5 * (x[i] - mu[1]) ** 2 / var[1]

            lse = logsumexp2(log_prob[i, 0], log_prob[i, 1])
            ll += lse

            gamma[i, 0] = np.exp(log_prob[i, 0] - lse)
            gamma[i, 1] = np.exp(log_prob[i, 1] - lse)

        # ===== M-step =====
        Nk = np.full(2, fill_value=eps, dtype=np.float64)
        for i in range(n):
            Nk[0] += gamma[i, 0]
            Nk[1] += gamma[i, 1]

        pi[0] = Nk[0] / n
        pi[1] = Nk[1] / n

        pi[0] = eps if pi[0] < eps else pi[0]
        pi[1] = eps if pi[1] < eps else pi[1]

        pisum = pi[0] + pi[1]
        pi[0] /= pisum
        pi[1] /= pisum

        mu = np.zeros(2, dtype=np.float64)
        for i in range(n):
            mu[0] += gamma[i, 0] * x[i]
            mu[1] += gamma[i, 1] * x[i]
        
        mu[0] /= Nk[0]
        mu[1] /= Nk[1]

        var = np.zeros(2, dtype=np.float64)
        for i in range(n):
            d = x[i] - mu[0]
            var[0] += gamma[i, 0] * d * d

            d = x[i] - mu[1]
            var[1] += gamma[i, 1] * d * d

        var[0] /= Nk[0]
        var[1] /= Nk[1]

        if var[0] > var[1]:
            var[0], var[1] = var[1], var[0]
            mu[0], mu[1] = mu[1], mu[0]
            pi[0], pi[1] = pi[1], pi[0]

        # clipping var значений для строгого разделения волатильностей на низкую и высокую
        if var[0] < var_x/9:
            var[0] = var_x/9
        if var[0] > var_x:
            var[0] = var_x
        
        if var[1] < var_x:
            var[1] = var_x
        if var[1] > 9*var_x:
            var[1] = 9*var_x

        diff = np.abs(ll - prev_ll)

        if diff < tol:
            break

        prev_ll = ll

    return (pi, mu, var, ll, diff, iter, gamma)

class GMM_2(BaseTimeSeriesModel):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.params = None
        self.bic = np.inf

    def _init_params(self, x, eps = 1e-6, n_init=10):
        x = np.asarray(x).reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, n_init=n_init, random_state=self.random_state)
        labels = kmeans.fit_predict(x)

        weights = np.empty(2, dtype=np.float64)
        means = np.empty(2, dtype=np.float64)
        vars = np.empty(2, dtype=np.float64)

        for k in range(2):
            xk = x[labels == k].ravel()

            weights[k] = len(xk) / len(x)

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
            weights = weights[::-1]

        return weights, means, vars
    
    def llh(self, data):
        #Расчёт лог-правдоподобия
        x = np.asarray(data)

        weights, means, vars = self.params
        log_weights = np.log(weights)

        x = x[:, None]

        log_densities = (
            -0.5 * np.log(2 * np.pi * vars)
            - 0.5 * (x - means) ** 2 / vars
        )

        log_component_probs = log_weights + log_densities

        return np.sum(logsumexp(log_component_probs, axis=1))
    
    def fit(self, x, init_p=None):
        if init_p is None:
            params = self._init_params(x, n_init=10)
        else:
            params = init_p
        pi, mu, var = params

        #Макcимизация лог-правдоподобия модели
        sol = EM(x, pi, mu, var)
        
        self.params = np.array([sol[0], sol[1], sol[2]])
        self.bic = -2 * sol[3] + 5*np.log(len(x))

        return sol
    
    def fit_n(self, x, n_init=10, pbar=None):
        scores = []
        params = []
        sols = []

        for _ in range(n_init):
            self.random_state = np.random.randint(0, 1000000)
            pi, mu, var = self._init_params(x)

            sol = EM(x, pi, mu, var)

            scores.append(sol[3])
            param = np.array([sol[0], sol[1], sol[2]])
            params.append(param)
            sols.append(sol)

            if pbar is not None: pbar.update(1)

        scores = np.asarray(scores)

        self.params = params[np.argmax(scores)]
        self.bic = -2 * np.max(scores) + 5*np.log(len(x))

        return sols[np.argmax(scores)]
    
    def cdf(self, x, history=None):
        weights, means, vars = self.params

        x = np.asarray(x)

        probs_k = norm.cdf(
            x[:, None],
            loc=means,
            scale=np.sqrt(vars)
        )

        return probs_k @ weights
    
    def params_get(self):
        return self.params