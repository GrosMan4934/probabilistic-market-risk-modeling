import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from numba import njit
from models.base import BaseTimeSeriesModel
from skopt import gp_minimize
from skopt.space import Real

@njit(fastmath=True)
def heston_llh(data, mu, kappa, omega, psi, eps_all, u_all, M, floor, dt):
    v = np.linspace(0.001, 0.3, M).astype(np.float64)
    logW = np.full(M, -np.log(M), dtype=np.float64)
    rho = -0.4

    loglike = 0.0
    log_2pi = np.log(2.0 * np.pi)
    
    n_steps = len(data)
    filtered_v = np.zeros(n_steps, dtype=np.float64)
    filtered_m = np.zeros(n_steps, dtype=np.float64)

    for t in range(n_steps):
        y = data[t]
        eps = eps_all[t]
        u_resample = u_all[t]

        v = np.maximum(v, floor)

        a = v + kappa * (omega - v) * dt
        v_new = np.maximum(a + psi * np.sqrt(v * dt) * eps, floor)

        m = mu * dt - 0.5 * v * dt + rho * np.sqrt(v * dt) * eps

        s_2 = np.maximum((1.0 - rho**2) * v * dt + 1e-6, floor)

        log_g = -0.5 * (log_2pi + np.log(s_2) + (y - m)**2 / s_2)
        
        logW_g = logW + log_g
        max_logW_g = np.max(logW_g)
        if np.isinf(max_logW_g):
            log_prob_y = max_logW_g
        else:
            log_prob_y = max_logW_g + np.log(np.sum(np.exp(logW_g - max_logW_g)))
        
        loglike += log_prob_y

        W_updated = np.exp(logW_g - max_logW_g)
        W_updated = W_updated / np.sum(W_updated)
        
        filtered_v[t] = np.sum(W_updated * v)
        filtered_m[t] = np.sum(W_updated * m)

        logW = logW_g - log_prob_y

        logW_2 = 2.0 * logW
        max_logW_2 = np.max(logW_2)
        if np.isinf(max_logW_2):
            logsumexp_2logW = max_logW_2
        else:
            logsumexp_2logW = max_logW_2 + np.log(np.sum(np.exp(logW_2 - max_logW_2)))
            
        effective_sample_size = np.exp(-logsumexp_2logW)

        if effective_sample_size < M / 2.0:
            max_logW = np.max(logW)
            W = np.exp(logW - max_logW)
            W = W / np.sum(W)

            positions = (u_resample + np.arange(M)) / M
            cumulative_sum = np.cumsum(W)
            cumulative_sum[-1] = 1.0

            idx = np.searchsorted(cumulative_sum, positions, side='right')
            v = v_new[idx].copy()
            logW[:] = -np.log(M)
        else:
            v = v_new.copy()

    return loglike, filtered_v, filtered_m

class Heston(BaseTimeSeriesModel):
    def __init__(self, n_particles=1000, random_state=42):
        self.random_state = random_state
        self.params = None
        self.bic = np.inf
        self.n_particles = n_particles
    
    def _fiting_llh(self, params, data, floor=1e-10, dt=1/252):
        rng = np.random.default_rng(self.random_state)
        M = self.n_particles

        params = np.array(params)
        mu, kappa, omega, psi = params

        feller_val = 2 * kappa * omega - psi**2
        
        if feller_val <= 0:
            # Штраф за нарушение условия Феллера
            penalty = 1e6 + abs(feller_val) * 1e5
            return penalty

        eps_all = rng.normal(size=(len(data), M))
        u_all = rng.random(size=len(data))
        data_arr = np.asarray(data, dtype=np.float64)

        loglike, _, _ = heston_llh(
            data_arr, mu, kappa, omega, psi, 
            eps_all, u_all, M, floor, dt
        )

        params_center = np.array([0.0, 3.0, 0.05, 0.4])
    
        # Регуляризация для предотвращения необоснованного завышения или занижения параметров.
        lmbda_params = np.array([0.1, 0.2, 10.0, 0.5])
        
        penalty = np.sum(lmbda_params * (params - params_center)**2)

        return - loglike + penalty
    
    def llh(self, data, floor=1e-10, dt=1/252):
        rng = np.random.default_rng(self.random_state)
        M = self.n_particles

        mu, kappa, omega, psi = self.params

        eps_all = rng.normal(size=(len(data), M))
        u_all = rng.random(size=len(data))
        data_arr = np.asarray(data, dtype=np.float64)

        loglike, _, _ = heston_llh(
            data_arr, mu, kappa, omega, psi,
            eps_all, u_all, M, floor, dt
        )

        return loglike

    def filter_states(self, data, floor=1e-10, dt=1/252):
        # Получение фильтрованной траектории (лог-доходностей и волатильностей)
        rng = np.random.default_rng(self.random_state)
        M = self.n_particles

        mu, kappa, omega, psi = self.params

        eps_all = rng.normal(size=(len(data), M))
        u_all = rng.random(size=len(data))
        data_arr = np.asarray(data, dtype=np.float64)

        _, filtered_v, filtered_m = heston_llh(
            data_arr, mu, kappa, omega, psi,
            eps_all, u_all, M, floor, dt
        )

        return filtered_v, filtered_m
    
    def fit(self, x, p_init=None):
        if p_init is None:
            x0 = None
        else:
            x0 = list(p_init)

        sol_bayess = self.fit_bayess(x, x0)

        self.params = sol_bayess.x
        self.bic = -2 * self.llh(x) + 5*np.log(len(x))

        return sol_bayess
    
    def fit_n(self, x, n_init=10, pbar=None):
        n_calls = 50 * n_init
        n_initial_params = 2**np.round(np.log(n_calls/3)/np.log(2))

        sol_bayess = self.fit_bayess(x, None, n_calls, n_initial_params)

        self.params = sol_bayess.x
        self.bic = -2 * self.llh(x) + 5*np.log(len(x))

        return sol_bayess

    def fit_bayess(self, x, x0, n_calls=100, n_initial_params=32, dt=1/252):
        dimensions = [
            Real(-1.0, 1.0, name='mu'),
            Real(0.01, 10, name='kappa'), 
            Real(0.001, 0.3, name='omega'),
            Real(0.01, 0.7, name='psi')
        ]
        
        def objective(params):
            return self._fiting_llh(params, x, dt=dt)
        
        sol = gp_minimize(
            func=objective,
            dimensions=dimensions,
            x0=x0,
            n_calls=n_calls,
            n_initial_points=n_initial_params if x0 is None else n_initial_params-1,
            acq_func='EI',
            random_state=self.random_state,
            initial_point_generator='sobol',
            n_jobs=-1
        )
        
        self.params = sol.x
        self.bic = -2 * (-sol.fun) + 5 * np.log(len(x))

        return sol
    
    def cdf(self, x, history=None, floor=1e-10, dt=1/252):
        rng = np.random.default_rng(self.random_state)
        M = self.n_particles
        mu, kappa, omeg, psi = self.params
        rho = -0.4
        
        log_2pi = np.log(2.0 * np.pi)

        # Прогрев на исторических данных
        v = np.linspace(0.001, 0.3, M).astype(np.float64)
        logW = np.full(M, -np.log(M), dtype=np.float64)

        if history is not None:
            for y in history:
                v = np.maximum(v, floor)
                a = v + kappa * (omeg - v) * dt
                eps = rng.normal(size=M)
                v_new = np.maximum(a + psi * np.sqrt(v * dt) * eps, floor)
                m = mu * dt - 0.5 * v * dt + rho * np.sqrt(v * dt) * eps
                s_2 = np.maximum((1 - rho**2) * v * dt, floor)

                log_g = -0.5 * (log_2pi + np.log(s_2) + (y - m)**2 / s_2)
                log_prob_y = logsumexp(logW + log_g)
                logW = logW + log_g - log_prob_y

                if np.exp(-logsumexp(2.0 * logW)) < M / 2:
                    W_resample = np.exp(logW - np.max(logW))
                    W_resample = W_resample / np.sum(W_resample)
                    positions = (rng.random() + np.arange(M)) / M
                    cumulative_sum = np.cumsum(W_resample)
                    cumulative_sum[-1] = 1.0
                    idx = np.searchsorted(cumulative_sum, positions, side='right')
                    v = v_new[idx]
                    logW[:] = -np.log(M)
                else:
                    v = v_new[:]

        # Расчёт cdf на тестовой выборке
        cdf_values = np.empty_like(x, dtype=float)

        for i, yt in enumerate(x):
            v = np.maximum(v, floor)

            # Прогноз делается на базе прошлого состояния v ДО генерации новых шоков eps.
            pred_m = mu * dt - 0.5 * v * dt
            pred_s2 = np.maximum(v * dt, floor)

            W = np.exp(logW - np.max(logW))
            W = W / np.sum(W)

            # Считаем CDF
            cdf_values[i] = np.sum(W * norm.cdf((yt - pred_m) / np.sqrt(pred_s2)))

            # Uенерируем шоки и обновляем v для следующего дня
            a = v + kappa * (omeg - v) * dt
            eps = rng.normal(size=M)
            v_new = np.maximum(a + psi * np.sqrt(v * dt) * eps, floor)

            # Условные m и s_2 для расчета правдоподобия
            m = mu * dt - 0.5 * v * dt + rho * np.sqrt(v * dt) * eps
            s_2 = np.maximum((1 - rho**2) * v * dt, floor)

            log_g = -0.5 * (log_2pi + np.log(s_2) + (yt - m)**2 / s_2)
            log_prob_y = logsumexp(logW + log_g)
            logW = logW + log_g - log_prob_y

            # resampling
            if np.exp(-logsumexp(2.0 * logW)) < M / 2:
                W_resample = np.exp(logW - np.max(logW))
                W_resample = W_resample / np.sum(W_resample)
                positions = (rng.random() + np.arange(M)) / M
                cumulative_sum = np.cumsum(W_resample)
                cumulative_sum[-1] = 1.0
                idx = np.searchsorted(cumulative_sum, positions, side='right')
                v = v_new[idx]
                logW[:] = -np.log(M)
            else:
                v = v_new[:]

        return cdf_values
    
    def params_get(self):
        mu, kappa, omeg, psi = self.params
        return mu, kappa, omeg, psi