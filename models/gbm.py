import numpy as np
import scipy.stats as scs
from models.base import BaseTimeSeriesModel

class GBM(BaseTimeSeriesModel):
    def __init__(self, type='normal'):
        self.params = None
        self.bic = np.inf
        self.type = type

    def _init_params(self, x):
        self.fit(x)
        return self.params
    
    def llh(self, data):
        #Расчёт лог-правдоподобия
        x = np.asarray(data)

        if self.type == 'normal':
            den = scs.norm.pdf(x, *self.params)
        elif self.type == 'student':
            den = scs.t.pdf(x, *self.params)
        elif self.type == 'skew':
            den = scs.skewnorm.pdf(x, *self.params)
        elif self.type == 'laplace':
            den = scs.laplace.pdf(x, *self.params)
        elif self.type == 'gennorm':
            den = scs.gennorm.pdf(x, *self.params)
        log_densities = np.log(den)

        return np.sum(log_densities)
    
    def fit(self, x, init_p=None):
        if self.type == 'normal':
            params = scs.norm.fit(x)
        elif self.type == 'student':
            params = scs.t.fit(x)
        elif self.type == 'skew':
            params = scs.skewnorm.fit(x)
        elif self.type == 'laplace':
            params = scs.laplace.fit(x)
        elif self.type == 'gennorm':
            params = scs.gennorm.fit(x)

        self.params = params

        llh = self.llh(x)
        self.bic = -2 * llh + len(self.params)*np.log(len(x))

        return llh
    
    def fit_n(self, x, n_init = 10, pbar=None):
        return self.fit(x)
    
    def cdf(self, x, history=None):
        x = np.asarray(x)

        if self.type == 'normal':
            probs_k = scs.norm.cdf(x, *self.params)
        elif self.type == 'student':
            probs_k = scs.t.cdf(x, *self.params)
        elif self.type == 'skew':
            probs_k = scs.skewnorm.cdf(x, *self.params)
        elif self.type == 'laplace':
            probs_k = scs.laplace.cdf(x, *self.params)
        elif self.type == 'gennorm':
            probs_k = scs.gennorm.cdf(x, *self.params)

        return probs_k
    
    def params_get(self):
        return self.params