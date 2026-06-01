from abc import ABC, abstractmethod
import numpy as np

class BaseTimeSeriesModel(ABC):
    """
    Абстрактный базовый класс для всех моделей временных рядов.
    Гарантирует, что любая новая модель будет иметь нужные методы.
    """
    @abstractmethod
    def fit(self, x: np.ndarray, init_p=None):
        pass

    @abstractmethod
    def fit_n(self, x: np.ndarray, n_init: int = 10, pbar=None):
        pass

    @abstractmethod
    def llh(self, data: np.ndarray) -> float:
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray, history: np.ndarray = None, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def params_get(self):
        pass