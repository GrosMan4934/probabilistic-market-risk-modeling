import numpy as np
import scipy as sp
from data_loader import train_test_generator

def main_evaluation(model, log_profit, pbar=None):
    I = []
    alpha = 0.01
    params = None

    for train, test in train_test_generator(log_profit):
        # Оптимизация параметров для переданной модели
        model.fit(train, params)
        params = model.params
        
        # Обнаружение исключений
        probs = model.cdf(test, history=train)
        I_part = np.asarray(probs < alpha, dtype=int)
        I.append(I_part)

        if pbar is not None: pbar.update(1)

    return I

def kupiec_test(I, alpha=0.01):
    I = np.asarray(I, dtype=int)
    n = len(I)
    n1 = sum(I)
    n0 = n - n1

    phat = sum(I) / len(I)
    eps = 1e-12
    phat = min(max(phat, eps), 1 - eps)

    lr_uc = -2.0 * (n0 * np.log((1 - alpha) / (1 - phat)) + n1 * np.log(alpha / phat))
    p_value = 1.0 - sp.stats.chi2.cdf(lr_uc, df=1)
    return p_value

def kristofferson_test(I):
    I = np.asarray(I, dtype=int)
    n = len(I)
    t00 = sum((I[:-1] == 0) & (I[1:] == 0))
    t01 = sum((I[:-1] == 0) & (I[1:] == 1))
    t10 = sum((I[:-1] == 1) & (I[1:] == 0))
    t11 = sum((I[:-1] == 1) & (I[1:] == 1))

    ind = n * (abs(t00 * t11 - t01 * t10) - n / 2)**2 / ((t00 + t01) * (t00 + t10) * (t11 + t01) * (t11 + t10) + 1e-12)
    p_value = 1.0 - sp.stats.chi2.cdf(ind, df=1)
    return p_value

def testing(I):
    kupiec_reject = 0
    kristofferson_reject = 0
    threshold_p_value = 0.05

    kupiec_p_mean = 0
    kristofferson_p_mean = 0

    for period in I:
        kupiec_p_value = kupiec_test(period)
        kristofferson_p_value = kristofferson_test(period)

        kupiec_p_mean += kupiec_p_value
        kristofferson_p_mean += kristofferson_p_value

        if kupiec_p_value <= threshold_p_value: kupiec_reject += 1
        if kristofferson_p_value <= threshold_p_value: kristofferson_reject += 1

    N = len(I)
    print("Total periods:", N)
    print(f"Kupiec test: {N - kupiec_reject} [{round(100 - 100*kupiec_reject/N, 2)}%] | p-val = {round(kupiec_p_mean/N, 3)}")
    print(f"Kristofferson test: {N - kristofferson_reject} [{round(100 - 100*kristofferson_reject/N, 2)}%] | p-val = {round(kristofferson_p_mean/N, 3)}")