import time
import numpy as np
from tqdm.auto import tqdm
from data_loader import get_data, train_test_generator
from utils import flatten

def _print_stats(llh_train, llh_test, params_stat):
    """Вспомогательная функция для вывода статистики, чтобы избежать дублирования кода."""
    print("LLH Train (min, mean, max):", np.min(llh_train), np.mean(llh_train), np.max(llh_train))
    print("LLH Test (min, mean, max):", np.min(llh_test), np.mean(llh_test), np.max(llh_test))
    print()
    print("mean:", np.mean(params_stat, axis=0))
    print("std:", np.std(params_stat, ddof=1, axis=0))
    print("min:", np.min(params_stat, axis=0))
    print("max:", np.max(params_stat, axis=0))
    print("-" * 50)

def debug_single_run(model, ticker='SBER'):
    """
    Одиночный прогон модели. 
    Необходим для оценки работоспособности на известных данных и 
    'прогрева' JIT-компилятора Numba перед массовыми тестами.
    """
    print(f"=== Debug: Single Run / Warmup [{ticker}] ===")
    
    # Загружаем данные для конкретного тикера
    log_profit = get_data(ticker)
    
    with tqdm() as pbar:
        # Берем только самую первую пару обучающей и тестовой выборок
        train, test = next(train_test_generator(log_profit))
        print(np.mean(train))

        st_f = time.perf_counter_ns()
        sol = model.fit(train)
        en_f = time.perf_counter_ns()
        
        # Замеряем время выполнения (первый раз включит JIT-компиляцию)
        st_l = time.perf_counter_ns()
        res = model.llh(train)
        en_l = time.perf_counter_ns()
        
        pbar.update(1)
    
    # Вывод результатов в формате из вашего блокнота
    print(res)
    print(res / len(train))
    print(en_f - st_f)
    print(en_l - st_l)
    print("-" * 50)

    return sol


def debug_init_params(stocks, model):
    """Отладочный цикл 1: Оценка работы только функции инициализации (без EM)."""
    print("=== Debug: _init_params ===")
    llh_train, llh_test, params_stat = [], [], []
    
    with tqdm() as pbar:
        for st in stocks:
            tqdm.write(f"Stock: {st}")
            log_profit = get_data(st)
            for train, test in train_test_generator(log_profit):
                params = model._init_params(train)
                model.params = params
                
                params_stat.append(list(flatten(model.params_get())))
                llh_train.append(model.llh(train) / len(train))
                llh_test.append(model.llh(test) / len(test))
                
                pbar.update(1)
            
    _print_stats(llh_train, llh_test, params_stat)


def debug_standard_fit(stocks, model):
    """Отладочный цикл 2: Обычное обучение model.fit() с замером времени."""
    print("=== Debug: model.fit() [Standard] ===")
    llh_train, llh_test, params_stat, times = [], [], [], []
    
    with tqdm() as pbar:
        for st in stocks:
            tqdm.write(f"Stock: {st}")
            log_profit = get_data(st)
            for train, test in train_test_generator(log_profit):
                start_time = time.perf_counter_ns()
                model.fit(train)
                end_time = time.perf_counter_ns()
                
                params_stat.append(list(flatten(model.params_get())))
                llh_train.append(model.llh(train) / len(train))
                llh_test.append(model.llh(test) / len(test))
                times.append(end_time - start_time)
                
                pbar.update(1)
            
    print("time mean (ns):", np.mean(times, axis=0))
    _print_stats(llh_train, llh_test, params_stat)

    return llh_train, llh_test, params_stat


def debug_sequential_fit(stocks, model):
    """Отладочный цикл 3: Обучение с передачей параметров с предыдущего шага."""
    print("=== Debug: model.fit() [Sequential Init] ===")
    llh_train, llh_test, params_stat = [], [], []
    params = None
    
    with tqdm() as pbar:
        for st in stocks:
            tqdm.write(f"Stock: {st}")
            log_profit = get_data(st)
            for train, test in train_test_generator(log_profit):
                model.fit(train, params)
                params = model.params
                
                params_stat.append(list(flatten(model.params_get())))
                llh_train.append(model.llh(train) / len(train))
                llh_test.append(model.llh(test) / len(test))
                
                pbar.update(1)
            
    _print_stats(llh_train, llh_test, params_stat)


def debug_fit_n(stocks, model, n_init=10):
    """Отладочный цикл 4: Обучение с множественным стартом (fit_n)."""
    print(f"=== Debug: model.fit_n() [n_init={n_init}] ===")
    llh_train, llh_test, params_stat = [], [], []
    
    with tqdm() as pbar:
        for st in stocks:
            tqdm.write(f"Stock: {st}")
            log_profit = get_data(st)
            for train, test in train_test_generator(log_profit):
                model.fit_n(train, n_init=n_init, pbar=pbar)
                
                params_stat.append(list(flatten(model.params_get())))
                llh_train.append(model.llh(train) / len(train))
                llh_test.append(model.llh(test) / len(test))
            
    _print_stats(llh_train, llh_test, params_stat)