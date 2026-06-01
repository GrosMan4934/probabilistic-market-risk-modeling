import pandas as pd
import numpy as np
from pathlib import Path

def get_data(ticker, data_dir="data"):
    # Выгрузка исторических данных об акции
    path = Path(f"{data_dir}/{ticker}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Расчёт лог-доходностей
    close_prices = np.asarray(df['Close'], dtype=float)
    log_profit = np.log(close_prices[1:] / close_prices[:-1])
    return log_profit[-252*11-1:]

def train_test_generator(log_profits, train_len=189, test_len=63):
    # Генератор последовательных обучающих и тестовых выборок 
    # размерами train_len и test_len соответственно
    i = 0
    while i + train_len + test_len < len(log_profits):
        train = log_profits[i:i+train_len]
        test = log_profits[i+train_len:i+train_len+test_len]
        i += (train_len + test_len) // 1
        yield train, test