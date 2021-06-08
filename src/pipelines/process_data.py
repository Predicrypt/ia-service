from typing import Tuple
from get_data import get_all_data
from kfp.components import create_component_from_func, InputPath
import os


def process_data(input_path: InputPath(str)) -> Tuple:

    from sklearn import preprocessing
    import numpy as np
    from pandas.core.frame import DataFrame
    import pickle

    n = 14
    train_size = 0.8
    timesteps = 60

    def rma(x, n, y0):
        a = (n-1) / n
        ak = a**np.arange(len(x)-1, -1, -1)
        return np.r_[np.full(n, np.nan, dtype=np.float64), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

    dataDict = {
        "closeTime": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    }

    data = []

    with open(input_path, 'rb') as fp:
        data = pickle.load(fp)

    for kline in data:
        dataDict["closeTime"].append(kline[0])
        dataDict["open"].append(float(kline[1]))
        dataDict["high"].append(float(kline[2]))
        dataDict["low"].append(float(kline[3]))
        dataDict["close"].append(float(kline[4]))
        dataDict["volume"].append(float(kline[5]))

    df = DataFrame(dataDict)

    # Calculate SMA
    df["SMA_20"] = df.iloc[:, 4].rolling(window=20).mean()
    df["SMA_50"] = df.iloc[:, 4].rolling(window=50).mean()
    df["SMA_200"] = df.iloc[:, 4].rolling(window=200).mean()

    # Calculate RSI

    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[n+1:].to_numpy(), n,
                         np.nansum(df.gain.to_numpy()[:n+1])/n)
    df['avg_loss'] = rma(df.loss[n+1:].to_numpy(), n,
                         np.nansum(df.loss.to_numpy()[:n+1])/n)
    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi'] = 100 - (100 / (1 + df.rs))

    df.drop(columns=['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'])

    # MACD

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()

    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Train and Test data

    size = int(len(df['close']) * train_size)

    train_set = df.iloc[:size, :].values
    test_set = df.iloc[size:, :].values

    # Feature scaling

    scaler = preprocessing.StandardScaler().fit(train_set)

    # X and Y and reshape
    train_set_scaled = scaler.transform(train_set)
    test_set_scaled = scaler.transform(test_set)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(60, len(train_set_scaled)):
        x_train.append(train_set_scaled[i - 60:i, :])
        if i != len(train_set_scaled):
            y_train.append(train_set_scaled[i, 4])
        else:
            y_train.append(test_set_scaled[0, 4])

    for i in range(60, len(test_set_scaled) - 1):
        x_test.append(test_set_scaled[i - 60:i, :])
        y_test.append(test_set_scaled[i, 4])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = np.reshape(
        x_train, (x_train.shape[0], timesteps, x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], timesteps, x_test.shape[2]))

    return x_train, y_train, x_test, y_test


process_data_component = create_component_from_func(
    func=process_data,
    output_component_file=f'{os.getcwd()}/ia-service/src/pipelines/definitions/ProcessDataComponent.yaml',
    base_image='python:3.8',
    packages_to_install=['pandas==1.2.4', 'scikit-learn==0.24.2']
)

if __name__ == '__main__':
    process_data(get_all_data('ETHUSDT', '1h', 15000))
