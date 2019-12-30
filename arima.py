import pandas as pd
import numpy as np
from numpy import log
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def process_data(file):
    """
    This function is used to prepare data for
    ARIMA time series analysis and training

    input: csv data
    output: DataFrame
    """
    sensor_data = pd.read_csv(file, sep=" ", header=None)
    sensor_data.columns = ['date',
                           'time',
                           'epoch',
                           'mote_id',
                           'temperature',
                           'humidity',
                           'light',
                           'voltage']

    # set datetime data to hourly
    sensor_data['datetime'] = sensor_data['date'] + ' ' + sensor_data['time']
    sensor_data['datetime'] = pd.to_datetime(sensor_data['datetime'])
    sensor_data['datetime'] = sensor_data['datetime'].values.astype('<M8[h]')

    # drop duplicates and null values
    sensor_data.sort_values(by=['mote_id', 'datetime'], inplace=True)
    sensor_data.dropna(inplace=True)
    sensor_data.drop_duplicates('datetime', inplace=True)

    print(f'sensor_data is uploaded.')
    return sensor_data

def train(data):
    """
    This function is used to train ARIMA model.

    input: sensor data
    output: ARIMA model
    """
    # feature selection
    mote1 = sensor_data[sensor_data['mote_id'] == 1.0 ]
    mote1 = mote1[:500]

    # inspect time series pattern
    result = adfuller(mote1['temperature'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # train model
    model = ARIMA(mote1.temperature, order=(1,1,2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    print(f'ARIMA model has been trained.')
    return model_fit


def predict(model, n_windows):
    """
    This function is used to predict time series

    input: ARIMA trained model and window period
    output: time series prediction
    """
    forecast = model.forecast(n_windows)
    print(f'Forecast: {forecast}')


def main():
    DATA_DIR = 'data/data.txt'

    sensor_data = process_data(DATA_DIR)
    arima_model_fit = train(sensor_data)
    predict(arima_model_fit, n_windows=50)


if __name__ == "__main__":
    main()