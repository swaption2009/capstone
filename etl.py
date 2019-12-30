import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


def preprocess_sensor_data(file):
    """
    This function is used to process sensor csv file
    and load it to Panda DataFrame

    input: txt file
    output: Panda DataFrame
    """
    # Load txt file to Panda DataFrame
    sensor_data = pd.read_csv('data/data.txt', sep=" ", header=None)
    sensor_data.columns = ['date',
                           'time',
                           'epoch',
                           'mote_id',
                           'temperature',
                           'humidity',
                           'light',
                           'voltage']

    # Format datetime type
    sensor_data['datetime'] = sensor_data['date'] + ' ' + sensor_data['time']
    sensor_data['datetime'] = pd.to_datetime(sensor_data['datetime'])
    sensor_data['datetime'] = sensor_data['datetime'].values.astype('<M8[s]')

    # Remove duplicates and null
    sensor_data.drop_duplicates(inplace=True)
    sensor_data.dropna(inplace=True)



    print(f'Sensor data preprocessing is completed')
    return weather_df


def preprocess_weather_data(file):
    """
    This function is used to process weather json file,
    parse nested json data, and load it to Panda DataFrame

    input: json file
    output: Panda DataFrame
    """
    # Load json file
    weather_data = pd.read_json('data/weather.json')

    # Select data and load it to Panda DataFrame
    columns = ['datetime', 'external temp']
    temp_list = []
    for x in weather_data.loc[0].forecast[0]['list']:
        temp_list.append({'datetime': x['dt'], 'external temp': x['main']['temp']})
    weather_df = pd.DataFrame(temp_list)

    # Convert datetime to Panda format
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df['datetime'] = weather_df['datetime'].values.astype('<M8[s]')

    print(f'Weather data preprocessing is completed')
    return weather_df


def main():
    WEATHER_DATA = 'data/weather.json'
    SENSOR_DATA = 'data/data.txt'
    SENSOR_LOC_DATA = ''

    preprocess_weather_data(WEATHER_DATA)
    preprocess_sensor_data(SENSOR_DATA)
    preprocess_sensor_loc_data(SENSOR_LOC_DATA)


if __name__ == "__main__":
    main()