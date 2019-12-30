import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


def preprocess_weather_data(file):
    """
    This function is used to process weather json file,
    parse nested json data, and load it to Panda DataFrame

    input: json file
    output: Panda DataFrame
    """
    # Load json file
    weather_data = pd.read_json(file)

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


def preprocess_sensor_data(file):
    """
    This function is used to process sensor csv file
    and load it to Panda DataFrame

    input: txt file
    output: Panda DataFrame
    """
    # Load txt file to Panda DataFrame
    sensor_data = pd.read_csv(file, sep=" ", header=None)
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
    return sensor_data


def preprocess_sensor_loc_data(file):
    """
    This function is used to load sensor location data to DataFrame

    input: txt file
    output: Panda DataFrame
    """
    # Load json file
    sensor_locs = pd.read_csv(file, sep=" ", header=None)
    sensor_locs.columns = ['mote_id', 'x_coord', 'y_coord']
    sensor_locs['mote_id'] = sensor_locs['mote_id'].astype(float)

    print(f'Sensor location data preprocessing is completed')
    return sensor_locs

def combine_dataset(sensor_df, loc_df, weather_df):
    """
    This function is used to combine query dataset

    input: dataframes
    output: Panda DataFrame
    """
    # combine dataframes
    sensor_signals = sensor_df.merge(loc_df, how='left', left_on='mote_id', right_on='mote_id')
    sensor_signals.sort_values(['mote_id', 'datetime'], inplace=True)
    sensor_signals.dropna(inplace=True)

    # add time dimensions
    sensor_signals['year'] = sensor_signals['datetime'].apply(lambda x: x.year)
    sensor_signals['month'] = sensor_signals['datetime'].apply(lambda x: x.month)
    sensor_signals.drop(['date', 'time', 'epoch'], axis=1, inplace=True)

    print(f'Dataset has been combined')
    return sensor_signals


def write_to_parquet(df):
    """
    This function is used to save clean data in parquet format.

    input: DataFrame
    output: parquet files
    """
    table = pa.Table.from_pandas(sensor_signals)

    pq.write_to_dataset(
        table,
        root_path='output.parquet',
        partition_cols=['year', 'month', 'mote_id'],
    )

    print(f'Clean data has been saved to parquet files')


def main():
    WEATHER_DATA = 'data/weather.json'
    SENSOR_DATA = 'data/data.txt'
    SENSOR_LOC_DATA = ''

    weather_data = preprocess_weather_data(WEATHER_DATA)
    sensor_data = preprocess_sensor_data(SENSOR_DATA)
    sensor_loc_data = preprocess_sensor_loc_data(SENSOR_LOC_DATA)
    all_data = combine_dataset(sensor_data, sensor_loc_data, weather_data)
    write_to_parquet(all_data)

if __name__ == "__main__":
    main()