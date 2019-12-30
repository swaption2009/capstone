# DEND CAPSTONE PROJECT

## How to run this repo
1. git clone this [repo](https://github.com/swaption2009/capstone)
2. create and add `data` folder
  * Note: due to GitHub repo size limit, the dataset is not uploaded.
  * See links below to download datasets
3. run `python3 etl.py`
4. run `python3 ml.py`
5. run `python3 arima.py`

## Folder and Files

### Files
1. `etl.py`: preprocess raw data and output clean data as parquet file.
2. `ml.py`: preprocess data, train & evaluate Machine Learning model, and save final model.
3. `arima.py`: preprocess data, train and forecast time series data using ARIMA model.
4. `finalized_model.sav`: trained Machine Learning model using Logistics Regression algorithm.

### Folders
1. `Notebooks`: consists of the workbooks in Jupyter Notebook
2. `output.parquet`: clean database partitioned by `year`, `month`, `mote_id`.

## Project Write-up

PDF file: https://github.com/swaption2009/capstone/blob/master/%5BDEND%20Capstone%5D%20Project%20Write-up.pdf

## References and Links

1. [Intel Lab Dataset](http://db.csail.mit.edu/labdata/labdata.html)
2. [OpenWeather Dataset](https://openweathermap.org/history)