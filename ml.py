import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


def read_parquet(file):
    """
    This function is used to load parquet files to DataFrame

    input: parquet files
    output: Panda DataFrame
    """
    sensor_signals = pd.read_parquet('output.parquet/')
    # limit training and test dataset to 10,000 rows
    print(f'Dataset is loaded!')
    return sensor_signals = sensor_signals[:10000]


def preprocess_data(df):
    """
    This function is used to prepare dataset for
    Machine Learning training

    input: DataFrame
    output: Machine Learning input dataset and labels
    """
    sensor_signals = df

    # add time dimensions
    sensor_signals['day'] = sensor_signals['datetime'].apply(lambda x: x.day)
    sensor_signals['hour'] = sensor_signals['datetime'].apply(lambda x: x.hour)
    sensor_signals['minute'] = sensor_signals['datetime'].apply(lambda x: x.minute)
    sensor_signals.drop(['datetime', 'x_coord', 'y_coord'], axis=1, inplace=True)

    # prepare input and label from dataset
    X = sensor_signals.drop('mote_id', axis=1)
    Y = sensor_signals['mote_id']

    # scale input parameters
    scaler = StandardScaler()
    X[['temperature', 'humidity', 'light', 'voltage']] = scaler.fit_transform(X[['temperature',
                                                                                 'humidity',
                                                                                 'light',
                                                                                 'voltage']])

    # split train-test dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=122)

    print(f'Dataset is ready!')
    return X_train, X_test, Y_train, Y_test


def train_and_save_model(X_train, X_test, Y_train, Y_test):
    """
    This function is used to train, evaluate, and save ML model

    input: training and test dataset and labels
    output: ML model
    """
    # train model using Logistic Regression CV algorithm
    model = LogisticRegressionCV(cv=10,
                                 verbose=1,
                                 n_jobs=-1,
                                 scoring='accuracy',
                                 solver='lbfgs',
                                 penalty='l2')
    model.fit(X_train, Y_train)

    # use the model to make predictions with the test data
    Y_pred = model.predict(X_test)
    # how did our model perform?
    count_misclassified = (Y_test != Y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))

    # save model to pickle file
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f'ML model has been saved!')


def main():
    DATA_DIR = 'output.parquet/'

    raw_data = read_parquet(DATA_DIR)
    X_train, X_test, Y_train, Y_test = preprocess_data(raw_data)
    train_and_save_model(X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    main()