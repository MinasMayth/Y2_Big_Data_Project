import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow_addons.metrics import RSquare


def load_data(folder, filename):
    """
    Loads a csv file from the 'Project Data' directory and returns it as a pandas.Dataframe object.

    :param folder: from which folder to load data
    :param filename: .csv file to load
    :return: pandas.Dataframe of corresponding .csv
    """
    csv_data = pd.read_csv(os.path.join('..', folder, filename))
    return csv_data


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Actual Cases]')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_and_compile_model(data):
    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=data.shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def preprocess_data(data, features):
    for col in data.columns:
        print(col)
    data = data[features]

    # One-hot encode categorical variables
    # data = pd.get_dummies(data)

    # Normalize data
    x = data.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)

    # Clean data by dropping NA rows
    return data.dropna()


def split_data(data):
    # Train - Test split
    train_dataset = data.sample(frac=0.6, random_state=0)
    test_dataset = data.drop(train_dataset.index)

    # Look at ranges of features
    print(train_dataset.describe().transpose())

    # Split features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(4)
    test_labels = test_features.pop(4)

    return train_features, train_labels, test_features, test_labels


def train_model(model, train_features, train_labels):
    # Train model on data
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=1, epochs=100)

    return model, history


def plot_predictions(predictions, test_labels):
    # Plot predictions

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values [Infections]')
    plt.ylabel('Predictions [Infections]')
    lims = [0, 1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


def predict_infections_rsquare(model, test_features, test_labels):
    # Predict values of test data and compute R-Squared coefficient
    predicted = model.predict(test_features)
    predicted = predicted.flatten()

    # Measure the RSquare coefficient
    metric = RSquare()
    metric.update_state(test_labels, predicted)
    result = metric.result()

    # Return result
    return result.numpy()


def train_usa_test_usa():
    # Load data
    data = load_data('Project Data', 'US States Data.csv')

    # Select some features to train and test on
    features = ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
                 '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']

    # Select features and clean data
    data = preprocess_data(data, features)

    # Split data into train and test samples
    train_features, train_labels, test_features, test_labels = split_data(data)

    # Define model
    model = build_and_compile_model(train_features)
    model.summary()

    # Train model and retrieve performance of model during training
    model, history = train_model(model, train_features, train_labels)

    # Plot loss during training history
    plot_loss(history)

    # Evaluate model on test data
    dnn_results_usa = model.evaluate(test_features, test_labels, verbose=0)
    print(f'DNN Validation Loss: {dnn_results_usa}')

    # Get the RSquare
    dnn_results_usa_rsquare = predict_infections_rsquare(model, test_features, test_labels)
    print(f'RSquare: {dnn_results_usa_rsquare}')

    # Get predictions of model on test data
    test_predictions = model.predict(test_features).flatten()

    # Plot model predictions against actual values
    plot_predictions(test_predictions, test_labels)


def train_usa_test_europe():
    # Load data
    data = load_data('Project Data', 'europe.csv')

    # Select some features to train and test on
    features = ['population (discrete data)', 'tests (discrete data)', 'Gini (discrete data)',
                '%urban pop. (continuous data)', 'actual cases (measured) (discrete data)']

    # Select features and clean data
    data = preprocess_data(data, features)

    # Split data into train and test samples
    train_features, train_labels, test_features, test_labels = split_data(data)

    # Define model
    model = build_and_compile_model(train_features)
    model.summary()

    # Train model and retrieve performance of model during training
    model, history = train_model(model, train_features, train_labels)

    # Plot loss during training history
    plot_loss(history)

    # Evaluate model on test data
    dnn_results_usa = model.evaluate(test_features, test_labels, verbose=0)
    print(f'DNN Validation Loss: {dnn_results_usa}')

    # Get the RSquare
    dnn_results_usa_rsquare = predict_infections_rsquare(model, test_features, test_labels)
    print(f'RSquare: {dnn_results_usa_rsquare}')

    # Get predictions of model on test data
    test_predictions = model.predict(test_features).flatten()

    # Plot model predictions against actual values
    plot_predictions(test_predictions, test_labels)


if __name__ == '__main__':
    # Pipeline that trains a neural network model on USA data and tests on USA data
    train_usa_test_europe()

