import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras import Sequential


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
        layers.Dense(64, activation='relu', input_shape=data.shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


if __name__ == '__main__':
    # Load data
    data = load_data('Project Data', 'US States Data.csv')
    for col in data.columns:
        print(col)
    data = data[['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
                 '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']]

    # One-hot encode categorical variables
    # data = pd.get_dummies(data)

    # Normalize data
    x = data.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)

    # Clean data by dropping NA rows
    data = data.dropna()

    # Train - Test split
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset = data.drop(train_dataset.index)

    # Look at ranges of features
    print(train_dataset.describe().transpose())

    # Split features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(4)
    test_labels = test_features.pop(4)

    print(train_features.shape)

    # Define model
    model = build_and_compile_model(train_features)
    model.summary()

    # Train model on data
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=1, epochs=100)

    # Plot loss during training history
    plot_loss(history)

    # Evaluate model on test data
    dnn_results = model.evaluate(test_features, test_labels, verbose=0)
    print(f'DNN Validation Loss: {dnn_results}')
