import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow_addons.metrics import RSquare
from tensorflow.keras.utils import plot_model

# Globals for data files
US_DATA = 'US States Data.csv'
EUROPE_DATA = 'europe.csv'


"""
Objectives:
1. Initial draft of paper.
2. Predict death rates.
3. Conduct K-Fold cross validation.
4. 
"""

def load_data(folder, filename):
    """
    Loads a csv file from the 'Project Data' directory and returns it as a pandas.Dataframe object.

    :param folder: from which folder to load data
    :param filename: .csv file to load
    :return: pandas.Dataframe of corresponding .csv
    """
    csv_data = pd.read_csv(os.path.join('..', folder, filename))
    return csv_data


def plot_loss(history, fold):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Actual Cases]')
    plt.title(f'Losses at fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'k_fold_results/train_fold_{fold}')


def build_and_compile_model(data):
    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=data.shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def remove_commas(data):
    print(type(data))
    print(data.shape)
    columns = data.columns
    for column in data[columns]:
        for index, value in enumerate(data[column].values):
            if isinstance(value, str):
                print(value)
                data[column].iloc[index] = value.replace(',', '')
    return data


def preprocess_data(data, features):
    # Select only certain features in the preprocessing pipeline
    data = data[features]

    # Remove potential commas present in the .csv
    data = remove_commas(data)

    # One-hot encode categorical variables (might be useful if we decide to use categorical variables)
    # data = pd.get_dummies(data)

    # Normalize data
    x = data.values  # returns a numpy array
    print(x.shape)

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
    print(train_features.shape)

    # Train model on data
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

    return model, history


def plot_predictions(predictions, test_labels, train, test, fold):
    # Plot predictions

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values [Infections]')
    plt.ylabel('Predictions [Infections]')
    lims = [0, 1]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title(f'FOLD: {fold} Trained on {train} predicting on {test}')
    _ = plt.plot(lims, lims)
    plt.savefig(f'k_fold_results/predictions_fold_{fold}')
    plt.close()


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


def usa_features_deaths():
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def europe_features_deaths():
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def usa_features():
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def europe_features():
    return ['population (discrete data)', 'tests     (discrete data)', 'Gini      (discrete data)',
            '%urban pop.  (continuous data)', 'Actual cases']


def train_x_test_y(train='US States Data.csv', test='europe.csv', folds=5, debug=False):
    """
    Trains a neural network model on the 'x' dataset and predicts infection rates on the 'y' dataset. The function
    plots the actual vs predicted values and computes the RSquare coefficient of the model.
    :param train: The dataset that the model will train on.
    :param test: The dataset that the model will test on.
    :return: None
    """

    # If we are training and testing on the same model, we need to split the data into training and testing samples
    same_data = False
    if train == test:
        same_data = True

    # Helper dict object
    data_dict = {US_DATA: usa_features(), EUROPE_DATA: europe_features()}

    # Load in training data
    train_data = load_data('Project Data', train)

    # Preprocess train data by cleaning and normalizing
    train_features = data_dict[train]
    train_features = preprocess_data(train_data, train_features)
    train_labels = train_features.pop(4)

    # Load in test data
    test_data = load_data('Project Data', test)

    # Preprocess test data by cleaning and normalizing
    test_features = data_dict[test]
    test_features = preprocess_data(test_data, test_features)
    test_labels = test_features.pop(4)

    # K-Fold initialization
    k_fold = KFold(n_splits=folds, shuffle=False)
    n_fold = 0
    r_squares = []
    for train_index, test_index in k_fold.split(train_features):
        n_fold += 1
        print(f'FOLD: {n_fold} TRAIN: {train_index} TEST: {test_index}')
        X_train, y_train = train_features.iloc[train_index], train_labels.iloc[train_index]

        # Define model
        model = build_and_compile_model(X_train)

        # Summary of the model
        if debug:
            model.summary()
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # Train model and retrieve performance of model during training
        model, history = train_model(model, X_train, y_train)

        # Plot loss during training history
        plot_loss(history, n_fold)

        if same_data:
            X_test = train_features.iloc[test_index]
            y_test = train_labels.iloc[test_index]
        else:
            X_test = test_features
            y_test = test_labels

        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'DNN Validation Loss: {loss}')

        # Get the RSquare
        rsquare_result = predict_infections_rsquare(model, X_test, y_test)
        print(f'RSquare at fold {n_fold}: {rsquare_result}')

        # Sum r-square for final statistic
        r_squares.append(rsquare_result)

        # Get predictions of model on test data
        test_predictions = model.predict(X_test).flatten()

        # Plot model predictions against actual values
        plot_predictions(test_predictions, y_test, train, test, n_fold)

    # Print average RSquare
    print(f'Average RSquare: {sum(r_squares) / folds}')

    # Save RSquare score for each fold and store in k-folds results folder
    with open('k_fold_results/rsquares.txt', "w") as each_fold:
        temp = [str(x) for x in r_squares]
        for index, result in enumerate(temp):
            each_fold.write(str(index) + ': ' + result + '\n')


if __name__ == '__main__':
    # Pipeline that trains a neural network model on `train` argument of `train_x_test_y` function tests on `test`
    # argument of `train_x_test_y` function
    train_x_test_y(US_DATA, EUROPE_DATA, folds=5)
