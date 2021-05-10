# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:08:37 2021

@author: samya
"""
import os
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import RSquare
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns

US_DATA = r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\Project Data\US States Data.csv'
EUROPE_DATA = r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\Project Data\europe.csv'

def predict_infections_rsquare(model, test_features, test_labels):
    # Predict values of test data and compute R-Squared coefficient
    predicted = model.predict(test_features)
    predicted = predicted.astype(int)

    # Measure the RSquare coefficient
    metric = RSquare()
    metric.update_state(test_labels, predicted)
    result = metric.result()

    # Return result
    return result.numpy()

def usa_features():
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def europe_features():
    return ['population (discrete data)', 'tests     (discrete data)', 'Gini      (discrete data)',
            '%urban pop.  (continuous data)', 'Actual cases']


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
    plt.ylim([0, .25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Actual Cases]')
    plt.legend()
    plt.grid(True)
    plt.show()


def snsregressionplot(x, y, title, xlabel, ylabel):
    fig = plt.figure()
    ax = sns.regplot(x=x, y=y.astype(float), ci=None, color="b")
    ax.title.set_text(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def remove_commas(data):
    # print(type(data))
    # print(data.shape)
    columns = data.columns
    for column in data[columns]:
        for index, value in enumerate(data[column].values):
            if isinstance(value, str):
                # print(value)
                data[column].iloc[index] = value.replace(',', '')
    return data


def split_data(data):
    # Train - Test split
    train_dataset = data.sample(frac=0.6, random_state=0)
    test_dataset = data.drop(train_dataset.index)

    # Look at ranges of features
    # print(train_dataset.describe().transpose())

    # Split features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(4)
    test_labels = test_features.pop(4)

    return train_features, train_labels, test_features, test_labels


def preprocess_data(data, features):
    # Select only certain features in the preprocessing pipeline
    data = data[features]

    # Remove potential commas present in the .csv
    data = remove_commas(data)

    # One-hot encode categorical variables (might be useful if we decide to use categorical variables)
    # data = pd.get_dummies(data)

    # Normalize data
    x = data.values  # returns a numpy array

    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)

    # Clean data by dropping NA rows
    return data.dropna()


def feature_plot(X, Y):
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(np.array(X)[:, 0], Y)
    ax1.title.set_text("Population vs Actual cases")
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(np.array(X)[:, 1], Y)
    ax2.title.set_text("Tests vs Actual cases")
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(np.array(X)[:, 2], Y)
    ax3.title.set_text("Gini vs Actual cases")
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(np.array(X)[:, 3], Y)
    ax4.title.set_text("% Urban population vs Actual cases")
    plt.show()


def MultiLinearReg_train_x_test_y(train, test):
    # If we are training and testing on the same model, we need to split the data into training and testing samples
    train_test_split = False
    if train == test:
        train_test_split = True

    # Helper dict object
    data_dict = {US_DATA: usa_features(), EUROPE_DATA: europe_features()}

    # Load in training data
    train_data = load_data('Project Data', train)

    # Load in test data
    test_data = load_data('Project Data', test)

    if not train_test_split:
        # Select some features to train on
        train_features = data_dict[train]

        # Preprocess train data by cleaning and normalizing
        train_features = preprocess_data(train_data, train_features)
        train_labels = train_features.pop(4)

        # Select some testing features
        test_features = data_dict[test]

        # Select features and clean data
        test_features = preprocess_data(test_data, test_features)
        test_labels = test_features.pop(4)
    else:
        train_features = data_dict[train]
        train_data = preprocess_data(train_data, train_features)
        train_features, train_labels, test_features, test_labels = split_data(train_data)

    regr = linear_model.LinearRegression().fit(train_features, train_labels)

    print('Regression Intercept: \n', regr.intercept_)
    print('Regression Coefficients: \n', regr.coef_)

    # SM model creation
    model = sm.OLS(train_labels, train_features).fit()
    print_model = model.summary()
    print(print_model)

    predictions1 = regr.predict(train_features)  # SKlearn model predictions

    predictions2 = model.predict(train_features)  # OLS model predictions

    #SNS plots
    snsregressionplot(predictions1, train_labels, "SKlearn prediction results vs Actual cases (USA)",
                      "SKLearn prediction results", "Actual Cases (USA)")

    snsregressionplot(predictions2, train_labels, "OLS prediction results vs Actual cases (USA)",
                      "OLS prediction results", "Actual Cases (USA)")

    # Get the RSquare
    rsquare_result = predict_infections_rsquare(model, test_features, test_labels)
    print(f'RSquare: {rsquare_result}')


    EUpredictions1 = regr.predict(test_features)

    EUpredictions2 = model.predict(test_features)

    snsregressionplot(EUpredictions1, test_labels, "SKLearn prediction results vs Actual cases (EU)",
                      "SKLearn prediction results", "Actual Cases (EU)")
    snsregressionplot(EUpredictions2, test_labels, "OLS prediction results vs Actual cases (EU)", "OLS prediction results",
                      "Actual Cases (EU)")


if __name__ == "__main__":
    MultiLinearReg_train_x_test_y(US_DATA, EUROPE_DATA)
