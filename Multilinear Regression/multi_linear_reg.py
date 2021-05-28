# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:08:37 2021

@author: samya
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns

US_DATA = r'USAclean.csv'
EUROPE_DATA = r'EUclean.csv'
TEST_COUNTRIES = r'Test Data.csv'


def usa_features() -> list:
    """
    Features to be used from USA data
    :return: List of column titles of Pandas Dataframes
    """
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def europe_features() -> list:
    """
    Features to be used from EU data - This should be the same as the US data
    :return: List of column titles of Pandas Dataframes
    """
    return ['population (discrete data)', 'tests (discrete data)', 'Gini (discrete data)',
            '%urban pop. (continuous data)', 'Actual cases']


def hypothetical1_features() -> list:
    """
    Hypothetical Case 1 - tests is equal to 1.1 * population
    :return: List of column titles of Pandas Dataframes
    """
    return ['population (discrete data)', 'Pop*1.1', 'Gini (discrete data)',
            '%urban pop. (continuous data)', 'Actual cases']


def testcountry_features() -> list:
    """
    Features to be used from EU data - This should be the same as the US data
    :return: List of column titles of Pandas Dataframes
    """
    return ['Population', 'Number of tests', 'Gini Index',
            'Urban Population (%)', 'Measured number of infections']


def hypothetical2_features() -> list:
    """
    Hypothetical Case 1 - tests is equal to population
    :return: List of column titles of Pandas Dataframes
    """
    return ['Population', 'Pop*1.1', 'Gini Index',
            'Urban Population (%)', 'Measured number of infections']


def load_data(folder, filename):
    """
    Loads a csv file from the 'Project Data' directory and returns it as a pandas.Dataframe object.
    :param folder: from which folder to load data
    :param filename: .csv file to load
    :return: pandas.Dataframe of corresponding .csv
    """
    csv_data = pd.read_csv(os.path.join(os.getcwd(), "..", folder, filename))
    return csv_data


def snsregressionplot(x, y, title, xlabel, ylabel):
    """
    Regression plot using Seaborn library, with information on the graph being returned via
    the scipy stats library
    :param x: Array of x values
    :param y: Array of y values
    :param title: Plot title
    :param xlabel: X axis title
    :param ylabel: Y axis title
    :return: scipy stats linear regression information for x and y
    """
    fig = plt.figure()
    ax = sns.regplot(x=x, y=y.astype(float), ci=None, color="b")
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
             color='red')
    # ax.title.set_text(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return stats.linregress(x, y)


def remove_commas(data):
    """
    Remove commas from a pandas dataset
    :param data: Pandas dataframe
    :return: cleaned dataframe
    """
    columns = data.columns
    for column in data[columns]:
        for index, value in enumerate(data[column].values):
            if isinstance(value, str):
                data[column].iloc[index] = value.replace(',', '')
    return data


def split_data(data, fraction=0.6):
    """
    train_test split data
    :param data: pandas dataframe
    :param fraction: what fraction of data to split, default is 0.6 (60%)
    :param index_of_dependent_feature: index of the dependent feature, which will be assigned to train_labels and
    test_labels
    :return: train_features, train_labels, test_features, test_labels - data to be used for building model
    """
    # Train - Test split
    train_dataset = data.sample(frac=fraction, random_state=0)
    test_dataset = data.drop(train_dataset.index)

    # Look at ranges of features
    # print(train_dataset.describe().transpose())

    # Split features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(train_features.columns[-1])
    test_labels = test_features.pop(test_features.columns[-1])

    return train_features, train_labels, test_features, test_labels


def clean_and_convert_data(data, features, filename):
    """
    Minor utility function to download the cleaned version of a dataframe with the current features being used
    :param data: dataset to extract data from
    :param features: function that returns features to be extracted
    :param filename: filename to save to
    :return: excel file
    """
    # Select only certain features in the preprocessing pipeline
    data = data[features]

    # Remove potential commas present in the .csv
    data = remove_commas(data)

    return data.to_excel(filename + ".xlsx")


def preprocess_data(data, features):
    """
    Select features, clean data, normalize and scale data
    :param data: dataset to preprocess
    :param features: features to select
    :return: preprocessed dataset
    """
    # Select only certain features in the preprocessing pipeline
    data = data[features]

    # Remove potential commas present in the .csv
    data = remove_commas(data)

    # One-hot encode categorical variables (might be useful if we decide to use categorical variables)
    # data = pd.get_dummies(data)

    # Normalize data
    x = data.values  # returns a numpy array

    min_max_scaler = MinMaxScaler()
    global x_scaler
    x_scaler = min_max_scaler.fit(x)
    x_scaled = x_scaler.transform(x)
    data = pd.DataFrame(x_scaled)

    # Clean data by dropping NA rows
    return data.dropna()


def main_split_and_preprocess(train, test, train_features, test_features, set1=US_DATA, set2=EUROPE_DATA):
    """
    Main function to preprocess data and split into features and labels
    :param train: Data to train on
    :param test: Data to test on
    :param train_features: features used for training - should contain equivalent information to test_features
    :param test_features: features used for testing - should contain equivalent information to train_features
    :return: 
    """
    # If we are training and testing on the same model, we need to split the data into training and testing samples
    train_test_split = False
    if train == test:
        train_test_split = True

    # Helper dict object
    data_dict = {set1: train_features, set2: test_features}

    # Load in training data
    train_data = load_data('Project Data', train)

    # Load in test data
    test_data = load_data('Project Data', test)

    if not train_test_split:
        # Select some features to train on
        train_features = data_dict[train]

        # clean_and_convert_data(train_data, train_features, "USAtrain")

        # Preprocess train data by cleaning and normalizing
        train_features = preprocess_data(train_data, train_features)

        # Save the last column to train_labels, this should be the dependent feature
        train_labels = train_features.pop(train_features.columns[-1])

        # Select some testing features
        test_features = data_dict[test]

        # clean_and_convert_data(test_data, test_features, "EUtest")
        # Select features and clean data
        test_features = preprocess_data(test_data, test_features)

        test_labels = test_features.pop(test_features.columns[-1])
    else:
        train_features = data_dict[train]
        train_data = preprocess_data(train_data, train_features)
        train_features, train_labels, test_features, test_labels = split_data(train_data)

    return train_features, train_labels, test_features, test_labels


def SK_build_model(train_features, train_labels):
    """
    Build a linear regression model using the SKLearn library and print some information relating to it
    :param train_features: Independent data to train on
    :param train_labels: Dependent data to train on
    :return: SKLearn model object
    """
    model = linear_model.LinearRegression().fit(train_features, train_labels)

    print('SKLearn Regression Intercept: \n', model.intercept_)
    print('SKlearn Regression Coefficients: \n', model.coef_)

    importance = model.coef_

    # summarize feature importance
    for i, v in enumerate(importance):
        print('SKLearn Feature: %0d, Score: %.5f' % (i, v))

    return model


def SM_build_model(train_features, train_labels):
    """
    Build a linear regression model using the Statsmodel library and print some information relating to it
    :param train_features: Independent data to train on
    :param train_labels: Dependent data to train on
    :return: SM model object
    """
    # SM model creation
    model = sm.OLS(train_labels, train_features).fit()
    print_model = model.summary()
    print(print_model)

    return model


def predict_and_metrics(to_predict, actual_values, model, model_type="N/A", dataset="N/A"):
    """
    Perform predictions using a built linear regression model and display metrics for it
    Current metrics supported:
        -   mean_absolute_error
        -   R_squared
    :param to_predict: Unseen data that the model will base its predictions of
    :param actual_values: Actual recorded values for comparison
    :param model: Model object to use for predictions
    :param model_type: string for mentioning which library is being used for the model
    :param dataset: string for mentioning which dataset is being analysed
    :return: Predictions given by model
    """
    predictions = model.predict(to_predict)  # model predictions

    to_predict1, predictions = unscale_data(to_predict, predictions)
    to_predict, actual_values = unscale_data(to_predict, actual_values)  # we need to unscale the data for plotting

    print(snsregressionplot(actual_values, predictions,
                            "Predicted Cases vs Actual Cases (" + dataset + ")",
                            ylabel=" Predicted Cases", xlabel="Actual Cases (" + dataset + ")"))

    print(dataset + " data " + model_type + " model mean_absolute_error:",
          mean_absolute_error(actual_values, predictions))
    print(dataset + " data " + model_type + " model R^2:", r2_score(actual_values, predictions), "\n")

    return predictions


def unscale_data(features, labels):
    results = np.concatenate((features, labels[:, None]), axis=1)

    rescaled_results = x_scaler.inverse_transform(results)

    features = rescaled_results[:, 0:-2]
    labels = rescaled_results[:, -1]

    return features, labels


def run_main():
    """
    Main Function
    """

    """
    Train US Test EU
    """

    train_features, train_labels, test_features, test_labels = main_split_and_preprocess(US_DATA, EUROPE_DATA,
                                                                                         usa_features(),
                                                                                         europe_features())

    SKlearn_model = SK_build_model(train_features, train_labels)
    SM_model = SM_build_model(train_features, train_labels)

    #SKpredictionsUS = predict_and_metrics(train_features, train_labels, SKlearn_model, "SKLearn", "US")
    #SMpredictionsUS = predict_and_metrics(train_features, train_labels, SM_model, "Statsmodel", "US")

    #US_tostore = load_data('Project Data', US_DATA)
    #US_tostore['SKLearn Predictions US'] = SKpredictionsUS
    #US_tostore['Statsmodel Predictions US'] = SMpredictionsUS

    #US_tostore.to_csv('Final US Data.csv')
    #US_tostore.to_excel('Final US Data.xlsx')

    #SKpredictionsEU = predict_and_metrics(test_features, test_labels, SKlearn_model, "SKLearn", "EU")
    #SMpredictionsEU = predict_and_metrics(test_features, test_labels, SM_model, "Statsmodel", "EU")

    EU_tostore = load_data('Project Data', EUROPE_DATA)
    #EU_tostore['SKLearn Predictions EU'] = SKpredictionsEU
    #EU_tostore['Statsmodel Predictions EU'] = SMpredictionsEU

    """"
    Train US Test Hypothetical-1 (Tests = Population EU)
    """

    train_features, train_labels, test_features, test_labels = main_split_and_preprocess(US_DATA, EUROPE_DATA,
                                                                                         usa_features(),
                                                                                         hypothetical1_features())

    Hypo1SK = predict_and_metrics(test_features, test_labels, SKlearn_model, "SKLearn", "Tests = 1.1 * Population")
    Hypo1SM = predict_and_metrics(test_features, test_labels, SM_model, "Statsmodel", "Tests = 1.1 * Population")

    EU_tostore['SKLearn Predictions Hypothetical EU'] = Hypo1SK
    EU_tostore['Statsmodel Predictions Hypothetical EU'] = Hypo1SM

    EU_tostore.to_csv('Final EU Data.csv')
    EU_tostore.to_excel('Final EU Data.xlsx')

    train_features, train_labels, test_features, test_labels = main_split_and_preprocess(US_DATA, TEST_COUNTRIES,
                                                                                         usa_features(),
                                                                                         testcountry_features(),
                                                                                         set2=TEST_COUNTRIES)
    testSK = predict_and_metrics(test_features, test_labels, SKlearn_model, "SKLearn", "Test_countries")
    testSM = predict_and_metrics(test_features, test_labels, SM_model, "Statsmodel", "Test_countries")

    train_features, train_labels, test_features, test_labels = main_split_and_preprocess(US_DATA, TEST_COUNTRIES,
                                                                                         usa_features(),
                                                                                         hypothetical2_features(),
                                                                                         set2=TEST_COUNTRIES)

    HypotestSK = predict_and_metrics(test_features, test_labels, SKlearn_model, "SKLearn", "Test_countries")
    HypotestSM = predict_and_metrics(test_features, test_labels, SM_model, "Statsmodel", "Test_countries")

    test_tostore = load_data('Project Data', TEST_COUNTRIES)

    test_tostore['SKLearn Predictions normal EU'] = testSK
    test_tostore['Statsmodel Predictions normal EU'] = testSM
    test_tostore['SKLearn Predictions Hypothetical EU'] = HypotestSK
    test_tostore['Statsmodel Predictions Hypothetical EU'] = HypotestSM

    test_tostore.to_csv('Final Test Country Data.csv')
    test_tostore.to_excel('Final Test Data.xlsx')


if __name__ == "__main__":
    run_main()
