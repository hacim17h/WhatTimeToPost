# Name: Micah Calloway Student ID: 010663003
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Use a breakpoint in the code line below to debug your script.
# Press Ctrl+F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def post_formatter_a(filename, output_name):
    """Creates a CSV file formatted with the posting hour and upvote strength."""
    dataset = pd.read_csv(filename, encoding='latin-1')
    X = dataset.iloc[:, 3].values
    y = dataset.iloc[:, -1].values

    # Separates the time into hours and days of the week
    y = pd.to_datetime(y, unit='s').round('h')
    hours = y.hour
    minutes = y.minute
    day_of_the_week = y.day_of_week
    time_in_minutes = (hours * 60) + minutes
    dataset['day_of_the_week'] = day_of_the_week
    dataset['time_in_minutes'] = time_in_minutes
    y = hours

    # Stores the original data and then breaks it into groups based upon predetermined score levels
    original_data = pd.concat([pd.DataFrame(y), pd.DataFrame(day_of_the_week), pd.DataFrame(X)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'score']
    original_data.columns = headers

    very_low_scores = original_data[(original_data['score'] >= 0) & (original_data['score'] <= 3)].sample(
        n=1000, random_state=59)
    low_scores = original_data[(original_data['score'] >= 4) & (original_data['score'] <= 24)].sample(
        n=1000, random_state=59)
    moderate_scores = original_data[(original_data['score'] >= 25) & (original_data['score'] <= 99)].sample(
        n=1000, random_state=59)
    high_scores = original_data[(original_data['score'] >= 100) & (original_data['score'] <= 999)].sample(
        n=1000, random_state=59)
    very_high_scores = original_data[original_data['score'] >= 1000].sample(n=1000, random_state=59)

    # Re-combines the data that had been selected
    grouped_data = pd.concat([very_low_scores, low_scores, moderate_scores, high_scores, very_high_scores])
    scores = grouped_data.iloc[:, -1].values
    X2 = grouped_data.iloc[:, 0].values

    # Assigns an upvote strength based upon upvotes and transforms the data with the new score
    upvote_strength = scores.copy()
    upvote_strength[np.logical_and(upvote_strength >= 0, upvote_strength <= 3)] = 0
    upvote_strength[np.logical_and(upvote_strength >= 4, upvote_strength <= 24)] = 1
    upvote_strength[np.logical_and(upvote_strength >= 25, upvote_strength <= 99)] = 2
    upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 999)] = 3
    upvote_strength[upvote_strength >= 1000] = 4

    # Creates the CSV file with the cleaned and prepared data
    combined_data = pd.concat([pd.DataFrame(X2), pd.DataFrame(upvote_strength)], axis=1)
    headers = ['posting_hour', 'upvote_strength']
    combined_data.columns = headers

    file_path = output_name
    combined_data.to_csv(file_path, index=False)


def post_formatter_b(filename, output_name):
    """Creates a CSV file formatted with the posting hour day of the week, and upvote strength."""
    dataset = pd.read_csv(filename, encoding='latin-1')
    X = dataset.iloc[:, 3].values
    y = dataset.iloc[:, -1].values

    # Separates the time into hours and days of the week
    y = pd.to_datetime(y, unit='s').round('h')
    hours = y.hour
    minutes = y.minute
    day_of_the_week = y.day_of_week
    time_in_minutes = (hours * 60) + minutes
    dataset['day_of_the_week'] = day_of_the_week
    dataset['time_in_minutes'] = time_in_minutes
    y = hours

    # Stores the original data and then breaks it into groups based upon predetermined score levels
    original_data = pd.concat([pd.DataFrame(y), pd.DataFrame(day_of_the_week), pd.DataFrame(X)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'score']
    original_data.columns = headers

    very_low_scores = original_data[(original_data['score'] >= 0) & (original_data['score'] <= 3)].sample(
        n=1000, random_state=59)
    low_scores = original_data[(original_data['score'] >= 4) & (original_data['score'] <= 24)].sample(
        n=1000, random_state=59)
    moderate_scores = original_data[(original_data['score'] >= 25) & (original_data['score'] <= 99)].sample(
        n=1000, random_state=59)
    high_scores = original_data[(original_data['score'] >= 100) & (original_data['score'] <= 999)].sample(
        n=1000, random_state=59)
    very_high_scores = original_data[original_data['score'] >= 1000].sample(n=1000, random_state=59)

    # Re-combines the data that had been selected
    grouped_data = pd.concat([very_low_scores, low_scores, moderate_scores, high_scores, very_high_scores])
    scores = grouped_data.iloc[:, -1].values
    X2 = grouped_data.iloc[:, :-1].values

    # Assigns an upvote strength based upon upvotes and transforms the data with the new score
    upvote_strength = scores.copy()
    upvote_strength[np.logical_and(upvote_strength >= 0, upvote_strength <= 3)] = 0
    upvote_strength[np.logical_and(upvote_strength >= 4, upvote_strength <= 24)] = 1
    upvote_strength[np.logical_and(upvote_strength >= 25, upvote_strength <= 99)] = 2
    upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 999)] = 3
    upvote_strength[upvote_strength >= 1000] = 4

    # Creates the CSV file with the cleaned and prepared data
    combined_data = pd.concat([pd.DataFrame(X2), pd.DataFrame(upvote_strength)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'upvote_strength']
    combined_data.columns = headers

    file_path = output_name
    combined_data.to_csv(file_path, index=False)


def post_formatter_c(filename, output_name):
    """Creates a CSV file formatted with the posting hour day of the week, and upvote strength.
    This is for extreme outliers and only the highest and lowest will be added.
    """
    dataset = pd.read_csv(filename, encoding='latin-1')
    X = dataset.iloc[:, 3].values
    y = dataset.iloc[:, -1].values

    # Separates the time into hours and days of the week
    y = pd.to_datetime(y, unit='s').round('h')
    hours = y.hour
    minutes = y.minute
    day_of_the_week = y.day_of_week
    time_in_minutes = (hours * 60) + minutes
    dataset['day_of_the_week'] = day_of_the_week
    dataset['time_in_minutes'] = time_in_minutes
    y = hours

    # Stores the original data and then breaks it into groups based upon predetermined score levels
    original_data = pd.concat([pd.DataFrame(y), pd.DataFrame(day_of_the_week), pd.DataFrame(X)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'score']
    original_data.columns = headers

    very_low_scores = original_data[original_data['score'] == 0].sample(n=1000, random_state=59)
    very_high_scores = original_data[original_data['score'] >= 10000].sample(n=1000, random_state=59)

    # Re-combines the data that had been selected
    grouped_data = pd.concat([very_low_scores, very_high_scores])
    scores = grouped_data.iloc[:, -1].values
    X2 = grouped_data.iloc[:, :-1].values

    # Assigns an upvote strength based upon upvotes and transforms the data with the new score
    upvote_strength = scores.copy()
    upvote_strength[upvote_strength == 0] = 0
    upvote_strength[upvote_strength >= 10000] = 1

    # Creates the CSV file with the cleaned and prepared data
    combined_data = pd.concat([pd.DataFrame(X2), pd.DataFrame(upvote_strength)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'upvote_strength']
    combined_data.columns = headers

    file_path = output_name
    combined_data.to_csv(file_path, index=False)


if __name__ == '__main__':
    #post_formatter_a('data_posts_2.csv', 'clean_posts_24.csv')
    #post_formatter_b('data_posts_2.csv', 'clean_posts_25.csv')
    post_formatter_c('data_posts_2.csv', 'clean_posts_26.csv')

    # dataset = pd.read_csv('data_posts_2.csv', encoding='latin-1')
    # X = dataset.iloc[:, 3].values
    # y = dataset.iloc[:, -1].values
    # #y = pd.to_datetime(y, unit='s').round('30min')
    # y = pd.to_datetime(y, unit='s').round('h')
    #
    # # This section creates a new column out of the timestamp data calculating the total minutes of that
    # # particular time to be more easily used by the model.
    # hours = y.hour
    # minutes = y.minute
    # day_of_the_week = y.day_of_week
    # time_in_minutes = (hours * 60) + minutes
    # dataset['day_of_the_week'] = day_of_the_week
    # dataset['time_in_minutes'] = time_in_minutes
    # y = hours
    # # upvote_strength = X.copy()
    # # upvote_strength[np.logical_and(upvote_strength >= 0, upvote_strength <= 3)] = 0
    # # upvote_strength[np.logical_and(upvote_strength >= 4, upvote_strength <= 24)] = 1
    # # upvote_strength[np.logical_and(upvote_strength >= 25, upvote_strength <= 99)] = 2
    # # upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 999)] = 3
    # # upvote_strength[upvote_strength >= 1000] = 4
    # # upvote_strength[np.logical_and(upvote_strength >= 10, upvote_strength <= 100)] = 0
    # # upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 1000)] = 1
    # # upvote_strength[upvote_strength > 1000] = 2
    # # upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 200)] = 0
    # # upvote_strength[np.logical_and(upvote_strength >= 201, upvote_strength <= 1000)] = 1
    # # upvote_strength[upvote_strength > 1000] = 2
    #
    # original_data = pd.concat([pd.DataFrame(y), pd.DataFrame(day_of_the_week), pd.DataFrame(X)], axis=1)
    # headers = ['posting_hour', 'day_of_the_week', 'score']
    # original_data.columns = headers
    # very_low_scores = original_data[(original_data['score'] >= 0) & (original_data['score'] <= 3)].sample(n=1000, random_state=59)
    # low_scores = original_data[(original_data['score'] >= 4) & (original_data['score'] <= 24)].sample(n=1000, random_state=59)
    # moderate_scores = original_data[(original_data['score'] >= 25) & (original_data['score'] <= 99)].sample(n=1000, random_state=59)
    # high_scores = original_data[(original_data['score'] >= 100) & (original_data['score'] <= 999)].sample(n=1000, random_state=59)
    # very_high_scores = original_data[original_data['score'] >= 1000].sample(n=1000, random_state=59)
    # grouped_data = pd.concat([very_low_scores, low_scores, moderate_scores, high_scores, very_high_scores])
    #
    # scores = grouped_data.iloc[:, -1].values
    # X2 = grouped_data.iloc[:, 0].values
    # upvote_strength = scores.copy()
    # upvote_strength[np.logical_and(upvote_strength >= 0, upvote_strength <= 3)] = 0
    # upvote_strength[np.logical_and(upvote_strength >= 4, upvote_strength <= 24)] = 1
    # upvote_strength[np.logical_and(upvote_strength >= 25, upvote_strength <= 99)] = 2
    # upvote_strength[np.logical_and(upvote_strength >= 100, upvote_strength <= 999)] = 3
    # upvote_strength[upvote_strength >= 1000] = 4
    #
    # # print("These are very low scores")
    # # print(very_low_scores)
    # #
    # # print("These are very high scores")
    # # print(very_high_scores)
    #
    # # combined_data = pd.concat([pd.DataFrame(y), pd.DataFrame(upvote_strength)], axis=1)
    # combined_data = pd.concat([pd.DataFrame(X2), pd.DataFrame(upvote_strength)], axis=1)
    # headers = ['posting_hour', 'upvote_strength']
    # combined_data.columns = headers
    # file_path = 'clean_posts_23.csv'
    # combined_data.to_csv(file_path, index=False)


    # # The dataset is split into test and training sets and feature scaling is applied.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    # regressor.fit(X_train.reshape(-1, 1), y_train)
    #
    # print(regressor.predict([[10000]]))
    #
    # X_grid = np.arange(min(X_train), max(X_train), 0.1)
    # X_grid = X_grid.reshape((len(X_grid), 1))
    # plt.scatter(X_train, y_train, color='red')
    # plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    # plt.title('Upvotes vs Posting Time (Random Forest Regression)')
    # plt.xlabel('Upvotes')
    # plt.ylabel('Post Time')
    # plt.show()
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train.reshape(-1, 1))
    # X_test = sc.transform(X_test.reshape(-1, 1))
    #
    # # A simple linear regression model is applied and scatter plots are made for the dataset.
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    #
    # plt.scatter(X_train, y_train, color='red')
    # plt.plot(X_train, regressor.predict(X_train), color='blue')
    # plt.title('Upvotes vs Posting time (Training Set)')
    # plt.xlabel('Upvotes')
    # plt.ylabel('Post time')
    # plt.show()
    #
    # plt.scatter(X_test, y_test, color='red')
    # plt.plot(X_train, regressor.predict(X_train), color='blue')
    # plt.title('Upvotes vs Posting time (Test Set)')
    # plt.xlabel('Upvotes')
    # plt.ylabel('Post time')
    # plt.show()

    # print("This is the x train and x test values")
    # print(X_train)
    # print(X_test)
    #
    # print("This is the y train and y test values")
    # print(y_train)
    # print(y_test)

    # print(day_of_the_week)
    # print(X)
    # print(y)




