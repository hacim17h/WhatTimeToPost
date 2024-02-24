# Name: Micah Calloway Student ID: 010663003
import datetime
from datetime import datetime, timedelta
from datetime import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


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

    # Stores the original data and then breaks it into groups based
    # upon predetermined score levels
    original_data = pd.concat([pd.DataFrame(y), pd.DataFrame(day_of_the_week), pd.DataFrame(X)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'score']
    original_data.columns = headers

    very_low_scores = original_data[original_data['score'] == 0].sample(n=1000, random_state=59)
    very_high_scores = original_data[original_data['score'] >= 10000].sample(n=1000, random_state=59)

    # Re-combines the data that had been selected
    grouped_data = pd.concat([very_low_scores, very_high_scores])
    scores = grouped_data.iloc[:, -1].values
    X2 = grouped_data.iloc[:, :-1].values

    # Assigns an upvote strength based upon upvotes and transforms the
    # data with the new score
    upvote_strength = scores.copy()
    upvote_strength[upvote_strength == 0] = 0
    upvote_strength[upvote_strength >= 10000] = 1

    # Creates the CSV file with the cleaned and prepared data
    combined_data = pd.concat([pd.DataFrame(X2), pd.DataFrame(upvote_strength)], axis=1)
    headers = ['posting_hour', 'day_of_the_week', 'upvote_strength']
    combined_data.columns = headers

    file_path = output_name
    combined_data.to_csv(file_path, index=False)


def predict_best_time(date, hour):
    """Makes a prediction on what time the user should post based upon
    the day they are posting and the time they want to post after. The
    hour variable is in 24-hour format and accepts 0-23 and the
    date variable is a date time object that will be converted
    into a day of the week numeric format with 0-6 representing
    Monday - Sunday after time zone conversions.
    """

    # Stores the users selected time and converts it to UTC
    time_now = datetime.now()
    utc_time_now = datetime.utcnow()
    utc_offset = utc_time_now - time_now
    print(f"the utc offset is: {utc_offset}")

    selected_time = datetime.combine(date, hour)
    end_of_day = datetime.combine(date.now() + timedelta(days=1), time(0, 0, 0))
    print(f"The user selected time is: {selected_time}")
    print(f"The end of the day is: {end_of_day}")
    times_to_predict = [selected_time]

    while selected_time < (end_of_day - timedelta(hours=1)):
        times_to_predict.append(selected_time + timedelta(hours=1))
        selected_time = selected_time + timedelta(hours=1)

    local_time_group = []
    for times in times_to_predict:
        local_time_group.append(times.hour)

    local_time_group = np.array(local_time_group)

    converted_group = []
    for times in times_to_predict:
        converted_group.append(times + utc_offset)

    prediction_group = []
    for times in converted_group:
        prediction_group.append([times.hour, times.weekday()])

    print("The times to predict are:")
    print(times_to_predict)

    print("The converted times to predict are:")
    print(converted_group)

    print("The final converted prediction group is: ")
    print(prediction_group)

    utc_time = selected_time + utc_offset
    print(f"The user selected time in utc is: {utc_time}")

    day_of_week = utc_time.weekday()
    print(f"the day of the week is: {day_of_week}")

    hour = utc_time.hour
    print(f"the hour in utc is: {hour}")

    prediction = clf.predict([[hour, day_of_week]])
    print(f"For the {day_of_week} day of the week on the {hour} hour "
          f"the model predicted a {prediction} value for the post strength")

    selected_predictions = clf.predict(prediction_group)

    print("The predictions for the converted prediction groups are: ")
    print(selected_predictions)
    print(f"The local time group is: {local_time_group}")
    print(f"The selected predictions are: {selected_predictions}")

    results = np.concatenate((local_time_group.reshape(-1, 1), selected_predictions.reshape(-1, 1)), axis=1)

    print("The results are: ")
    print(results)
    best_time = None
    for item in results:
        if item[1] == 0:
            print(f"{item[0]} is not a viable hour to post")
        if item[1] == 1:
            print(f"{item[0]} is a good time to post")
            if best_time is None:
                best_time = item[0]

    print(f"The best time to post is {best_time}")



if __name__ == '__main__':
    # Take the cleaned dataset and create data for the independent and dependent variable
    df = pd.read_csv('post_data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Creates the test set and the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    # Trains the model using a decision tree classifier
    clf = DecisionTreeClassifier(criterion='entropy', random_state=10)
    clf.fit(X_train, y_train)

    # Predicts the test set results
    predictions = clf.predict(X_test)

    # Creates a confusion matrix to verify the accuracy and prints it
    cm = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(cm)
    print(f"The accuracy is {accuracy}")

    selected_hour = 0
    predict_best_time(datetime.now(), time(selected_hour, 0, 0))



    #post_formatter_a('data_posts_2.csv', 'clean_posts_24.csv')
    #post_formatter_b('data_posts_2.csv', 'clean_posts_25.csv')
    #post_formatter_c('data_posts_2.csv', 'clean_posts_27.csv')

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




