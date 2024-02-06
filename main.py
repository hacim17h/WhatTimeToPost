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
if __name__ == '__main__':
    dataset = pd.read_csv('top_posts.csv')
    X = dataset.iloc[:, 3:6].values
    y = dataset.iloc[:, -1].values
    y = pd.to_datetime(y, unit='s').round('30min')

    # This section creates a new column out of the timestamp data calculating the total minutes of that
    # particular time to be more easily used by the model.
    hours = y.hour
    minutes = y.minute
    day_of_the_week = y.day_of_week
    time_in_minutes = (hours * 60) + minutes
    dataset['day_of_the_week'] = day_of_the_week
    dataset['time_in_minutes'] = time_in_minutes
    y = time_in_minutes

    # The dataset is split into test and training sets and feature scaling is applied.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train[:, :])
    X_test = sc.transform(X_test[:, :])

    print("This is the x train and x test values")
    print(X_train)
    print(X_test)

    print("This is the y train and y test values")
    print(y_train)
    print(y_test)

    # print(day_of_the_week)
    # print(X)
    # print(y)




