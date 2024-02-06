# Name: Micah Calloway Student ID: 010663003
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Use a breakpoint in the code line below to debug your script.
# Press Ctrl+F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv('top_posts.csv')
    X = dataset.iloc[:, 3].values
    y = dataset.iloc[:, -1]
    print(X)
    print(y)



