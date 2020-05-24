# ================================================================================================================
# ----------------------------------------------------------------------------------------------------------------
#									SIMPLE LINEAR REGRESSION
# ----------------------------------------------------------------------------------------------------------------
# ================================================================================================================

# Simple linear regression is applied to stock data, where the x values are time and y values are the stock closing price.
# This is not an ideal application of simple linear regression, but it suffices to be a good experiment.

import math
import pandas as pd
import config
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas
import datetime
import quandl
import streamlit as st

# for plotting
plt.style.use('ggplot')


class CustomLinearRegression:

    def __init__(self):
        self.intercept = 0
        self.slope = 0

    # arithmetic mean
    def arithmetic_mean(self, arr):
        total = 0.0
        for i in arr:
            total += i
        return total/len(arr)

    # finding the slope in best fit line
    def best_fit(self, dimOne, dimTwo):
        self.slope = ((self.arithmetic_mean(dimOne) * self.arithmetic_mean(dimTwo)) - self.arithmetic_mean(dimOne*dimTwo)) / \
            (self.arithmetic_mean(dimOne)**2 -
             self.arithmetic_mean(dimOne**2))  # formula for finding slope
        return self.slope

    # finding the best fit intercept
    def y_intercept(self, dimOne, dimTwo):
        self.intercept = self.arithmetic_mean(
            dimTwo) - (self.slope * self.arithmetic_mean(dimOne))
        return self.intercept

    # predict for future values based on model
    def predict(self, ip):
        ip = np.array(ip)
        # create a "predicted" array where the index corresponds to the index of the input
        predicted = [(self.slope*param) + self.intercept for param in ip]
        return predicted

    # find the squared error
    def squared_error(self, original, model):
        return sum((model - original) ** 2)

    # find co-efficient of determination for R^2
    @st.cache
    def r_squared(self, original, model):
        am_line = [self.arithmetic_mean(original) for y in original]
        sq_error = self.squared_error(original, model)
        sq_error_am = self.squared_error(original, am_line)
        # R^2 is nothing but 1 - of squared error for our model / squared error if the model only consisted of the mean
        return 1 - (sq_error/sq_error_am)


def main():
    # add a slider or button to change the dates length of time shown, currently only 180 days
    ticker = st.sidebar.selectbox(
        "Choose a stock to see real-time predictions", ("TSLA", "GOOG", "AAPL", "AMZN", "FB", "MSFT"))
    quandl.ApiConfig.api_key = config.QUANDL_API_KEY
    stk = quandl.get("WIKI/" + ticker)

    # model = CustomLinearRegression()
    model2 = LinearRegression(n_jobs=-1)

    # reset index to procure date - date was the initial default index
    stk = stk.reset_index()

    # Add them headers
    stk = stk[['Date', 'Adj. Open', 'Adj. High',
               'Adj. Low', 'Adj. Close', 'Volume']]

    stk['Date'] = pandas.to_datetime(stk['Date'])
    stk['Date'] = (stk['Date'] - stk['Date'].min()) / np.timedelta64(1, 'D')

    # The column that needs to be forcasted using linear regression
    forecast_col = 'Adj. Close'

    # take care of NA's
    stk.fillna(-999999, inplace=True)
    stk['label'] = stk[forecast_col]

    # IN CASE THE INPUT IS TO BE TAKEN IN FROM THE COMMAND PROMPT UNCOMMENT THE LINES BELOW

    # takes in input from the user
    # x = list(map(int, input("Enter x: \n").split()))
    # y = list(map(int, input("Enter y: \n").split()))

    # convert to an numpy array with datatype as 64 bit float.
    # x = np.array(x, dtype = np.float64)
    # y = np.array(y, dtype = np.float64)

    stk.dropna(inplace=True)

    x = np.array(stk['Date']).reshape(-1, 1)
    y = np.array(stk['label']).reshape(-1, 1)
    model2 = model2.fit(x, y)

    # Always in the order: first slope, then intercept
    # slope = model.best_fit(x, y)  # find slope
    # intercept = model.y_intercept(x, y)  # find the intercept

    max_value = int(x.max())
    st.markdown(""" # Linear Regression
    Simple linear regression is applied to stock data, where the $x$ values are time and $y$ values are the stock closing price. """)
    date_input = st.number_input(
        label="Enter a date to predict the stock price", min_value=0, max_value=max_value, value=int(round(max_value/2, -1)), step=1000)

    # line = model.predict([date_input])  # predict based on model

    # reg = [(slope*param) + intercept for param in x]

    # print("Predicted value(s) after linear regression :", line)

    # r_sqrd = model.r_squared(y, reg)
    # st.markdown("The $R^2$ Value is " + str(r_sqrd))
    st.markdown("The $R^2$ Value is " + str(model2.score(x, y)))
    st.markdown("The $R^2$ value (or coefficient of determination) is a value bounded between 0.0 and 1.0 that represents how well a model fits a set of data or the \"goodness of fit\".")

    # plt.scatter(x, y)
    # plt.plot(x, reg, color='green')
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Closing Price (USD)')
    plt.plot(x, y)
    plt.plot(x, model2.predict(x), color='black')
    plt.scatter([date_input], model2.predict(
        np.array([date_input]).reshape(1, -1)), color="green")
    st.pyplot()
    chart_data = pd.DataFrame(
        np.hstack((x, np.hstack((y, model2.predict(x))))), columns=["x", "True Values", "Regression Line"]).set_index("x")
    st.line_chart(chart_data,use_container_width=True,height=450)
