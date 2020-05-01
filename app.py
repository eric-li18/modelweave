import time

import matplotlib

matplotlib.use('Agg')
from scipy.spatial import voronoi_plot_2d, Voronoi
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import model_selection, datasets
import logreg
import polyreg
import matplotlib.pyplot as plt


def showRegression():
    with st.echo():  # Shows code that appears
        def GetBestPolynomial(xTrain, yTrain, xTest, yTest, h):
            errorsTrainVec = []
            errorsTestVec = []
            for i in range(1, h + 1):
                weights = polyreg.FitPolynomialRegression(i, xTrain, yTrain)
                outputsTrain = polyreg.EvalPolynomial(xTrain, weights)
                outputsTest = polyreg.EvalPolynomial(xTest, weights)
                errorTrain = (np.linalg.norm(yTrain - outputsTrain)) ** 2
                errorTest = (np.linalg.norm(yTest - outputsTest)) ** 2
                errorsTrainVec.append(errorTrain / 75)
                errorsTestVec.append(errorTest / 25)
            return np.array(errorsTestVec), np.array(errorsTrainVec)

def showLogistic():
    with st.echo():
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def Gradient(w, X, y):
            return np.dot(X.T, sigmoid(np.dot(X, w)) - y)

        def FindWeight(tolerance, learningrate, X, y):
            # Stop when we reach descent tolerance
            iterations = 1
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            wLast = np.ones(X.shape[1])
            wNext = wLast - learningrate * Gradient(wLast, X, y)
            while abs(np.sqrt((wLast - wNext).dot(wLast - wNext))) >= tolerance:
                iterations += 1
                wLast = wNext
                wNext = wNext - learningrate * Gradient(wNext, X, y)
            return wNext, iterations

def aboutSection():
    st.markdown('''
            # About

            This web-app was created using [streamlit](https://streamlit.io), [matplotlib](https://matplotlib.org), 
            [sci-kit learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/) and [NumPy](
            https://numpy.org). It is currently a WIP with plans to add more features soon!
            ''')


if __name__ == "__main__":

    option = st.sidebar.selectbox("Select an interactive visualization",
                                  ("Polynomial Regression", "Logistic Regression", "1-NN"))
    if option == "Polynomial Regression":
        st.markdown('''
            # Visualize Non-Linear Regression
            This simple web-app allows you to see exactly how [Polynomial Regression](
            https://en.wikipedia.org/wiki/Polynomial_regression) takes place. 

            From the sidebar, you can control the size of the test data as well as the polynomial being fit to the data. 
            Experiment with different test sizes and watch as the regression line underfits or overfits based on your choice 
            of polynomial!

            Enjoy and Happy Fitting!   
            ''')

        df = pd.read_csv("data/polyreg.csv")
        X = df.iloc[:, 0:1].values
        y = df.iloc[:, 1:].values
        split = st.sidebar.slider('Test Size (number of points)', min_value=0, max_value=X.shape[0] - 1, value=int(0.5 *
                                                                                                                   X.shape[
                                                                                                                       0]))
        XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=(float(split) / X.shape[0]),
                                                                        random_state=0)
        polynomial = st.sidebar.slider('Polynomial Size', min_value=1, max_value=5)

        if st.sidebar.checkbox("Show Dataframe"):
            st.markdown("#### Here is the Dataframe we're working with!")
            st.write(df)
        if st.sidebar.checkbox("Show Testing Data"):
            tempTest = np.hstack((XTest, yTest))
            st.markdown("#### Here is the Testing Data!")
            st.write(tempTest)

        st.markdown("### Here is a chart for the data fitted with the test data and the regression line")
        errTotal = polyreg.GetBestPolynomial(XTrain, yTrain, XTest, yTest, polynomial)
        if st.checkbox("Show Code"):
            showRegression()

        data = pd.DataFrame(errTotal, columns=["Train Error", "Test Error"])
        errTrain = errTotal[:,:1]
        errTest = errTotal[:,:2]
        st.markdown("## Error vs. Model Complexity")
        lastrowTrain = errTrain[0]
        lastrowTest = errTrain[0]

        chart = st.line_chart(errTotal[:1], 800, 800)
        for i in range(errTotal.shape[0]):
            newrowTrain = errTotal[i]
            chart.add_rows(newrowTrain)
            time.sleep(0.1)

        st.markdown("### Let's look into the train and test errors producing the line chart above")
        st.write(data)
        st.markdown('''
        As we can see, as the polynomial value increases, the regression line begins to overfit the data, resulting in a 
        lower train error, but increase in test error.
        ''')



    elif option == "Logistic Regression":
        # take the first two classes of the dataset i.e., first 100 instances.
        iris = datasets.load_iris()
        X = iris.data[:100, :2]  # first is rows (100 rows) and second is features (1-4 features)
        y = iris.target[:100]  # the labels
        split = st.sidebar.slider('Test Size (number of points)', min_value=0, max_value=X.shape[0] - 1, value=int(0.5 *
                                                                                                                   X.shape[
                                                                                                                       0]))
        XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=(float(split) / X.shape[0]),
                                                                        random_state=0)
        # tolerance = [0.0001, 0.9]
        # lr = [0.001, 0.5, 0.99]

        # add part to maybe move the descent tolerance and learning rate?
        # show iteration number
        # show gradient descent
        model, iterations = logreg.FindWeight(0.0001, 0.001, XTrain, yTrain)
        ConfusionMatrix = logreg.Predict(model, XTest, yTest)
        met = logreg.Metrics(ConfusionMatrix)
        print("True Positive: ", ConfusionMatrix[0], "True Negative: ", ConfusionMatrix[3], "\nFalse Positive: ",
              ConfusionMatrix[1], "False Negative: ", ConfusionMatrix[2])

        plt.scatter(XTrain[yTrain == 0][:, 0], XTrain[yTrain == 0][:, 1], color='b', label='0 Train')
        plt.scatter(XTrain[yTrain == 1][:, 0], XTrain[yTrain == 1][:, 1], color='g', label='1 Train')
        # plt.scatter(XTest[yTest == 0][:, 0], XTest[yTest == 0][:, 1], color='c', label='0 Test')
        # plt.scatter(XTest[yTest == 1][:, 0], XTest[yTest == 1][:, 1], color='y', label='1 Test')
        plt.title("Logistic Regression on Iris Dataset, alpha=" + str(0.001) + ", T=" + str(0.0001))
        x_values = [np.min(X[:, 0]), np.max(X[:, 1] + 2.5)]
        y_values = - (model[0] + np.dot(model[1], x_values)) / model[2]
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.xlabel("Feature x1")
        plt.ylabel("Feature x2")
        plt.legend()
        st.pyplot()


        if st.sidebar.checkbox("Show Code"):
            showLogistic()

    elif option == "1-NN":

        iris = datasets.load_iris()
        X = iris.data[:100, :2]  # first is rows (100 rows) and second is features (1-4 features)
        y = iris.target[:100]  # the labels
        split = st.sidebar.slider('Test Size (number of points)', min_value=0, max_value=X.shape[0] - 5, value=int(0.5 *
                                                                                                                   X.shape[
                                                                                                                       0]))
        XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=(float(split) / X.shape[0]))
        vor = Voronoi(XTrain, incremental=True)
        voronoi_plot_2d(vor)
        plt.scatter(XTrain[yTrain == 0][:, 0], XTrain[yTrain == 0][:, 1], color='b', label='0 Train')
        plt.scatter(XTrain[yTrain == 1][:, 0], XTrain[yTrain == 1][:, 1], color='g', label='1 Train')
        plt.xlabel("Feature x1")
        plt.ylabel("Feature x2")
        plt.title("Voronoi Tessellation")
        st.pyplot()

    aboutSection()
