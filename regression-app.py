import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def FitPolynomialRegression(K,x,y):
    Xmatrix = np.ones((x.size,1))
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    for i in range(1,K+1):
        colVec = np.power(x,i)
        Xmatrix = np.hstack((Xmatrix,colVec))
    Xmatrix_pinv = np.linalg.pinv(Xmatrix)
    ret = np.dot(Xmatrix_pinv, y)
    return ret.reshape(1,ret.size)

def EvalPolynomial(x,w):
    y = []
    # x = x.reshape((x.size,1))
    for i in range(x.size):
        output = 0
        for j in range(w.size):
            output += w[j]*x[i]**j
        y.append(float(output))
    return np.array(y)

def GetBestPolynomial(xTrain,yTrain,xTest,yTest,h):
    errorsTrainVec = []
    errorsTestVec = []
    xTrain = xTrain.reshape((xTrain.shape[0],1))
    xTest = xTest.reshape((xTest.shape[0],1))
    yTest = yTest.reshape((yTest.shape[0],1))
    for i in range(1,h+1):
        weights = FitPolynomialRegression(i,xTrain,yTrain)
        weights = weights.reshape(weights.size,1)
        outputsTrain = EvalPolynomial(xTrain, weights)
        outputsTrain = outputsTrain.reshape(outputsTrain.size,1)
        outputsTest = EvalPolynomial(xTest, weights)
        outputsTest = outputsTest.reshape(outputsTest.size,1)
        # plt.scatter(xTrain,outputsTrain,color='green',label='Training')
        # plt.scatter(xTest,yTest,color='orange',label='Testing')
        # plt.xlabel('Inputs (x)')
        # plt.ylabel('Outputs (y)')
        # title = 'Data Fitting, h = ' + str(i)
        # plt.title(title)
        # plt.legend()
        # name = "data" + str(i)+".jpg"
        # plt.savefig(name)
        # plt.show()

        errorTrain = (np.linalg.norm(yTrain - outputsTrain))**2
        errorTest = (np.linalg.norm(yTest - outputsTest))**2
        errorsTrainVec.append(errorTrain/75)
        errorsTestVec.append(errorTest/25)
    plt.plot(np.arange(1,h+1),errorsTrainVec,label='Training')
    plt.plot(np.arange(1,h+1),errorsTestVec,label='Testing')
    # plt.plot([3,3,3,3],[0,10,100,1000],'r--',alpha=0.4,label='d = 3')
    plt.xlabel("Model Complexity (Polynomial Degree)")
    plt.ylabel("Mean Squared Error (Based on Log Scale)")
    plt.title("Error vs. Model Complexity")
    plt.xticks(np.arange(1,10,step=1))
    plt.yscale('log')
    plt.legend()
    # name1 = "emc.jpg"
    # plt.savefig(name1)
    plt.show()
    print("\nTraining Error:\n")
    print(np.array(errorsTrainVec))
    print("\nTesting Error:\n")
    print(np.array(errorsTestVec))
    print("\nBest choice of d = "+str(errorsTestVec.index(min(errorsTestVec))+1)+"\n")

if __name__ == "__main__":
    df = pd.read_csv("hw1_polyreg.csv")
    X = df.iloc[:, 1:2].values
    y = df.iloc[:, 2:].values

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)

    # print(FitPolynomialRegression(9,XTrain,yTrain))
    # print(np.polyfit(XTrain.reshape(1,XTrain.size)[0],yTrain,9))
    # print(FitPolynomialRegression(2, np.array([[1],[2],[3]]), np.array([[1],[2],[3]])))
    # print(EvalPolynomial(np.arange(7),np.arange(5)))
    GetBestPolynomial(XTrain,yTrain,XTest, yTest, 10)