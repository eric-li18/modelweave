# from sklearn import datasets, model_selection
# import matplotlib.pyplot as plt
import numpy as np


def Predict(w, X, yTruth):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    pred = sigmoid(np.dot(X, w)).round()
    # Confusion Matrix as a list
    # ConfMat[0] is True Positive
    # ConfMat[1] is False Positive
    # ConfMat[2] is False Negative
    # ConfMat[3] is True Negative
    ConfMat = [0, 0, 0, 0]
    for i in range(pred.size):
        if pred[i] and yTruth[i]:
            # true positive
            ConfMat[0] += 1
        elif pred[i] and not yTruth[i]:
            # false positive
            ConfMat[1] += 1
        elif not pred[i] and yTruth[i]:
            # false negative
            ConfMat[2] += 1
        elif not pred[i] and not yTruth[i]:
            # true negative
            ConfMat[3] += 1
    return ConfMat


def Metrics(ConfMat):
    if not ConfMat[0] and not ConfMat[2]:
        recall = 0.0001
    else:
        recall = ConfMat[0] / (ConfMat[0] + ConfMat[2])
    if not ConfMat[0] and not ConfMat[1]:
        precision = 0.0001
    else:
        precision = ConfMat[0] / (ConfMat[0] + ConfMat[1])
    f1score = 2 * ((precision * recall) / (precision + recall))
    return [recall, precision, f1score]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def LossGradient(w, X, y):
    return np.dot(X.T, sigmoid(np.dot(X, w)) - y)


# def Loss(w, X, y):
#     h = sigmoid(np.dot(X, w))
#     return (-y * np.log(h + 1e-9) - (1 - y) * np.log(1 - h + 1e-9)).mean()

def FindWeight(tolerance, learningrate, X, y):
    # stop when reach descent tolerance
    iterations = 1
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    wLast = np.ones(X.shape[1])
    wNext = wLast - learningrate * LossGradient(wLast, X, y)
    # while abs(Loss(wLast, X, y) - Loss(wNext, X, y)) >= tolerance:
    while abs(np.sqrt((wLast - wNext).dot(wLast - wNext))) >= tolerance:
        iterations += 1
        wLast = wNext
        wNext = wNext - learningrate * LossGradient(wNext, X, y)
    return wNext, iterations
