import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


data = pd.read_csv('./SATvsGPA.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

regression = LinearRegression().fit(X_train,y_train)
#regression.fit(X_train,y_train)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regression.predict(X_train), color='green')
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Training Data)')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regression.predict(X_train), color='green')
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Testing Data)')
plt.show()


X_pred = []
i=1
while(i):
    input_data=input("Enter a SAT Score (or enter any non integer to exit): ")
    if (input_data.isnumeric() and int(input_data) <=2400):
        X_pred.append([int(input_data)])
    else:
        i=0

y_pred = regression.predict(X_pred)
print(y_pred)

#Unblock this code to check that the predictions are correct in relation to the regression line
""" plt.scatter(X_pred, y_pred, color='red')
plt.plot(X_train,regression.predict(X_train), color='green')
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Prediction Data)')
plt.show() """
