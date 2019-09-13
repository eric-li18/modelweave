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



