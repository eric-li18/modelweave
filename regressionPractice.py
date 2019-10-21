import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

#Dataframe setup
data = pd.read_csv('./SATvsGPA.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

#We can take a 2/3 train, 1/3 test split of the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

model = LinearRegression().fit(X_train,y_train)
#model.fit(X_train,y_train)

#Find the slope or the weight of the regression line rounded to 7 digits
weight_label = 'w: ' + str(model.coef_[0].round(7))

#Plot the train data with regression line
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,model.predict(X_train), color='green',label=weight_label)
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Training Data)')
plt.legend()
plt.savefig("training.jpg")
plt.show()

#Plot the test data with regression line
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,model.predict(X_train), color='green',label=weight_label)
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Testing Data)')
plt.legend()
plt.savefig("testing.jpg")
plt.show()

#Predict GPA points based on inputted SAT scores
X_pred = []
i=1
while(i):
    input_data=input("Enter a SAT Score (or enter any non integer to exit): ")
    if (input_data.isnumeric() and int(input_data) <=2400):
        X_pred.append([int(input_data)])
    else:
        i=0
if(len(X_pred) != 0):
    y_pred = model.predict(X_pred)
    print(y_pred)
else:
    print("No points to show")


#Unblock this code to check that the predictions are correct in relation to the regression line through visualization
""" plt.scatter(X_pred, y_pred, color='red')
plt.plot(X_train,model.predict(X_train), color='green')
plt.xlabel('SAT Scores')
plt.ylabel('GPA (4.0 scale)')
plt.title('SAT vs. GPA Scores (Prediction Data)')
plt.show() """
