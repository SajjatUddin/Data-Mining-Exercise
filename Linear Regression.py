#Linear Regression

#Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,[1]]
y=dataset.iloc[:,[2]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3.0,random_state=0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# Visualising the Linear Regression results
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Exp Train')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()

# Visualising the Linear Regression results
plt.scatter(x_test,y_pred,color='red')
plt.plot(x_test,regressor.predict(x_test),color='green')
plt.title('Salary Vs Exp Test')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()

#print(np.int(np.round(regressor.predict(50000))))
