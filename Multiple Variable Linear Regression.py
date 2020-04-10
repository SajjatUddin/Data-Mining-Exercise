#Multiple Variable Linear Regression

#Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Dataset.csv')
x=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,[4]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#Finding mean Squared Error, r2 Score, rms 
from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
r2_score=r2_score(y_test,y_pred)

print(rms)
print(r2_score)