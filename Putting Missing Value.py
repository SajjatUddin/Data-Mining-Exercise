#Putting Missing Value

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataFrame = pd.read_csv("MIssingValueDataset.csv")
trainingData = dataFrame.iloc[:, :].values
dataset = dataFrame.iloc[:, :].values

#Putting The Missing Value by Importing Imputer libraries from Sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values="NaN", strategy="mean")
imputer = imputer.fit(trainingData[:, 1:2])
dataset[:, 1:2] = imputer.transform(dataset[:, 1:2])
print(trainingData)

