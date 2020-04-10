#Dataset Spliting

## Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('SplitingDataset.csv')

x=dataset.iloc[:,:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(x,test_size=0.2,random_state=0)

print(x_train)
print(x_test)