# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib .pyplot as plt
import pandas as pd

# Importing the dataset
# Note that you will need to enter the character 'r' before your csv file name
dataset = pd.read_csv(r'E:/Udemy/MachineLearningAZ/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
dataset

X = dataset.iloc[:, :-1].values # take all the columns except the last one (independent variables)
X

y = dataset.iloc[:, 3].values # we want our last column which is the purchased column (dependent variables)
y

# Take care of missing data
# Notice that we do have some missing data,
# To remedy this, we need to find a good idea to handle this.
# We can instead take the mean of the columns

# Imputer class allows us to take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

# Next we have to fit this imputer object to the matrix of feature X
# make sure that we only fit the values that contain NaN
imputer = imputer.fit(X[:, 1:3])
# transform allows us to replace the missing data with the imputer fitted one
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # this applies to the country column
# OneHotEncoder makes sure that the algorithm doesn't attribute order
# to the categorical variables
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# Now we have that the first three columns are replaced with 0s and 1s to represent the countries
X

# for the dependent variable y, we don't need to use onehotencoder
# we can just use LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

# Split data into training set and test set
# train_test_split is now factored into model_selection instead of cross_validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
# feature scaling allows models to converge faster
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# We don't need to apply feature scaling to the dependent variable because this time
# we have a classification problem i.e. 0 or 1
# However, for regression problems, this may differ.
