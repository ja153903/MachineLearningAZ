# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importing the dataset
PATH = 'E:/Udemy/MachineLearningAZ/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression'
dataset = pd.read_csv(r'{}/Salary_Data.csv'.format(PATH))
X = dataset.iloc[:, :-1].values # this is the years of experience
y = dataset.iloc[:, 1].values # these are the salaries

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling (most libraries in Python will take care of feature scaling such as linear regression)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # fit the training set

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred, y_test # the discrepancies become more clear when we graph these results

# Visualizing the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') # predictions of the training set
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()
