# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
PATH = 'E:/Udemy/MachineLearningAZ/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression'
dataset = pd.read_csv(r'{}/Position_Salaries.csv'.format(PATH))
dataset

# Set up our dependent and independent variables
# X should be thought of as a matrix while y should be a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# We don't need to split the data into training set and test set because
# we don't have enough data to do so.
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# We want to compare the results of the Linear Regression model and the Polynomial regression model

# Fitting the data to the Linear Regression model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X, y)

# Fitting the data to the Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=3)
# we use fit_transform because we have to fit poly_reg then apply it to the data
X_poly = polynomial_regression.fit_transform(X)
X_poly, X
linear_regression2 = LinearRegression()
linear_regression2.fit(X_poly, y)

# Visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regression.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_regression2.predict(polynomial_regression.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear regression
linear_regression.predict(6.5)

# Predicting a new result with Polynomial regression
linear_regression2.predict(polynomial_regression.fit_transform(6.5))
