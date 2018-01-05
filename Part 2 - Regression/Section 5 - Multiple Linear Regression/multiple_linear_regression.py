# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
PATH = 'E:/Udemy/MachineLearningAZ/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression'
dataset = pd.read_csv(r'{}/50_Startups.csv'.format(PATH));

# Set up independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Notice that we might need to encode the state names
# since we also want to remove relational order, we want to onehotencode it
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Note that the state is the last column in X
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
# Now we have to onehotencode this
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap (we're getting rid of one column to avoid dummy variable trap)
X = X[:, 1:]

# Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
y_pred, y_test

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# statsmodels doesn't take b0 into account so we have to add it ourselves
# since our X only accounts for the independent variables, we need to include
# the constant b0 in the multiple linear regression equation.
# that's why we need to add the column of 1s in the matrix of features
X
X = np.append(values=X, arr=np.ones((X.shape[0], 1)).astype(int), axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# endog is the dependent variable, exog is the independent variable
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# Now we remove index 2
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
# Now we see that we should remove index 1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# We then see we should remove index 4
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# Notice that we have left index 5 which we could leave out or choose to leave in.
# We can consider this because 0.06 is close enough to 0.05

# Suppose that we want to take it out. Then we're left with a simple linear regression problem
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
