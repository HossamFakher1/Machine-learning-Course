# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))

print(regressor.score(X_test, y_test))
# print(regressor.coef_,"DDDD",regressor.intercept_)

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)

# # Visualising the Training set results
# plt.scatter(X_train, y_train, color = 'red')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.show()



# # Visualising the Test set results
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_test,regressor.predict(X_test), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
