# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1, 3]].values
y = dataset.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
X[:,0]=lb.fit_transform(X[:,0])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print('Accuracy of Train : ',classifier.score(X_train, y_train))
print('Accuracy of Test : ',classifier.score(X_test, y_test))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(X, test_size = 0.25, random_state = 0)

