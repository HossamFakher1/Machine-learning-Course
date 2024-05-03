import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Heart attack.csv')
X = dataset.iloc[:, [1, 3]].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.metrics import accuracy_score ,classification_report,f1_score,precision_score,recall_score
from sklearn.svm import SVC

######################################################################
# classifier = SVC(kernel='linear', random_state = 0,C=.001)
# classifier.fit(X_train, y_train)
# acc_train = classifier.score(X_train,y_train)
# print(acc_train)

# y_pred = classifier.predict(X_test)
# acc_test = accuracy_score(y_test, y_pred)
# print(acc_test)
# ######################################################################
# classifier = SVC(kernel='rbf', random_state = 0,C=.001)
# classifier.fit(X_train, y_train)
# acc_train = classifier.score(X_train,y_train)
# print(acc_train)

# y_pred = classifier.predict(X_test)
# acc_test = accuracy_score(y_test, y_pred)
# print(acc_test)


# ######################################################################
# classifier = SVC(kernel='poly', degree=3,random_state = 0,C=.01)
# classifier.fit(X_train, y_train)
# acc_train = classifier.score(X_train,y_train)
# print(acc_train)

# y_pred = classifier.predict(X_test)
# acc_test = accuracy_score(y_test, y_pred)
# print(acc_test)

# ######################################################################
classifier = SVC(kernel='sigmoid', gamma=.02,random_state = 0,C=1)
classifier.fit(X_train, y_train)
acc_train = classifier.score(X_train,y_train)
# print(acc_train)

y_pred = classifier.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
# print(acc_test)
print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

