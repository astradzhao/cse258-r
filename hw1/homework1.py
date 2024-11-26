import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

len(dataset)

answers = {} # Put your answers to each question in this dictionary

### Question 1
X = np.array([review['review_text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([review['rating'] for review in dataset])

model = linear_model.LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
theta0 = model.intercept_
theta1 = model.coef_[0]
answers['Q1'] = [theta0, theta1, mse]

### Question 2
num_exclamations = np.array([review['review_text'].count('!') for review in dataset]).reshape(-1, 1)
review_length = np.array([len(review['review_text']) for review in dataset]).reshape(-1, 1)

X = np.hstack((review_length, num_exclamations))
y = np.array([review['rating'] for review in dataset])

model = linear_model.LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)

theta0 = model.intercept_
theta1 = model.coef_[0]
theta2 = model.coef_[1]
answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)

from sklearn.preprocessing import PolynomialFeatures
### Question 3
mses = []

X = np.array([review['review_text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([review['rating'] for review in dataset])

for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = linear_model.LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    
    mse = mean_squared_error(y, y_pred)
    
    mses.append(mse)

answers['Q3'] = mses
assertFloatList(answers['Q3'], 5)# List of length 5

### Question 4
X = np.array([review['review_text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([review['rating'] for review in dataset])

split_index = len(X) // 2
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

mses = []
for degree in range(1, 6):
    poly = PolynomialFeatures(degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = linear_model.LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred_test = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred_test)
    
    mses.append(mse)

answers['Q4'] = mses
assertFloatList(answers['Q4'], 5)

from sklearn.metrics import mean_absolute_error
### Question 5
theta_0 = np.median(y_train)
y_pred_test = np.full_like(y_test, theta_0)

mae = mean_absolute_error(y_test, y_pred_test)

answers['Q5'] = mae
assertFloat(answers['Q5'])

### Question 6
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))
    
len(dataset)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X = np.array([review['review/text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([1 if review['user/gender'] == 'Female' else 0 for review in dataset])
model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
sens = TP / (TP + FN) 
spec = TN / (TN + FP)
BER = 1 - 0.5 * (sens + spec)

print(f"True Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Balanced Error Rate (BER): {BER}")

answers['Q6'] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q6'], 5)

### Question 7
X = np.array([review['review/text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([1 if review['user/gender'] == 'Female' else 0 for review in dataset])
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

y_pred = model.predict(X)

TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
sens = TP / (TP + FN) 
spec = TN / (TN + FP)
BER = 1 - 0.5 * (sens + spec)

print(f"True Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Balanced Error Rate (BER): {BER}")

answers["Q7"] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q7'], 5)

### Question 8
X = np.array([review['review/text'].count('!') for review in dataset]).reshape(-1, 1)
y = np.array([1 if review['user/gender'] == 'Female' else 0 for review in dataset])
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

y_prob = model.predict_proba(X)[:, 1]
sorted_indices = np.argsort(y_prob)[::-1]

K_values = [1, 10, 100, 1000, 10000]

# Calculate precision@K for each value of K
precisionList = []
for K in K_values:
    top_K_indices = sorted_indices[:K]
    y_top_K = y[top_K_indices]

    precision_at_K = np.sum(y_top_K == 1) / K
    precisionList.append(precision_at_K)

print(f"Precision@K for K={K_values}: {precisionList}")

answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5) #List of five floats

f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()