import json
import gzip
import math
import numpy as np
from collections import defaultdict
from sklearn import linear_model
import random
import statistics

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

answers = {}

# From https://cseweb.ucsd.edu/classes/fa24/cse258-b/files/steam.json.gz
z = gzip.open("/Users/danielzhao/Documents/GitHub/cse258-r/midterm/steam.json.gz")

dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

z.close()

### Question 1
def MSE(y, ypred):
    return ((y - ypred) ** 2).mean()

X = [[len(d['text'])] for d in dataset]
y = [d['hours'] for d in dataset]

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X, y)

theta_1 = model.coef_[0]
y_pred = model.predict(X)

mse1 = MSE(y, y_pred)

answers['Q1'] = [float(theta_1), float(mse1)] # Remember to cast things to float rather than (e.g.) np.float64

### Question 2
dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]

X_train = [[len(d['text'])] for d in dataTrain]
y_train = [d['hours'] for d in dataTrain]

X_test = [[len(d['text'])] for d in dataTest]
y_test = [d['hours'] for d in dataTest]

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mse_test = MSE(y_test, y_pred_test)

under = 0
over = 0

for y_i, y_pred_i in zip(y_test, y_pred_test):
    if y_pred_i < y_i:
        under += 1
    elif y_pred_i > y_i:
        over += 1

mse2 = MSE(y_test, y_pred_test)
answers['Q2'] = [float(mse2), under, over]

### Question 3

# 3a
y2 = y[:]
y2.sort()
perc90 = y2[int(len(y2)*0.9)] # 90th percentile
X3a = [x for x, y in zip(X_train, y_train) if y <= perc90]
y3a = [y for y in y_train if y <= perc90]

mod3a = linear_model.LinearRegression(fit_intercept=True)
mod3a.fit(X3a,y3a)
pred3a = mod3a.predict(X_test)

under3a = 0
over3a = 0

for y_i, y_pred_i in zip(y_test, pred3a):
    if y_pred_i < y_i:
        under3a += 1
    elif y_pred_i > y_i:
        over3a += 1

# 3b
y_train3b =  [d['hours_transformed'] for d in dataTrain]
y_test3b =  [d['hours_transformed'] for d in dataTest]
model_b = linear_model.LinearRegression()
model_b.fit(X_train, y_train3b)

pred3b = model_b.predict(X_test)

under3b, over3b = 0, 0
for y_i, y_pred_i in zip(y_test3b, pred3b):
    if y_pred_i < y_i:
        under3b += 1
    elif y_pred_i > y_i:
        over3b += 1

# 3c
median_review_length = np.median([len(d['text']) for d in dataTrain])
median_hours = np.median(y_train)

theta_0c = model.intercept_
theta_1c = (median_hours - theta_0c) / median_review_length

pred3c = [theta_0c + theta_1c * x[0] for x in X_test]
under3c, over3c = 0, 0

for yi, y_pred_i in zip(y_test, pred3c):
    if y_pred_i < yi:
        under3c += 1
    elif y_pred_i > yi:
        over3c += 1

answers['Q3'] = [under3a, over3a, under3b, over3b, under3c, over3c]

### Question 4
from sklearn.metrics import confusion_matrix

y_train4 = [1 if y > median_hours else 0 for y in y_train]
y_test4 = [1 if y > median_hours else 0 for y in y_test]

mod = linear_model.LogisticRegression(C=1)
mod.fit(X_train,y_train4)
predictions4 = mod.predict(X_test)
TN, FP, FN, TP = confusion_matrix(y_test4, predictions4).ravel()

fpr = FP / (FP + TN)
fnr = FN / (FN + TP)
BER = (fpr + fnr) / 2

answers['Q4'] = [int(TP), int(TN), int(FP), int(FN), float(BER)]

### Question 5
answers['Q5'] = [int(FP), int(FN)]

### Question 6
def compute_ber(X_train, y_train, X_test, y_test):
    model = linear_model.LogisticRegression(C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    ber = (fpr + fnr) / 2
    return ber

X2014 = [[len(d['text'])] for d in dataTrain if int(d['date'][:4]) <= 2014]
y2014 = [1 if d['hours'] > median_hours else 0 for d in dataTrain if int(d['date'][:4]) <= 2014]

X2014test = [[len(d['text'])] for d in dataTest if int(d['date'][:4]) <= 2014]
y2014test = [1 if d['hours'] > median_hours else 0 for d in dataTest if int(d['date'][:4]) <= 2014]

X2015 = [[len(d['text'])] for d in dataTrain if int(d['date'][:4]) >= 2015]
y2015 = [1 if d['hours'] > median_hours else 0 for d in dataTrain if int(d['date'][:4]) >= 2015]

X2015test = [[len(d['text'])] for d in dataTest if int(d['date'][:4]) >= 2015]
y2015test = [1 if d['hours'] > median_hours else 0 for d in dataTest if int(d['date'][:4]) >= 2015]

BER_A = compute_ber(X2014, y2014, X2014test, y2014test)
BER_B = compute_ber(X2015, y2015, X2015test, y2015test)
BER_C = compute_ber(X2014, y2014, X2015test, y2015test)
BER_D = compute_ber(X2015, y2015, X2014test, y2014test)
answers['Q6'] = [float(BER_A), float(BER_B), float(BER_C), float(BER_D)]

### Question 7
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(dict)
reviewsPerItem = defaultdict(dict)

for d in dataTrain:
    user = d['userID']
    item = d['gameID']
    review = d['hours_transformed']
    year = int(d['date'][:4])

    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user][item] = review
    reviewsPerItem[item][user] = review

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

first_user = dataTrain[0]['userID']
first_user_items = itemsPerUser[first_user]

similarities = []
for user, items in itemsPerUser.items():
    if user != first_user:
        s = jaccard_similarity(first_user_items, items)
        similarities.append((user, s))

similarities.sort(key=lambda x: x[1], reverse=True)
top_10_similar = similarities[:10]

first = top_10_similar[0][1]
tenth = top_10_similar[9][1]

answers['Q7'] = [first, tenth]

### Question 8
avg_hrs = np.mean([d['hours_transformed'] for d in dataTrain])
def predict_hrs_user(user, item):
    similar_users = usersPerItem[item]
    num = 0
    den = 0
    for i in similar_users:
        if i != user:
            similarity = jaccard_similarity(itemsPerUser[user], itemsPerUser[i])
            num += reviewsPerUser[i][item] * similarity
            den += similarity
    return num / den if den != 0 else avg_hrs

def predict_hrs_item(user, item):
    similar_items = itemsPerUser[user]
    num = 0
    den = 0
    for i in similar_items:
        if i != item:
            similarity = jaccard_similarity(usersPerItem[item], usersPerItem[i])
            num += reviewsPerItem[i][user] * similarity
            den += similarity
    return num / den if den != 0 else avg_hrs

def calculate_mse(predictor, test_data):
    squared_errors = []
    for d in test_data:
        user, item, true_value = d['userID'], d['gameID'], d['hours_transformed']
        predicted_value = predictor(user, item)
        squared_errors.append((true_value - predicted_value) ** 2)
    return np.mean(squared_errors)

MSEU = calculate_mse(predict_hrs_user, dataTest)
MSEI = calculate_mse(predict_hrs_item, dataTest)

answers['Q8'] = [float(MSEU), float(MSEI)]

### Question 9
reviewYearUserItem = defaultdict(dict)
reviewYearItemUser = defaultdict(dict)
for d in dataset:
    user = d['userID']
    item = d['gameID']
    year = int(d['date'][:4])
    reviewYearUserItem[user][item] = year
    reviewYearItemUser[item][user] = year

def predict_hrs_user_time(user, item):
    similar_users = usersPerItem[item]
    target_year = reviewYearUserItem[user][item]
    
    numerator = 0
    denominator = 0
    for i in similar_users:
        if i != user:
            similarity = jaccard_similarity(itemsPerUser[user], itemsPerUser[i])
            year_diff = abs(target_year - reviewYearUserItem[i][item])
            time_weight = np.exp(-year_diff)
            numerator += reviewsPerUser[i][item] * similarity * time_weight
            denominator += similarity * time_weight
    
    return numerator / denominator if denominator != 0 else avg_hrs

MSE9 = calculate_mse(predict_hrs_user_time, dataTest)

answers['Q9'] = float(MSE9)