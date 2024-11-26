import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = open("5year.arff", 'r')
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

answers = {} # Your answers

# Question 1
def accuracy(predictions, y):
    correct = np.sum(predictions == y)
    total = len(y)
    return correct / total

def BER(predictions, y):
    classes = np.unique(y)
    error_rates = []
    
    for cls in classes:
        idx = (y == cls)
        true_positive = np.sum((predictions == cls) & (y == cls))
        
        total_class = np.sum(idx)

        error_rate = 1 - (true_positive / total_class)
        error_rates.append(error_rate)
    
    return np.mean(error_rates)

mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

acc1 = accuracy(pred, y)
ber1 = BER(pred, y)

answers['Q1'] = [float(acc1), float(ber1)] # Accuracy and balanced error rate

# Question 2
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

acc2 = accuracy(pred, y)
ber2 = BER(pred, y)
answers['Q2'] = [float(acc2), float(ber2)]

# Question 3
random.seed(3)
random.shuffle(dataset)
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)
predTrain = mod.predict(Xtrain)
predValid = mod.predict(Xvalid)
predTest = mod.predict(Xtest)
berTrain = BER(predTrain, ytrain)
berValid = BER(predValid, yvalid)
berTest = BER(predTest, ytest)
answers['Q3'] = [float(berTrain), float(berValid), float(berTest)]


# Question 4
cList = [10**-4, 10**-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]
berList = []
for c in cList:
    mod = linear_model.LogisticRegression(C=c, class_weight='balanced')
    mod.fit(Xtrain,ytrain)
    predValid = mod.predict(Xvalid)
    berValid = BER(predValid, yvalid)
    berList.append(float(berValid))

answers['Q4'] = berList

# Question 5
bestC = 100
mod = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod.fit(Xtrain,ytrain)
predTest = mod.predict(Xtest)
ber5 = float(BER(predTest, ytest))
answers['Q5'] = [bestC, ber5]

# Question 6
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

dataTrain = dataset[:9000]
dataTest = dataset[9000:]

# Some data structures you might want
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user = d['user_id']
    item = d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = d['rating']

def Jaccard(s1, s2):
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    return intersection / union

def mostSimilar(i, N):
    similarities = []
    users_for_item_i = usersPerItem[i]
    
    for other_item in usersPerItem:
        if other_item == i:
            continue
        users_for_other_item = usersPerItem[other_item]
        similarity = Jaccard(users_for_item_i, users_for_other_item)
        similarities.append((similarity, other_item))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:N]

similar_items = mostSimilar('2767052', 10)

answers['Q6'] = similar_items

# Question 7
avgRatingPerItem = {}

for item in usersPerItem:
    ratings = [ratingDict[(user, item)] for user in usersPerItem[item]]
    avgRatingPerItem[item] = sum(ratings) / len(ratings) if ratings else 0

def predictRating(user, item):
    numerator = 0
    denominator = 0
    if item not in avgRatingPerItem:
        return 0  # If no rating exists for the item, return 0

    avg_item_rating = avgRatingPerItem[item]
    for other_item in itemsPerUser[user]:
        if other_item == item:
            continue
        
        # Similarity between the current item and other items the user has rated
        similarity = Jaccard(usersPerItem[item], usersPerItem[other_item])
        
        if similarity > 0:
            other_item_rating = ratingDict[(user, other_item)]
            avg_other_item_rating = avgRatingPerItem[other_item]
            
            numerator += (other_item_rating - avg_other_item_rating) * similarity
            denominator += similarity
    
    if denominator == 0:
        return avg_item_rating  # If no similar items, return the average rating for the item
    
    return avg_item_rating + (numerator / denominator)

def computeMSE():
    true_ratings = []
    predicted_ratings = []
    
    for d in dataTest:
        user = d['user_id']
        item = d['book_id']
        true_rating = d['rating']
        
        # Predict the rating
        predicted_rating = predictRating(user, item)
        
        true_ratings.append(true_rating)
        predicted_ratings.append(predicted_rating)
    
    # Calculate the MSE
    mse = mean_squared_error(true_ratings, predicted_ratings)
    return mse

mse7 = float(computeMSE())
answers['Q7'] = mse7

# Question 8
def predictRatingUserBased(user, item):
    numerator = 0
    denominator = 0
    if item not in avgRatingPerItem:
        return 0  # If no rating exists for the item, return 0

    avg_item_rating = avgRatingPerItem[item]
    
    for other_user in usersPerItem[item]:
        if other_user == user:
            continue
        
        # Similarity between the current user and other users who rated the item
        similarity = Jaccard(itemsPerUser[user], itemsPerUser[other_user])
        
        if similarity > 0:
            other_user_rating = ratingDict[(other_user, item)]
            
            numerator += (other_user_rating - avg_item_rating) * similarity
            denominator += similarity
    
    if denominator == 0:
        return avg_item_rating  # If no similar users, return the average rating for the item
    
    return avg_item_rating + (numerator / denominator)

def computeMSEUserBased():
    true_ratings = []
    predicted_ratings = []
    
    for d in dataTest:
        user = d['user_id']
        item = d['book_id']
        true_rating = d['rating']
        
        # Predict the rating using user-based similarity
        predicted_rating = predictRatingUserBased(user, item)
        
        true_ratings.append(true_rating)
        predicted_ratings.append(predicted_rating)
    
    # Calculate the MSE
    mse = mean_squared_error(true_ratings, predicted_ratings)
    return mse

mse8 = float(computeMSEUserBased())
answers['Q8'] = mse8

f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()