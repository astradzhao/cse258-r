import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
from sklearn import linear_model
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
bookUsers = defaultdict(set)
userBooks = defaultdict(set)

userInteractionCounts = defaultdict(int)
bookInteractionCounts = defaultdict(int)

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    bookUsers[b].add(u)
    userBooks[u].add(b)
    userInteractionCounts[u] += 1
    bookInteractionCounts[b] += 1
    

medianUserInteractions = np.median(list(userInteractionCounts.values()))
medianBookInteractions = np.median(list(bookInteractionCounts.values()))

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

negative_valid_samples = []

for user, book, _ in ratingsValid:
     negative_books = set()
     while len(negative_books) < 1:
        negative_book = random.choice(list(bookCount.keys()))
        if all(b != negative_book for b, _ in ratingsPerUser[user]) and negative_book not in negative_books:
            negative_books.add(negative_book)
            negative_valid_samples.append((user, negative_book, 0))  # 0 for negative sample

validation_with_negatives = [(u, b, 1) for u, b, _ in ratingsValid] + negative_valid_samples


### Read
def jaccardSim(book1, book2):
    users1 = bookUsers.get(book1, set())
    users2 = bookUsers.get(book2, set())
    if not users1 or not users2:
        return 0
    intersection = len(users1.intersection(users2))
    union = len(users1.union(users2))
    return intersection / union

def maxJaccardSimilarity(user, book):
    return max([jaccardSim(book, b_read) for b_read in userBooks[user] if b_read != book], default=0)

def avgJaccardSimilarity(user, book):
    ans = [jaccardSim(book, b_read) for b_read in userBooks[user] if b_read != book]
    return np.mean(ans) if ans else 0

def cosineSim(book1, book2):
    users1 = bookUsers.get(book1, set())
    users2 = bookUsers.get(book2, set())
    if not users1 or not users2:
        return 0
    intersection = len(users1.intersection(users2))
    magnitude1 = math.sqrt(len(users1))
    magnitude2 = math.sqrt(len(users2))
    return intersection / (magnitude1 * magnitude2)


def maxCosineSimilarity(user, book):
    return max([cosineSim(book, b_read) for b_read in userBooks[user]], default=0)

def avgCosineSimilarity(user, book):
    ans = [cosineSim(book, b_read) for b_read in userBooks[user] if b_read != book]
    return np.mean(ans) if ans else 0

import gzip
from collections import defaultdict
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

trainRecords = []

for u, b, _ in ratingsTrain:
    numBookInteractions = bookInteractionCounts.get(b, medianBookInteractions)
    numUserInteractions = userInteractionCounts.get(u, medianUserInteractions)
    maxJaccardSim = maxJaccardSimilarity(u, b)
    maxCosineSim = maxCosineSimilarity(u, b)
    avgJaccardSim = avgJaccardSimilarity(u, b)
    avgCosineSim = avgCosineSimilarity(u, b)
    isPopular = 1 if b in mostPopular else 0
    
    trainRecords.append({
        'user': u,
        'book': b,
        'book_popularity': numBookInteractions,
        'user_interactions': numUserInteractions,
        'is_popular': isPopular,
        'max_jaccard_sim': maxJaccardSim,
        'max_cosine_sim': maxCosineSim,
        'avg_jaccard_sim': avgJaccardSim,
        'avg_cosine_sim': avgCosineSim,
        'label': 1
    })
    
    while True:
        negBook = random.choice(list(bookCount.keys()))
        if negBook not in userBooks[u]:
            numBookInteractions = bookInteractionCounts.get(negBook, medianBookInteractions)
            numUserInteractions = userInteractionCounts.get(u, medianUserInteractions)
            maxJaccardSim = maxJaccardSimilarity(u, negBook)
            maxCosineSim = maxCosineSimilarity(u, negBook)
            avgJaccardSim = avgJaccardSimilarity(u, negBook)
            avgCosineSim = avgCosineSimilarity(u, negBook)
            isPopular = 1 if b in mostPopular else 0

            trainRecords.append({
                'user': u,
                'book': negBook,
                'book_popularity': numBookInteractions,
                'user_interactions': numUserInteractions,
                'is_popular': isPopular,
                'max_jaccard_sim': maxJaccardSim,
                'max_cosine_sim': maxCosineSim,
                'avg_jaccard_sim': avgJaccardSim,
                'avg_cosine_sim': avgCosineSim,
                'label': 0
            })
            break

trainDf = pd.DataFrame(trainRecords)

validRecords = []

for u, b, label in validation_with_negatives:
    numBookInteractions = bookInteractionCounts.get(b, medianBookInteractions)
    numUserInteractions = userInteractionCounts.get(u, medianUserInteractions)
    isPopular = 1 if b in mostPopular else 0
    maxJaccardSim = maxJaccardSimilarity(u, b)
    maxCosineSim = maxCosineSimilarity(u, b)
    avgJaccardSim = avgJaccardSimilarity(u, b)
    avgCosineSim = avgCosineSimilarity(u, b)

    validRecords.append({
        'user': u,
        'book': b,
        'book_popularity': numBookInteractions,
        'user_interactions': numUserInteractions,
        'is_popular': isPopular,
        'max_jaccard_sim': maxJaccardSim,
        'max_cosine_sim': maxCosineSim,
        'avg_jaccard_sim': avgJaccardSim,
        'avg_cosine_sim': avgCosineSim,
        'label': label
    })

validDf = pd.DataFrame(validRecords)

scaler = StandardScaler()

# Fill missing values and standardize features
featureCols = ['book_popularity', 'user_interactions', 'max_jaccard_sim', 'max_cosine_sim', 'avg_jaccard_sim', 'avg_cosine_sim']
trainDf[featureCols] = trainDf[featureCols].fillna(0)
validDf[featureCols] = validDf[featureCols].fillna(0)

trainDf[['book_popularity_scaled', 'user_interactions_scaled']] = scaler.fit_transform(trainDf[['book_popularity', 'user_interactions']])
validDf[['book_popularity_scaled', 'user_interactions_scaled']] = scaler.transform(validDf[['book_popularity', 'user_interactions']])

featureCols = ['book_popularity_scaled', 'user_interactions_scaled', 'is_popular', 'max_jaccard_sim', 'avg_cosine_sim']
trainFeatures = np.array(trainDf[featureCols])
validFeatures = np.array(validDf[featureCols])

model = LogisticRegression(max_iter=10000, C = 0.005)
model.fit(trainFeatures, trainDf['label'])

validProbs = model.predict_proba(validFeatures)[:, 1]
n_test = len(validProbs)

n_positive = n_test // 2
sorted_indices = np.argsort(-validProbs)

validPreds = np.zeros(n_test, dtype=int)
validPreds[sorted_indices[:n_positive]] = 1

validPreds = (validProbs >= 0.35).astype(int)

accuracy = accuracy_score(validDf['label'], validPreds)
auc = roc_auc_score(validDf['label'], validProbs)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation AUC: {auc:.4f}")


test_pairs = []

with open("pairs_Read.csv", 'r') as f:
    for l in f:
        if l.startswith("userID"):
            continue
        u, b = l.strip().split(',')
        test_pairs.append((u, b))

test_records = []

for u, b in test_pairs:
    # Book popularity from training data
    numBookInteractions = bookInteractionCounts.get(b, medianBookInteractions)
    numUserInteractions = userInteractionCounts.get(u, medianUserInteractions)
    isPopular = 1 if b in mostPopular else 0

    maxJaccardSim = maxJaccardSimilarity(u, b)
    maxCosineSim = maxCosineSimilarity(u, b)
    avgJaccardSim = avgJaccardSimilarity(u, b)
    avgCosineSim = avgCosineSimilarity(u, b)
    
    test_records.append({
        'user': u,
        'book': b,
        'book_popularity': numBookInteractions,
        'user_interactions': numUserInteractions,
        'is_popular': isPopular,
        'max_jaccard_sim': maxJaccardSim,
        'max_cosine_sim': maxCosineSim,
        'avg_jaccard_sim': avgJaccardSim,
        'avg_cosine_sim': avgCosineSim,
    })

test_df = pd.DataFrame(test_records)

test_df[['book_popularity_scaled', 'user_interactions_scaled']] = scaler.transform(test_df[['book_popularity', 'user_interactions']])
feature_cols = ['book_popularity_scaled', 'user_interactions_scaled', 'is_popular', 'max_jaccard_sim', 'avg_jaccard_sim']
test_df[feature_cols] = test_df[feature_cols].fillna(0)
test_features = test_df[feature_cols]

test_probs = model.predict_proba(test_features)[:, 1]
test_preds = (test_probs >= 0.35).astype(int)
# n_test = len(test_probs)

# n_positive = n_test // 2
# sorted_indices = np.argsort(-test_probs)

# test_preds = np.zeros(n_test, dtype=int)
# test_preds[sorted_indices[:n_positive]] = 1

with open("predictions_Read.csv", 'w') as predictions:
    predictions.write("userID,bookID,prediction\n")
    
    for idx, (u, b) in enumerate(test_pairs):
        prediction = test_preds[idx]
        predictions.write(f"{u},{b},{prediction}\n")



### Rating
from itertools import product
from surprise import Dataset, Reader, SVDpp, accuracy, SVD

df_train = pd.DataFrame(ratingsTrain, columns=['userID', 'itemID', 'rating'])
df_valid = pd.DataFrame(ratingsValid, columns=['userID', 'itemID', 'rating'])

reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)
trainset = train_data.build_full_trainset()

validset = list(zip(df_valid['userID'], df_valid['itemID'], df_valid['rating']))

# during training i tested different values here
param_grid = {
    'n_factors': [1],
    'n_epochs': [19],
    'lr_bu': [0.01],
    'lr_bi': [0.005],
    'lr_pu': [0.001],
    'lr_qi': [0.005],
    'reg_bu': [0.15],
    'reg_bi': [0.08],
    'reg_pu': [0.1],
    'reg_qi': [0.1],
}

# Generate all combinations of parameters
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = [dict(zip(param_names, v)) for v in product(*param_values)]

best_rmse = float('inf')
best_params = None

for params in param_combinations:
    algo_params = {k: v for k, v in params.items() if v is not None}
    algo_params['random_state'] = 420
    algo = SVD(**algo_params)
    algo.fit(trainset)
    predictions = algo.test(validset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"Parameters: {algo_params} => RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = algo_params

print(f"\nBest RMSE: {best_rmse:.4f} with parameters: {best_params}")

#Parameters: {'n_factors': 1, 'n_epochs': 20, 'lr_bu': 0.005, 'lr_bi': 0.005, 'lr_pu': 0.005, 'lr_qi': 0.001, 'reg_bu': 0.01, 'reg_bi': 0.05, 'reg_pu': 0.05, 'reg_qi': 0.01, 'random_state': 100} => RMSE: 1.1920

#best_pp_params = {'n_factors': 3, 'n_epochs': 20, 'lr_bu': 0.01, 'lr_bi': 0.006, 'lr_pu': 0.001, 'lr_qi': 0.005, 'reg_bu': 0.15, 'reg_bi': 0.1, 'reg_pu': 0.1, 'reg_qi': 0.1}
best_svd_params = {'n_factors': 1, 'n_epochs': 19, 'lr_bu': 0.01, 'lr_bi': 0.005, 'lr_pu': 0.001, 'lr_qi': 0.005, 'reg_bu': 0.15, 'reg_bi': 0.08, 'reg_pu': 0.1, 'reg_qi': 0.1, 'random_state': 139}
#algo = SVDpp(**best_params)
algo = SVD(**best_svd_params)
# Train the algorithm
algo.fit(trainset)

with open("pairs_Rating.csv", 'r') as pairs_file, open("predictions_Rating.csv", 'w') as predictions:
    for line in pairs_file:
        if line.startswith("userID"):
            predictions.write(line)
            continue
        userID, itemID = line.strip().split(',')

        # Predict the rating
        pred = algo.predict(userID, itemID)
        predRating = pred.est

        # Write the prediction
        predictions.write(f"{userID},{itemID},{predRating}\n")

