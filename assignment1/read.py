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
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    bookUsers[b].add(u)
    userBooks[u].add(b)

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

negative_samples = []
for user, book, _ in ratingsValid:
    while True:
        negative_book = random.choice(list(bookCount.keys()))
        if all(b != negative_book for b, _ in ratingsPerUser[user]):
            negative_samples.append((user, negative_book, 0))  # 0 for negative sample
            break

validation_with_negatives = [(u, b, 1) for u, b, _ in ratingsValid] + negative_samples


## Question 3/4
def jaccard_similarity(book1, book2):
    users1 = bookUsers.get(book1, set())
    users2 = bookUsers.get(book2, set())
    if not users1 or not users2:
        return 0
    intersection = len(users1.intersection(users2))
    union = len(users1.union(users2))
    return intersection / union

for fraction in [0.68, 0.7, 0.72, 0.74]:
    curr_pop_threshold = totalRead * fraction

    popular_books = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popular_books.add(i)
        if count > curr_pop_threshold:
            break

    for jaccard_threshold in range(0, 10):
        jaccard_threshold = jaccard_threshold / 100
        correct_predictions = 0

        for user, book, label in validation_with_negatives:
            if book in popular_books:
                prediction = 1
            else:
                max_jaccard_sim = max([jaccard_similarity(book, b_read) for b_read in userBooks[user]], default=0)
                prediction = 1 if max_jaccard_sim > jaccard_threshold else 0
            
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / len(validation_with_negatives)

        if accuracy > pop_accuracy:
            pop_accuracy = accuracy
            pop_threshold = (fraction, jaccard_threshold)

        print(f"Popularity fraction: {fraction}, Jaccard threshold: {jaccard_threshold}, Accuracy: {accuracy}")

predictions = open("predictions_Read.csv", 'w')

curr_pop_threshold = totalRead * 0.72
jaccard_threshold = 0.05

# Select popular books based on threshold
popular_books = set()
count = 0
for ic, i in mostPopular:
    count += ic
    popular_books.add(i)
    if count > curr_pop_threshold:
        break

for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')

    if b in popular_books:
        prediction = 1
    else:
        max_jaccard_sim = max([jaccard_similarity(b, b_read) for b_read in userBooks[u]], default=0)
        prediction = 1 if max_jaccard_sim > jaccard_threshold else 0
    
    predictions.write(u + ',' + b + ',' + str(prediction) + '\n')

predictions.close()
