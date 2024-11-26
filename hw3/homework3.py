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

answers = {}

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

# Copied from baseline code
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

### Question 1
negative_samples = []
for user, book, _ in ratingsValid:
    while True:
        negative_book = random.choice(list(bookCount.keys()))
        if all(b != negative_book for b, _ in ratingsPerUser[user]):
            negative_samples.append((user, negative_book, 0))  # 0 for negative sample
            break

validation_with_negatives = [(u, b, 1) for u, b, _ in ratingsValid] + negative_samples

correct_predictions = 0
for user, book, label in validation_with_negatives:
    prediction = 1 if book in return1 else 0
    if prediction == label:
        correct_predictions += 1

acc1 = correct_predictions / len(validation_with_negatives)
print(f"model accuracy on valid: {acc1}")

answers['Q1'] = acc1

### Question 2

threshold = None
acc2 = 0

for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    curr_threshold = totalRead * fraction

    popular_books = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        popular_books.add(i)
        if count > curr_threshold:
            break

    correct_predictions = 0
    for user, book, label in validation_with_negatives:
        prediction = 1 if book in popular_books else 0
        if prediction == label:
            correct_predictions += 1

    curr_acc = correct_predictions / len(validation_with_negatives)
    
    if curr_acc > acc2:
        acc2 = curr_acc
        threshold = fraction

print(f"Best threshold: {threshold}")
print(f"Best accuracy: {acc2}")

answers['Q2'] = [threshold, acc2]

## Question 3/4
def jaccard_similarity(book1, book2):
    users1 = bookUsers.get(book1, set())
    users2 = bookUsers.get(book2, set())
    if not users1 or not users2:
        return 0
    intersection = len(users1.intersection(users2))
    union = len(users1.union(users2))
    return intersection / union

best_threshold = None
best_accuracy = 0

for threshold in range(20, 40, 2):
    threshold = threshold / 10000
    correct_predictions = 0

    for user, book, label in validation_with_negatives:
        # find max jaccard sim between books
        max_jaccard_sim = max([jaccard_similarity(book, b_read) for b_read in userBooks[user]], default=0)
        
        # predict as read if above threshold
        prediction = 1 if max_jaccard_sim > threshold else 0

        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(validation_with_negatives)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

    print(threshold, accuracy)


print(f"Best Jaccard similarity threshold: {best_threshold}")
print(f"Accuracy with best Jaccard threshold: {best_accuracy}")

acc3 = best_accuracy

pop_threshold = None
pop_accuracy = 0

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

print(f"Best popularity fraction: {pop_threshold[0]}")
print(f"Best Jaccard similarity threshold: {pop_threshold[1]}")
print(f"Combined accuracy with best thresholds: {pop_accuracy}")

acc4 = pop_accuracy
answers['Q3'] = acc3
answers['Q4'] = acc4

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

answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

user_ids = set()
item_ids = set()
for u, b, r in ratingsTrain + ratingsValid:
    user_ids.add(u)
    item_ids.add(b)

user2index = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
item2index = {item_id: idx for idx, item_id in enumerate(sorted(item_ids))}
num_users = len(user2index)
num_items = len(item2index)

def prepare_data(data, user2index, item2index):
    user_indices = torch.tensor([user2index[u] for u, _, _ in data], dtype=torch.long)
    item_indices = torch.tensor([item2index[b] for _, b, _ in data], dtype=torch.long)
    ratings = torch.tensor([float(r) for _, _, r in data], dtype=torch.float)
    return user_indices, item_indices, ratings

train_users, train_items, train_ratings = prepare_data(ratingsTrain, user2index, item2index)
val_users, val_items, val_ratings = prepare_data(ratingsValid, user2index, item2index)

alpha = torch.nn.Parameter(torch.tensor(0.0))
beta_user = torch.nn.Parameter(torch.zeros(num_users))
beta_item = torch.nn.Parameter(torch.zeros(num_items))

optimizer = torch.optim.LBFGS([alpha, beta_user, beta_item])
lambda_reg = 1

def closure():
    optimizer.zero_grad()
    predictions = alpha + beta_user[train_users] + beta_item[train_items]
    errors = train_ratings - predictions
    loss = (errors ** 2).sum()
    reg_term = lambda_reg * (beta_user ** 2).sum() + lambda_reg * (beta_item ** 2).sum()
    total_loss = loss + reg_term
    total_loss.backward()
    return total_loss

optimizer.step(closure)

with torch.no_grad():
    val_predictions = alpha + beta_user[val_users] + beta_item[val_items]
    val_errors = val_ratings - val_predictions
    val_mse = (val_errors ** 2).mean()
    print("Validation MSE:", val_mse.item())

answers['Q6'] = float(val_mse)

### Question 7
beta_user_values = beta_user.detach()

idx_max = torch.argmax(beta_user_values).item()
idx_min = torch.argmin(beta_user_values).item()

index2user = {idx: user_id for user_id, idx in user2index.items()}

maxUser = index2user[idx_max]
minUser = index2user[idx_min]

maxBeta = beta_user_values[idx_max].item()
minBeta = beta_user_values[idx_min].item()

answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]

## Question 8
lambda_values = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
results = []

best_mse = float("inf")
best_lam = 0

for lambda_reg in lambda_values:
    print(f"\nTesting lambda = {lambda_reg}")
    alpha = torch.nn.Parameter(torch.tensor(0.0))
    beta_user = torch.nn.Parameter(torch.zeros(num_users))
    beta_item = torch.nn.Parameter(torch.zeros(num_items))
    
    optimizer = torch.optim.LBFGS([alpha, beta_user, beta_item])
    
    def closure():
        optimizer.zero_grad()
        predictions = alpha + beta_user[train_users] + beta_item[train_items]
        errors = train_ratings - predictions
        loss = (errors ** 2).sum()
        reg_term = lambda_reg * ((beta_user ** 2).sum() + (beta_item ** 2).sum())
        total_loss = loss + reg_term
        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)
    
    with torch.no_grad():
        val_predictions = alpha + beta_user[val_users] + beta_item[val_items]
        val_errors = val_ratings - val_predictions
        val_mse = (val_errors ** 2).mean().item()
        print(f"Validation MSE: {val_mse}")
    
    results.append({
        'lambda': lambda_reg,
        'val_mse': val_mse,
        'alpha': alpha.item(),
        'beta_user': beta_user.detach().clone(),
        'beta_item': beta_item.detach().clone()
    })

    if val_mse < best_mse:
        best_mse = val_mse
        best_lam = lambda_reg

print("\nLambda vs Validation MSE:")
for res in results:
    print(f"Lambda: {res['lambda']}, Validation MSE: {res['val_mse']}")

answers['Q8'] = (best_lam, best_mse)

lambda_reg = 5.0

alpha = torch.nn.Parameter(torch.tensor(0.0))
beta_user = torch.nn.Parameter(torch.zeros(num_users))
beta_item = torch.nn.Parameter(torch.zeros(num_items))

optimizer = torch.optim.LBFGS([alpha, beta_user, beta_item])

def closure():
    optimizer.zero_grad()
    predictions = alpha + beta_user[train_users] + beta_item[train_items]
    errors = train_ratings - predictions
    loss = (errors ** 2).sum()
    reg_term = lambda_reg * ((beta_user ** 2).sum() + (beta_item ** 2).sum())
    total_loss = loss + reg_term
    total_loss.backward()
    return total_loss

optimizer.step(closure)

alpha_value = alpha.detach().item()
beta_user_values = beta_user.detach()
beta_item_values = beta_item.detach()

predictions = open("predictions_Rating.csv", 'w')

with open("assignment1/pairs_Rating.csv", 'r') as pairs_file:
    for line in pairs_file:
        if line.startswith("userID"):
            predictions.write(line)
            continue
        userID, itemID = line.strip().split(',')

        if userID in user2index:
            user_idx = user2index[userID]
            beta_u = beta_user_values[user_idx].item()
        else:
            beta_u = 0.0

        if itemID in item2index:
            item_idx = item2index[itemID]
            beta_i = beta_item_values[item_idx].item()
        else:
            beta_i = 0.0
        pred_rating = alpha_value + beta_u + beta_i

        predictions.write(f"{userID},{itemID},{pred_rating}\n")
    
predictions.close()