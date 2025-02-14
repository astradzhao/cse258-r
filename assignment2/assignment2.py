import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from itertools import product
from surprise import Dataset, Reader, SVD, accuracy
import pandas as pd

# Load the dataset
file_path = '/content/drive/My Drive/Assignment 2/rating-Hawaii.csv.gz'
df = pd.read_csv(file_path)

# Split into training and temp (for validation + test)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

# Split temp into validation and test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
test_df_baseline = test_df.copy()

# Print sizes to verify
print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))

# Baseline
globalAverage = train_df['rating'].mean()

userAverage = train_df.groupby('user')['rating'].mean().to_dict()

predictions = []

for _, row in test_df_baseline.iterrows():
    user = row['user']
    business = row['business']
    if user in userAverage:
        predictions.append(userAverage[user])
    else:
        predictions.append(globalAverage)

test_df_baseline['prediction'] = predictions
rmse = np.sqrt(mean_squared_error(test_df_baseline['rating'], test_df_baseline['prediction']))
print("RMSE:", rmse)

reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(train_df[['user', 'business', 'rating']], reader)
trainset = train_data.build_full_trainset()

validset = list(zip(val_df['user'], val_df['business'], val_df['rating']))

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

param_names = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = [dict(zip(param_names, v)) for v in product(*param_values)]

best_rmse = float('inf')
best_params = None

# Hyperparameter tuning
for params in param_combinations:
    algo_params = {k: v for k, v in params.items() if v is not None}
    algo_params['random_state'] = 42
    algo = SVD(**algo_params)
    algo.fit(trainset)
    predictions = algo.test(validset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"Parameters: {algo_params} => RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = algo_params

print(f"\nBest RMSE: {best_rmse:.4f} with parameters: {best_params}")

algo = SVD(**best_params)
algo.fit(trainset)

predictions = [
    algo.predict(row['user'], row['business']).est for _, row in test_df.iterrows()
]

test_df['prediction'] = predictions

if 'rating' in test_df.columns:
    rmse = np.sqrt(mean_squared_error(test_df['rating'], test_df['prediction']))
    print(f"Test RMSE: {rmse:.4f}")

# Save predictions to a CSV file
#test_df[['user', 'business', 'prediction']].to_csv("predictions_Rating.csv", index=False)
