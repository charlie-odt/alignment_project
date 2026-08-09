import pandas as pd
import numpy as np

# 1. Load the dataset without outliers (2)
K_train = np.load("../data/similarity_tab/training/similarity_BB20004_clean.npy")
n = K_train.shape[0]

# 2. 

# Compute the mean of the distance matrix (without outliers)
moy = np.mean(K_train)

# Compute the values of f for the sequences in the training set
f_train = [K_train[i, i] - 2 * (1/n) * np.sum(K_train[i, :]) + moy for i in range(K_train.shape[0])]

# 3. Compute the threshold for outlier detection
rad = 0 ### to be specified, not equal to 0
delta = 0.05  # set a value for delta
threshold = np.mean(f_train) + 2 * rad + np.sqrt(np.log(1 / delta) / (2 * n))

# 4. Identify outliers
K_outliers = np.load("../data/similarity_tab/training/similarity_BB20004.npy")

# Compute the values of f for the sequences in the dataset with outliers
index_outlier = 1
f_outlier = K_outliers[index_outlier, index_outlier] - 2 * (1/n) * np.sum(K_outliers[index_outlier, :]) + moy

# 5. Print the results
print(f_outlier)
print(threshold)
# Here f_outlier > threshold, so the sequence is considered an outlier.