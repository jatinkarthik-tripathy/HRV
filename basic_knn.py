import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


DATA_PATH = "D:/Data/hrv dataset/data/final"
train_df = pd.read_csv(os.path.join(
    DATA_PATH, "train.csv")).drop(columns="datasetId")
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv")
                      ).drop(columns="datasetId")

X_train = train_df.drop(columns="condition")
y_train = train_df["condition"]

X_test = test_df.drop(columns="condition")
y_test = test_df["condition"]

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)

# print(f"acc:{knn.score(X_test, y_test)}")

def cust_dist(x, y):
    print(x)
    # mean_factor = float(len(x)**-1)
    # sum = 0.0
    # for i, j in zip(x, y):
    #     sum += float(abs((x**3-y**3)**0.5)/(x+y))
    # dist = sum*mean_factor
    return 0.0

knn = KNeighborsClassifier(n_neighbors=5, metric=cust_dist)
knn.fit(X_train, y_train)

print(f"acc:{knn.score(X_test, y_test)}")
