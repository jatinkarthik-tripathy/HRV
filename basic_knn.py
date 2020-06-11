from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# DATA_PATH = "D:/Data/hrv dataset/data/final"
# train_df = pd.read_csv(os.path.join(
#     DATA_PATH, "train.csv")).drop(columns="datasetId")
# test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv")
#                       ).drop(columns="datasetId")

# X_train = train_df.drop(columns="condition")
# y_train = train_df["condition"]

# X_test = test_df.drop(columns="condition")
# y_test = test_df["condition"]

# print(X_train.shape)

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)

# print(f"acc:{knn.score(X_test, y_test)}")


data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2)


def cust_dist(x, y):
    print(len(x), len(y))
    print(x, y)
    # mean_factor = float(len(x)**-1)
    # sum = 0.0
    # for i, j in zip(x, y):
    #     sum += float(abs((x**3-y**3)**0.5)/(x+y))
    # dist = sum*mean_factor
    return 0.0

xx = X_train[:15]
yy = y_train[:15]
print(xx[1])
knn = KNeighborsClassifier(n_neighbors=5, metric=cust_dist)
knn.fit(xx, yy)
