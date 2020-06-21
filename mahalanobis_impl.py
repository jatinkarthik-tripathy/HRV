from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric


DATA_PATH = "D:/Data/hrv dataset/data/final"
train_df = pd.read_csv(os.path.join(
    DATA_PATH, "train.csv")).drop(columns="datasetId")
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv")
                      ).drop(columns="datasetId")

X_train = train_df.drop(columns="condition")
y_train = train_df["condition"]

X_test = test_df.drop(columns="condition")
y_test = test_df["condition"]

print(X_train.shape)

knn = KNeighborsClassifier(metric='mahalanobis',
                           metric_params={'V': np.cov(X_train)})
knn.fit(X_train, y_train)
print(f"acc:{knn.score(X_test, y_test)}")
