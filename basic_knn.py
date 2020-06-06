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

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(f"acc:{knn.score(X_test, y_test)}")