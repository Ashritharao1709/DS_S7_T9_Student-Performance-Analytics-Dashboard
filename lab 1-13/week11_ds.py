
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------
# INPUT DATA
# ---------------------------------------------------
# Features:
# [Study Hours, Attendance Percentage]
X = np.array([
    [2, 50],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85],
    [9, 90]
])
# Labels:
# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
# ---------------------------------------------------
# TRAIN DECISION TREE MODEL
# ---------------------------------------------------
model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)
# ---------------------------------------------------
# TEST DATA
# ---------------------------------------------------
X_test = np.array([
    [3, 55],
    [5, 72],
    [8, 88],
    [4, 60]
])
# Actual outputs
y_test = np.array([0, 1, 1, 0])
# Predict outputs
y_pred = model.predict(X_test)
# ---------------------------------------------------
# PERFORMANCE EVALUATION
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------

print("Actual Labels:")
print(y_test)
print("\nPredicted Labels:")
print(y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:")
print(accuracy)
print("\nError Rate:")
print(error_rate)
print("\nPrecision:")
print(precision)
print("\nRecall:")
print(recall)
print("\nF1-Score:")
print(f1)

# K-Nearest Neighbors (KNN) Classification
# Performance Evaluation Metrics
# Application: Student Pass/Fail Prediction
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# ---------------------------------------------------
# TRAINING DATASET
# ---------------------------------------------------
# Features:
# [Study Hours, Attendance Percentage]
X_train = np.array([
    [2, 50],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85],
    [9, 90]
])
# Labels:
# 0 = Fail
# 1 = Pass
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])
# ---------------------------------------------------
# CREATE KNN MODEL
# ---------------------------------------------------
# k = 3 nearest neighbors
model = KNeighborsClassifier(n_neighbors=3)
# Train model
model.fit(X_train, y_train)
# ---------------------------------------------------
# TEST DATA
# ---------------------------------------------------
X_test = np.array([
    [3, 55],
    [5, 72],
    [8, 88],
    [4, 60]
])
# Actual outputs
y_test = np.array([0, 1, 1, 0])
# Predict outputs
y_pred = model.predict(X_test)
# ---------------------------------------------------
# PERFORMANCE EVALUATION
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
print("Actual Labels:")
print(y_test)
print("\nPredicted Labels:")
print(y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:")
print(accuracy)
print("\nError Rate:")
print(error_rate)
print("\nPrecision:")
print(precision)
print("\nRecall:")
print(recall)
print("\nF1-Score:")
print(f1)