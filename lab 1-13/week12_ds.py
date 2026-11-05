
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
# ---------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------
# Generate customer dataset
# Features:
# [Annual Income, Spending Score]
X, y = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.5,
    random_state=42
)
# ---------------------------------------------------
# K-MEANS CLUSTERING
# ---------------------------------------------------
print("----- K-Means Clustering -----")
# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
# Train model
kmeans.fit(X)
# Cluster labels
kmeans_labels = kmeans.labels_
# Cluster centers
centers = kmeans.cluster_centers_
print("Cluster Labels:")
print(kmeans_labels)
print("\nCluster Centers:")
print(centers)
# Plot K-Means Clusters
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
# ---------------------------------------------------
# HIERARCHICAL CLUSTERING
# ---------------------------------------------------
print("\n----- Hierarchical Clustering -----")
# Generate linkage matrix
linked = linkage(X, method='ward')
# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()