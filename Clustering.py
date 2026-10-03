import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

data = np.array([
    [1, 2],
    [2, 3],
    [3, 2],
    [8, 9],
    [9, 8],
    [10, 9],
    [30, 30]
])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

print("K-Means:")
print(kmeans.labels_)

hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical.fit(data)

print("Hierarchical Clustering:")
print(hierarchical.labels_)

dbscan = DBSCAN(eps=2, min_samples=2)
dbscan.fit(data)

print("DBSCAN:")
print(dbscan.labels_)