import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# creating data from sklearn
data = make_blobs(n_samples=380, n_features=2, cluster_std=1.8, centers=4, random_state=101)
print(data)
# data
print(data[0])

# label of data above
print(data[1])

# making model
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

# centroid of clusters
print(kmeans.cluster_centers_)

# labels predicted
print(kmeans.labels_)

# comparing original and kmeans predicted labels in visual form

fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax_2.set_title('kmean')
ax_2.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')
ax_1.set_title('original')
ax_1.scatter(data[0][:, 0], data[0][:, 1], cmap='rainbow', c=data[1])
