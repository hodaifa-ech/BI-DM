import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.cluster import KMeans


X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
print(f"Generated data shape: {X.shape}")


print("\nVisualizing raw generated data...")
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], s=50)
plt.title('Raw Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()


print("\nApplying Elbow Method...")
wcss = []
k_range = range(1, 11) 

for i in k_range:
  
    kmeans_elbow = KMeans(n_clusters=i,           
                     init='k-means++',     
                     max_iter=300,          
                     n_init=10,              
                     random_state=0)         
    kmeans_elbow.fit(X)
    wcss.append(kmeans_elbow.inertia_)


print("Plotting Elbow Method results...")
plt.figure(figsize=(8, 6))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


chosen_k = 4
print(f"\nApplying K-Means with k={chosen_k}...")
kmeans = KMeans(n_clusters=chosen_k, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(X)

print("Visualizing K-Means clustering results...")
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=pred_y, s=50, cmap='viridis') 


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X') 
plt.title(f'K-Means Clustering (k={chosen_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

print(f"\nCluster Centroids found by K-Means (k={chosen_k}):")
print(centers)
print("\nExperiment finished.")