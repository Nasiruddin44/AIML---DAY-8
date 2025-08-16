# Task 8: Clustering with K-Means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# Let's use only numerical features for clustering
X = df.select_dtypes(include=[np.number])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method to find optimal K
wcss = []  # within-cluster sum of squares
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()

# 3. Fit K-Means with chosen K (commonly k=5 for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = labels

# 4. Evaluate Clustering with Silhouette Score
sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score)

# 5. Visualization with PCA (2D projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
            cmap="viridis", s=50, alpha=0.7)
centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers)  # project centroids into 2D
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering (PCA reduced 2D)")
plt.legend()
plt.show()

# Save clustered data
df.to_csv("Mall_Customers_with_clusters.csv", index=False)
print("Clustered dataset saved as Mall_Customers_with_clusters.csv")
