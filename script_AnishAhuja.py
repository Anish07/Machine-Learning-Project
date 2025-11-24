import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the Dataset
df = pd.read_csv('/Volumes/Uni/Code/CPS803/assignment4/Mall_Customers.csv')

print(df.head())
print(df.tail())
print(df.shape)

# Clustering on 2 features
df1 = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scatterplot
sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatterplot of Annual Income vs Spending Score')
plt.show()

# Scatterplot with Gender
df3 = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]

sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df3)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatterplot of Annual Income vs Spending Score')
plt.show()

# Scatterplot with Age
df4 = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Age', data=df4)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatterplot of Annual Income vs Spending Score')
plt.show()

# Clustering Using Kmeans
errors = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(df1)
    errors.append(kmeans.inertia_)

# Plotting The Results
plt.figure(figsize=(13, 6))
plt.plot(range(1, 11), errors)
plt.plot(range(1, 11), errors, linewidth=3, color='red', marker='8')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE of Clusters Found')
plt.xticks(np.arange(1, 11, 1))
plt.show()

# Clustering using K-means
km = KMeans(n_clusters=5, n_init=10)
km.fit(df1)
y = km.predict(df1)
df1 = df1.copy()
df1.loc[:, 'Label'] = y

# Plotting the scatterplot with K-means
sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1, hue='Label', s=50, palette=['red', 'green', 'purple', 'blue', 'orange'])
plt.title('Clustering using K-means')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(df1, df['Cluster'])
print(f"K-Means Silhouette Score: {silhouette_avg}")

# Preprocessing is necessary for DBSCAN
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df1)

# Create DBSCAN object
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit and predict clusters
df['Cluster'] = dbscan.fit_predict(scaled_features)

# Plot the scatter plot with clusters using DBSCAN
sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=df)
plt.title('DBSCAN Clustering: Spending Score vs. Annual Income')
plt.show()

# Print Silhouette Score
silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print(f"DBSCAN Silhouette Score: {silhouette_avg}")

# Specify the number of clusters
n_clusters = 5

# Create Agglomerative Clustering object (Hierarchical)
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)

# Fit and predict clusters
df['Cluster'] = agg_clustering.fit_predict(scaled_features)

# Print Silhouette Score
silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print(f"Hierarchical Silhouette Score: {silhouette_avg}")

# Plot the scatter plot with clusters using hierarchical method
plt.figure(figsize=(10, 6))
sb.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Hierarchical Clustering: Spending Score vs. Annual Income')
plt.show()
