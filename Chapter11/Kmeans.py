import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)
beer = pd.read_csv(script_dir.replace("Chapter11","") + "data/beer.csv")
print(beer.head())

X = beer.drop('name', axis=1)
km = KMeans(n_clusters=3, random_state=1)
km.fit(X)
beer['cluster'] = km.labels_

beerCl = beer.groupby('cluster').mean()

print(beerCl.head())

centers = beer.groupby('cluster').mean()
colors = np.array(['red','green','blue','yellow'])
plt.scatter(beer.calories, beer.alcohol, c=colors[list(beer.cluster)], s=50)
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('caloeries')
plt.ylabel('alcohol')
plt.show()

#Silhouette
print(metrics.silhouette_score(X, km.labels_))
k_range = range(2,20)
scores = []
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_scaled)
    scores.append(metrics.silhouette_score(X_scaled, km.labels_))

plt.plot(k_range, scores)
plt.grid(True)
plt.xlabel("#clusters")
plt.ylabel("silhouette coeff.")
plt.show()