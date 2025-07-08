import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN

import joblib
from itertools import accumulate

base_path = ''

profiles_df = pd.read_csv(base_path+'profiles.csv')
anime_df = pd.read_csv(base_path+'anime.csv')
rating_df = pd.read_csv(base_path+'rating.csv')


# Scaling profiles

scaler = StandardScaler()
X = profiles_df.sample(frac=1.0, random_state=42)[profiles_df.columns.values[1:]]
X_scaled = scaler.fit_transform(X)

scaled_profiles_df = pd.DataFrame(X_scaled, columns=profiles_df.columns.values[1:])
print(scaled_profiles_df.describe())
print()


# Experimenting Different Models

dbscan_model = DBSCAN(eps=1.5, min_samples=5)

labels = dbscan_model.fit_predict(pd.DataFrame(X_scaled).sample(40000, random_state=123))

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}, noise points: {n_noise}")

if n_clusters > 1:
    score = silhouette_score(pd.DataFrame(X_scaled).sample(40000, random_state=123), labels)
    print(f"Silhouette Score: {score:.4f}")


def plot_scores(min_k, max_k, inertias, silhouette_scores):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  ax1.plot(range(min_k, max_k+1), inertias, marker='o')
  ax1.set_title('Inertias | Elbow Method')

  ax2.plot(range(min_k, max_k+1), silhouette_scores, marker='o')
  ax2.set_title('Silhouette Scores')

  plt.tight_layout()

  plt.show()


min_k = 5
k = min_k

silhouette_scores = []
inertias = []

for k in range(min_k, 35):
  kmeans_model = KMeans(n_clusters=k) 
  labels = kmeans_model.fit_predict(X_scaled)

  inertias.append(kmeans_model.inertia_)
  silhouette_scores.append(silhouette_score(X_scaled, labels))

  print(f'Done cluster {k}')

  k+=1

plot_scores(min_k, 34, inertias, silhouette_scores)

# Best Model
best_model = KMeans(n_clusters=9, random_state=42)
labels = best_model.fit_predict(X_scaled)

print(silhouette_score(X_scaled, labels))


# PCA Applied Data

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
X_pca_scaled = scaler.fit_transform(X)
print(pca.explained_variance_ratio_.shape)
print()


def plot_explained_variance(pca):
    # This function graphs the accumulated explained variance ratio for a fitted PCA object.
    acc = [*accumulate(pca.explained_variance_ratio_)]
    fig, ax = plt.subplots(1, figsize=(30, 12))
    ax.stackplot(range(pca.n_components_), acc)
    ax.scatter(range(pca.n_components_), acc, color='black')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, pca.n_components_-1)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('N Components', fontsize=28)
    ax.set_ylabel('Accumulated explained variance', fontsize=28)
    plt.tight_layout()
    plt.show()

plot_explained_variance(pca)


pca_base_model = KMeans(n_clusters=14, random_state=42)
labels_pca = pca_base_model.fit_predict(X_pca_scaled)

print(silhouette_score(X_pca_scaled, labels_pca))

# Creating the PCA Dataframe with labels and user ids
X_pca_df = pd.DataFrame(X_pca_scaled, columns=[f'Anime Feature {i}' for i in range(0, 17)])

X_pca_df['labels']= labels_pca
X_pca_df['user_id'] = profiles_df['user_id']


# Assigning Labels

profiles_df = profiles_df.sample(frac=1.0, random_state=42)
profiles_df['labels'] = labels # or labels_pca

print(profiles_df.groupby('labels')['user_id'].count())

labeled_rating_df = pd.merge(rating_df, profiles_df[['user_id', 'labels']], on='user_id', how='left')

anime_views_per_label = labeled_rating_df.groupby(['labels', 'anime_id']).size().reset_index(name='views')


anime_views_per_label.to_csv(base_path+'anime_views_per_label_scaled.csv', index=False)

joblib.dump(best_model, base_path+'cluster_model.joblib')
joblib.dump(scaler, base_path+'scaler.joblib')

