import numpy as np
from sklearn.cluster import KMeans as KMeans
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel('./Q1-data.xlsx')
data = np.array(file)
plt.scatter(data[:, 0], data[:, 1])

model_5 = KMeans(n_clusters=5)
model_10 = KMeans(n_clusters=10)
model_5.fit(data)
model_10.fit(data)
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(data[:, 0], data[:, 1], c=model_5.labels_)


obs = np.array([[0, 2], [-1, 3], [4, 0], [5, 1], [3, -1]])
centers = np.random.random([2, 2])
distance = np.zeros([5, 2])
labels = np.random.randint(0, 2, 5)
epochs = 10



def KM(epochs, centers, distance, labels, obs):
    for epoch in range(epochs):
        new_c1 = []
        new_c2 = []
        # calculate distance and update label
        for i in range(len(obs)):
            for j in range(len(centers)):
                dis = np.square(obs[i] - centers[j]).sum()
                distance[i][j] = dis
            labels[i] = distance[i].argmin()

        # update center
        for i in range(len(obs)):
            if labels[i] < 0.5:
                new_c1.append(obs[i])
            else:
                new_c2.append(obs[i])
        new_c1 = np.array(new_c1)
        new_c2 = np.array(new_c2)
        if len(new_c1) < 1:
            centers = np.array([centers[0], [new_c2[:, 0].mean(), new_c2[:, 1].mean()]])
        elif len(new_c2) < 1:
            centers = np.array([[new_c1[:, 0].mean(), new_c1[:, 1].mean()], centers[1]])
        else:
            centers = np.array([[new_c1[:, 0].mean(), new_c1[:, 1].mean()], [new_c2[:, 0].mean(), new_c2[:, 1].mean()]])
        return labels


# KM(epochs, centers, distance, labels, obs)
