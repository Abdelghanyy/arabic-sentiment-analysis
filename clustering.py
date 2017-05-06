from sklearn import cluster
dataset=[[False,True,False],[False,True,False],[True,False,False],[True,False,True]]
target=["pos","pos","neg"]
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(dataset)
print(k_means.labels_)
print(k_means.predict([[True,True,True]]))