import pylab
from matplotlib import pyplot
import matplotlib as mpl
from pylab import *
from numpy import *
import pandas as pd
import sklearn	
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def reading_dataset(filename):
	labels = []
	dataset=[]
	
	file = open(filename, 'r')
	i=0
	for line in file :
		line=file.readline()
		labels.append(line[-4:])
		labels[i]=labels[i][0:3]
		dataset.append(line[0:len(line)-5])
		i=i+1
	return dataset,labels

sets=reading_dataset("/Users/ahmed/Desktop/Bachelor/ASTD-master/data/tweets.txt")
dataset=sets[0]
labels=sets[1]
training_labels=labels[:3001]
testing_labels=labels[-2003:]
training_dataset=dataset[:3001]
testing_dataset=dataset[-2003:]
vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_dataset)
test_vectors = vectorizer.transform(testing_dataset)
kernels=["linear","poly","rbf","sigmoid"]
kernels_score=[]
i=0
print ("Kernel Functions :")
for i in range(0,len(kernels)):
	svc_kernel = OneVsRestClassifier(sklearn.svm.SVC(kernel=kernels[i]))
	svc_kernel.fit(train_vectors,training_labels)
	kernel_score = svc_kernel.score(test_vectors, testing_labels)
	kernels_score.append(kernel_score)
	print(kernels[i],"-->score:",kernel_score)
print("kernels :" ,kernels)
print("socres :",kernels_score)
