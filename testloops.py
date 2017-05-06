import pylab
from matplotlib import pyplot
import matplotlib as mpl
from pylab import *
from numpy import *
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
def reading_features(filename):
	dataset=[]
	file = open(filename, 'r',encoding='utf-8')
	i=0
	for line in file :
		dataset.append(line[:-1])
	return dataset

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

features=reading_features("/Users/ahmed/Desktop/Bachelor/besmallah.txt")
features=features[:-1]
sets=reading_dataset("/Users/ahmed/Desktop/Bachelor/ASTD-master/data/tweets.txt")
tweets=sets[0]
labels=sets[1]
data=[]
for tweet in tweets:
	flags=[]
	for feature in features:
		if feature in tweet:
			flags.append(True)
		else:
			flags.append(False)
	data.append(flags)
training_labels=labels[:3001]
testing_labels=labels[-2003:]
training_dataset=data[:3001]
testing_dataset=data[-2003:]
classfier= OneVsRestClassifier(GaussianNB())
classfier.fit(training_dataset,training_labels)
score = classfier.score(testing_dataset, testing_labels)
print(score)
x=get_params(classfier)
print(x)
classfier2= OneVsRestClassifier(MultinomialNB())
classfier2.fit(training_dataset,training_labels)
score2 = classfier.score(testing_dataset, testing_labels)
print(score2)
classfier3= OneVsRestClassifier(BernoulliNB())
classfier3.fit(training_dataset,training_labels)
score3 = classfier.score(testing_dataset, testing_labels)
print(score3)
