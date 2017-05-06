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
from collections import OrderedDict
from operator import itemgetter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from textblob.classifiers import NaiveBayesClassifier
#reading dataset and remove the new line and sperate the tweets from the labels
#takes the filepath and returns list of tweets and list of labels 
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
def reading_stopwords(filename):
	dataset=[]
	file = open(filename, 'r',encoding='utf-8')
	i=0
	for line in file :
		dataset.append(line[:-1])
	return dataset

def remove_hashtag (words):
	new_words_without_hashtags=[]
	for word in words:
		if "#" in word:
			hashtag_word=word.split("#")
			if "_" not in hashtag_word[1]:
				new_words_without_hashtags.append(hashtag_word[1])
			else:
				two_words=hashtag_word[1].split("_")
				for new_word in two_words:
					new_words_without_hashtags.append(new_word)
		else:
			new_words_without_hashtags.append(word)

	return new_words_without_hashtags

def remove_punctuation(words):
	new_words_without_punctuation=[]
	for word in words:
		if "،" in word:
			new_word=word.split("،")
			new_words_without_punctuation.append(new_word[0])
		else:
			new_words_without_punctuation.append(word)

	return new_words_without_punctuation

def clean_tweet(tweet):
	remove_spaces=tweet.split(" ")
	new_words_without_hashtags=remove_hashtag(remove_spaces)
	cleaned=remove_punctuation(new_words_without_hashtags)
	return cleaned

def frequency_calc (tweets):
	words={}
	for tweet in tweets:
		cleaned_tweet=clean_tweet(tweet)
		for word in cleaned_tweet:
			flag=True
			for key in words:
				if word==key:
					words[key]+=1
					flag=False
					break
			if flag:
				words[word]=1
	return words
def frequently_used (words):
	sorted_words = [(k, words[k]) for k in sorted(words, key=words.get, reverse=True)]
	for k, v in sorted_words:
		k, v

	return sorted_words

def with_stopwords(tweets):
	words=frequency_calc(tweets)
	sorted_words=frequently_used(words)
	return sorted_words

def without_stopwords(tweets):
	words=frequency_calc(tweets)
	stopwords=reading_stopwords("/Users/ahmed/Desktop/Bachelor/stopwords.txt")
	for word in words:
		if word in stopwords:
			words[word]=-1
	sorted_words=frequently_used(words)
	return sorted_words	

def class_NB(tweets,labels,words,limit):
	most_frequently_used=words[:limit]
	features=[]
	for tweet in tweets:
		flags=[]
		for word in most_frequently_used:
			if word[0] in tweet:
				flags.append(True)
			else:
				flags.append(False)
		features.append(flags)
	training_labels=labels[:4001]
	testing_labels=labels[-1003:]
	training_dataset=features[:4001]
	testing_dataset=features[-1003:]
	classfier= OneVsRestClassifier(MultinomialNB())
	classfier.fit(training_dataset,training_labels)
	score = classfier.score(testing_dataset, testing_labels)
	return score
def class_mlp(tweets,labels,words,limit):
	most_frequently_used=words[:limit]
	features=[]
	for tweet in tweets:
		flags=[]
		for word in most_frequently_used:
			if word[0] in tweet:
				flags.append(True)
			else:
				flags.append(False)
		features.append(flags)
	training_labels=labels[:4001]
	testing_labels=labels[-1003:]
	training_dataset=features[:4001]
	testing_dataset=features[-1003:]
	mlp_classfier= OneVsRestClassifier(MLPClassifier())
	mlp_classfier.fit(training_dataset,training_labels)
	score = mlp_classfier.score(testing_dataset, testing_labels)
	return score
def class_svm(tweets,labels,words,limit):
	most_frequently_used=words[:limit]
	features=[]
	for tweet in tweets:
		flags=[]
		for word in most_frequently_used:
			if word[0] in tweet:
				flags.append(True)
			else:
				flags.append(False)
		features.append(flags)
	training_labels=labels[:4001]
	testing_labels=labels[-1003:]
	training_dataset=features[:4001]
	testing_dataset=features[-1003:]
	svc = OneVsRestClassifier(sklearn.svm.SVC())
	svc.fit(training_dataset,training_labels)
	score = svc.score(testing_dataset, testing_labels)
	return score
def old_NB(tweets,labels,limit):
	new_tweets=[]
	i=0
	for tweet in tweets:
		new_tweets.append((tweet,labels[i]))
		i+=1
	training_dataset=new_tweets[:4001]
	testing_dataset=new_tweets[-1003:]
	print(new_tweets[0])
	cl = NaiveBayesClassifier(training_dataset)
	
	score=cl.accuracy(testing_dataset)
	
	return score

def main_method():
	sets=reading_dataset("/Users/ahmed/Desktop/Bachelor/ASTD-master/data/tweets.txt")
	tweets=sets[0]
	labels=sets[1]
	words_without=without_stopwords(tweets)
	words=with_stopwords(tweets)
	scores={}
	scores["score_NB_without_2000"]=old_NB(tweets,labels,2000)
	print(scores)
	#old_NB(tweets,labels,2000)



	



main_method()


