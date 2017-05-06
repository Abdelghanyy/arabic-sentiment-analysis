# This File contains the classfication of the small dataset using NB and SVM
# And extarcting features as most frequently used with stopwords and without ,2 gram words and 3 gram










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
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier


#reading dataset and dividing it into trainging dataset and testing dataset
def reading_datast():
	tweets=[]
	training_labels=[]
	training_dataset=[]
	testing_labels=[]
	testing_dataset=[]
	training=[]
	testing=[]

	for x in range(1,1001):
		filename="twitter/positive/positive"+str(x)+".txt"
		file = open(filename, 'r')
		lines = file.readlines()
		tweet=str(lines)
		new_tweet=tweet.replace("\\xa0","")
		if(x in range(800)):
			training_dataset.append(new_tweet[2:len(new_tweet)-4])
			training.append((new_tweet[2:len(new_tweet)-4],"pos"))
			training_labels.append("pos")
		else:
			testing_dataset.append(new_tweet[2:len(new_tweet)-4])
			testing.append((new_tweet[2:len(new_tweet)-4],"pos"))
			testing_labels.append("pos")

		
	for j in range(1,1001):
		if(j not in [103,116,176,178,180,184,186,189,191]):
			filename="twitter/negative/negative"+str(j)+".txt"
			file = open(filename, 'r')
			lines = file.readlines()
			tweet=str(lines)
			new_tweet=tweet.replace("\\xa0","")
			
			
		if(j in range(800)):
			training_dataset.append(new_tweet[2:len(new_tweet)-4])
			training.append((new_tweet[2:len(new_tweet)-4],"neg"))
			training_labels.append("neg")
		else:
			testing_dataset.append(new_tweet[2:len(new_tweet)-4])
			testing.append((new_tweet[2:len(new_tweet)-4],"neg"))
			testing_labels.append("neg")
	return tweets,training,testing,training_dataset,training_labels,testing_dataset,testing_labels
	
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

def reading_stopwords(filename):
	dataset=[]
	file = open(filename, 'r',encoding='utf-8')
	i=0
	for line in file :
		dataset.append(line[:-1])
	return dataset
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
def two_gram(tweets):
	words={}
	for tweet in tweets:
		cleaned_tweet=clean_tweet(tweet)
		two_gram_tweet=[]
		for x in range(len(cleaned_tweet)-1):
			two_gram_tweet.append(cleaned_tweet[x]+" "+cleaned_tweet[x+1])
		for word in two_gram_tweet:
			flag=True
			for key in words:
				if word==key:
					words[key]+=1
					flag=False
					break
			if flag:
				words[word]=1
	sorted_words=frequently_used(words)
	return sorted_words
def three_gram(tweets):
	words={}
	for tweet in tweets:
		cleaned_tweet=clean_tweet(tweet)
		three_gram_tweet=[]
		for x in range(len(cleaned_tweet)-1):
			if((x+2) in range (len(cleaned_tweet))):
				three_gram_tweet.append(cleaned_tweet[x]+" "+cleaned_tweet[x+1]+" "+cleaned_tweet[x+2])

		for word in three_gram_tweet:
			flag=True
			for key in words:
				if word==key:
					words[key]+=1
					flag=False
					break
			if flag:
				words[word]=1
	sorted_words=frequently_used(words)
	return sorted_words


def feature_extraction(tweets,words,limit):
	most_frequently_used=words[:limit]
	features=[]
	for tweet in tweets:
		flags=[]
		for word in most_frequently_used:
			if word[0] in tweet:
				flags.append(1)
			else:
				flags.append(0)
		features.append(flags)

	return features


def class_NB(tweets,labels,words,limit):
	features=feature_extraction(tweets,words,limit)
	

	training_labels=labels[:1598]
	testing_labels=labels[-402:]
	training_dataset=features[:1598]
	testing_dataset=features[-402:]
	#print(testing_labels)
	classfier= OneVsRestClassifier(GaussianNB())
	classfier.fit(training_dataset,training_labels)
	score = classfier.score(testing_dataset, testing_labels)
	classfier2= OneVsRestClassifier(BernoulliNB())
	classfier2.fit(training_dataset,training_labels)
	score2 = classfier2.score(testing_dataset, testing_labels)
	classfier3= OneVsRestClassifier(MultinomialNB())
	classfier3.fit(training_dataset,training_labels)
	score3 = classfier3.score(testing_dataset, testing_labels)

	return score,score2,score3

def class_svm(tweets,labels,words,limit):
	most_frequently_used=words[:limit]
	features=[]
	for tweet in tweets:
		flags=[]
		for word in most_frequently_used:
			if word[0] in tweet:
				flags.append(1)
			else:
				flags.append(0)
		features.append(flags)
	training_labels=labels[:1598]
	testing_labels=labels[-402:]
	training_dataset=features[:1598]
	testing_dataset=features[-402:]
	svc = OneVsRestClassifier(sklearn.svm.SVC())
	svc.fit(training_dataset,training_labels)
	score = svc.score(testing_dataset, testing_labels)
	return score


def voting_class (tweets,labels,words,limit):
	features=feature_extraction(tweets,words,limit)
	

	training_labels=labels[:1598]
	testing_labels=labels[-402:]
	training_dataset=features[:1598]
	testing_dataset=features[-402:]

	gauss=OneVsRestClassifier(GaussianNB())
	berr=OneVsRestClassifier(BernoulliNB())
	multi=OneVsRestClassifier(MultinomialNB())
	svm=OneVsRestClassifier(sklearn.svm.SVC())
	random=OneVsRestClassifier(RandomForestClassifier())
	mlp=OneVsRestClassifier(MLPClassifier())

	classifier=VotingClassifier(estimators=[("gauss",gauss),("ber",berr),("multi",multi),("svm",svm),("randomForest",random),("mlp",mlp)],voting='hard')
	classifier.fit(training_dataset,training_labels)
	score=classifier.score(testing_dataset,testing_labels)
	return score


def main_method():
	#data
	result={}
	svm={}
	sets=reading_datast()
	tweets=sets[0]
	training=sets[1]
	testing=sets[2]
	training_dataset=sets[3]
	training_labels=sets[4]
	testing_dataset=sets[5]
	testing_labels=sets[6]
	#print(training)

	new_tweets=training_dataset+testing_dataset
	new_lables=training_labels+testing_labels

	words_without=without_stopwords(new_tweets)
	words=with_stopwords(new_tweets)
	two_gram_words=two_gram(new_tweets)
	three_gram_words=three_gram(new_tweets)

	#scores=class_NB(new_tweets,new_lables,words,len(words)-1)
	#result["gaus"]=scores[0]
	#result["ber"]=scores[1]
	#result["multi"]=scores[2]

	result["kollo"]=voting_class(new_tweets,new_lables,words,len(words)-1)
	print(result)


main_method()