from nltk.classify import PositiveNaiveBayesClassifier
import re

chinese_names = ['gao', 'chan', 'chen', 'Tsai', 'liu', 'Lee']

nonchinese_names = ['silva', 'anderson', 'kidd', 'bryant', 'Jones', 'harris', 'davis']

def three_split(word):
    word = word.lower()
    word = word.replace(" ", "_")
    split = 3
    return dict(("contains(%s)" % word[start:start+split], True) 
        for start in range(0, len(word)-2))

positive_featuresets = list(map(three_split, chinese_names))
unlabeled_featuresets = list(map(three_split, nonchinese_names))
classifier = PositiveNaiveBayesClassifier.train(positive_featuresets, 
    unlabeled_featuresets)

name = "tsai"
print (three_split(name))
print (classifier.classify(three_split(name)))