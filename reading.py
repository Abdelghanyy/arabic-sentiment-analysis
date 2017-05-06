from textblob.classifiers import NaiveBayesClassifier
training=[]
testing=[]
for x in range(1,500):
	filename="twitter/positive/positive"+str(x)+".txt"
	f = open(filename, 'r')
	n = f.readlines()
	y=str(n)

	training.append((y,"pos"))
for x in range(500,1001):
	filename="twitter/positive/positive"+str(x)+".txt"
	f = open(filename, 'r')
	n = f.readlines()
	y=str(n)

	testing.append((y,"pos"))

for i in range(1,500):
	if(i not in [103,116,176,178,180,184,186,189,191]):
		filename="twitter/negative/negative"+str(x)+".txt"
		f = open(filename, 'r')
		n = f.readlines()
		y=str(n)
		training.append((n,"neg"))
for i in range(500,1001):
	if(i not in [103,116,176,178,180,184,186,189,191]):
		filename="twitter/negative/negative"+str(i)+".txt"
		f = open(filename, 'r')
		n = f.readlines()
		y=str(n)
		testing.append((y,"neg"))
print(training[0])
cl = NaiveBayesClassifier(training)
print(cl.accuracy(testing))
cl.show_informative_features(5)
