punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
digits = '0123456789'



def reading_stopwords(filename):
	dataset=[]
	file = open(filename, 'r',encoding='utf-8')
	i=0
	for line in file :
		dataset.append(line[:-1])
	return dataset
stopwords=reading_stopwords("/Users/ahmed/Desktop/Bachelor/stopwords.txt")
print(sets)