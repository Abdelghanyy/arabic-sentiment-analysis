def reading_new_dataset(filename):
    labels = []
    dataset=[]
    file = open(filename, 'r')
    i=0
    for line in file :
        labels.append(line[-4:])
        labels[i]=labels[i][:3]
        if (labels[i]=="RAL"):
            dataset.append(line[:len(line)-8])
        else:
            dataset.append(line[:len(line)-5])
            if (labels[i]=="OBJ"):
                labels[i]="RAL"
        i=i+1

    return dataset,labels


def setup():
    new_dataset=reading_new_dataset("/Users/Ali/Development/Python/arabic-sentiment-analysis/ASTD-master/data/tweets.txt")
    new_tweets=new_dataset[0]
    new_lables=new_dataset[1]
    # words_without=without_stopwords(new_tweets)
    # print("words_without")
    # words=with_stopwords(new_tweets)
    print("words")


setup()