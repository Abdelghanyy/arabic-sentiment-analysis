def reading_new_dataset(filename):
	labels = []
	dataset=[]
	file = open(filename, 'r')
	i=0
	for line in file:
		line=file.readline()
		dataset.append(line)
		x=file.readline()
		dataset.append(x)

	return dataset,labels,i



data=reading_new_dataset("/Users/ahmed/Desktop/Bachelor/newdataset.txt")
print(len(data[0]))