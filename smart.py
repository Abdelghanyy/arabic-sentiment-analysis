def reading_file(filename):
	file=open(filename,"r")
	lines=file.readlines()
	dataset=[]
	for line in lines:
		new_line=line[:len(line)-1]
		collection=new_line.split(",")
		for number in collection:
			dataset.append(float(number))
	return(dataset)
