test=[[True,False,True],[True,False,True]]
file=open("kolloyawaleed.txt","w")
for item in test:
	file.write("%s\n" % item)

newtest=[]
file=open("kolloyawaleed.txt","r")
for line in file:
	newtest.append(line)

print(newtest)