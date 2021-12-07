from random import random
import matplotlib.pyplot as plt
import numpy

sample= 100000
perday = a = numpy.zeros(shape=(400,sample))
duration=[]
p=0.02
for x in range(sample):
	lok = [0]*21
	lok[0]=3
	count=1
	day=0
	while(count >0):
		sick=[]
		for y in range(0,len(lok)):
			if lok[y]>0:
				sick.append(y)
		for num in range(len(sick)):
			for y in range(0,len(lok)):
				if (lok[y]==0 and random()<p):
					lok[y]=3
		for s in sick:
			lok[s]-=1
		count=0
		for y in range(0,len(lok)):
			if lok[y]>0:
				count+=1
		perday[day][x]+=count
		day+=1
	duration.append(day)
