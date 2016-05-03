# -*- coding: utf-8 -*


import random
import math
import numpy as np
#import pylab as plt


def WienerProcess(n,dt):
	result = []
	t = []
	X = 0
	n = int(n)
	for i in range(n):
		X = np.random.normal(X,math.sqrt(dt))
		result.append(X)
		t.append(i)
	return result

def generateXT(t,xt,n,dt,sigma):
	timeLeft = n-t
	for i in range(timeLeft):
		xt = xt+sigma*math.sqrt(xt)*(np.random.normal(0,math.sqrt(dt)))
	return xt

print generateXT(6,120,1000,0.001,0.1)
