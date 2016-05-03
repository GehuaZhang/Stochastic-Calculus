# -*- coding: utf-8 -*
import random
import math
import numpy as np
import pylab as plt


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

def Numerical_SDE(x0,alpha,sigma,dt,n):
	Wt = np.array(WienerProcess(n,dt))
	x_sde = np.zeros(n)
	t = np.arange(n)/float(n)
	x_sde[0] = x0
	for i in range(n-1):
		x_sde[i+1] = x_sde[i] + alpha*dt + sigma*(Wt[i+1]-Wt[i])
	x_ana = x0 + alpha*t + sigma*Wt
	return t,x_sde,x_ana

def LogNormal_SDE(x0,alpha,sigma,dt,n):
	Wt = np.array(WienerProcess(n,dt))
	x_sde = np.zeros(n)
	t = np.arange(n)/float(n)
	x_sde[0] = x0
	for i in range(n-1):
		x_sde[i+1] = x_sde[i] + alpha*x_sde[i]*dt + sigma*x_sde[i]*(Wt[i+1]-Wt[i])
	x_ana = x0*np.exp(sigma*Wt+(alpha-0.5*sigma**2)*t)
	return t,x_sde,x_ana
	
def Linear_SDE(x0,alpha,sigma,dt,n):
	Wt = np.array(WienerProcess(n,dt))
	x_sde = np.zeros(n)
	x_ana = np.zeros(n)
	t = np.arange(n)/float(n)
	x_sde[0] = x0
	for i in range(n-1):
		x_sde[i+1] = x_sde[i] + alpha*x_sde[i]*dt + sigma*(Wt[i+1]-Wt[i])
		summation = 0
		for j in range(i+1):
			summation = np.exp(alpha*(t[i]-t[j]))*(Wt[j+1]-Wt[j]) + summation
		x_ana[i] = np.exp(alpha*t[i])*x0 + sigma*summation
	return t,x_sde,x_ana

def CEV_SDE(x0,sigma,beta,dt,n):
	x_sde = np.zeros(n)
	x_ana = np.zeros(n)
	t = np.arange(n)/float(n)
	x_sde[0] = x0
	x_ana[0] = x0
	for i in range(n-1):
		dWt = np.random.normal(0,math.sqrt(dt))
		x_sde[i+1] = abs(x_sde[i] + (sigma*(x_sde[i]**beta))*dWt + 0.5*sigma*((x_sde[i])**beta)*beta*sigma*((x_sde[i])**(beta-1))*(dWt**2-dt))
		x_ana[i+1] = abs(x_ana[i] + (sigma*(x_ana[i]**beta))*dWt)
	return t,x_sde,x_ana

#t,x_sde,x_ana = Numerical_SDE(0,2,0.4,0.001,1000)
#t,x_sde,x_ana = LogNormal_SDE(1,0,1.4,0.001,1000)
#t,x_sde,x_ana = Linear_SDE(1,2,0.4,0.001,1000)
t,x_sde,x_ana = CEV_SDE(0.1,1,0.5,0.001,1000)


plt.subplot(2,1,1)
plt.plot(t,x_sde,color='blue',label = 'Milstein Scheme')
plt.legend(loc='upper left')
plt.subplot(2,1,2)
plt.plot(t,x_ana,color='red',label = 'Euler Solution')
plt.legend(loc='upper left')
plt.show()
