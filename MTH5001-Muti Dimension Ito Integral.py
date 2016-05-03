# -*- coding: utf-8 -*

import random
import math
import numpy as np
import pylab as plt
import sympy as sym


def WienerProcess(n,dt):
	result = []
	t = []
	X = 0
	n = int(n)
	for i in range(n):
		X = np.random.normal(X,math.sqrt(dt))
		result.append(X)
		t.append(i)
	return [t,result]

def GenerateY(n,dt,s):
	[t,x1] = WienerProcess(n,dt)
	[t,x2] = WienerProcess(n,dt)
	y1 = x1
	y2 = []
	for i in range(len(x1)):
		y2.append(s*x1[i]+math.sqrt(1-s**2)*x2[i])
	
	return [t,x1,x2,y2]
	
def Calculus_initial(function):
	t,w1t,w2t = sym.symbols('t w1t w2t')
	f = sym.sympify(function)
	dfdt = str(sym.diff(f,t))
	dfdw1t = str(sym.diff(f,w1t))
	dfdw2t = str(sym.diff(f,w2t))
	ddfddw1t = str(sym.diff(f,w1t,2))
	ddfddw2t = str(sym.diff(f,w2t,2))
	ddfdw1tdw2t = str(sym.diff(f,w1t,w2t))
	return [dfdt,dfdw1t,dfdw2t,ddfddw1t,ddfddw2t,ddfdw1tdw2t]

def MultiDimenItoIntegral(dfdt,dfdw1t,dfdw2t,ddfddw1t,ddfddw2t,ddfdw1tdw2t,n,dt,s):

	[t_list,w1t_list,w2t_list] = GenerateY(n,dt,s)
	func_dfdt = compile(dfdt,"",'eval')
	func_dfdw1t = compile(dfdw1t,"",'eval')
	func_dfdw2t = compile(dfdw2t,"",'eval')
	func_ddfddw1t = compile(ddfddw1t,"",'eval')
	func_ddfddw2t = compile(ddfddw2t,"",'eval')
	func_ddfdw1tdw2t = compile(ddfdw1tdw2t,"",'eval')

	value_dfdt,value_ddfddwt,value_dfdwt = 0,0,0

	for i in range(len(t_list)):
		t = t_list[i]
		w1t = w1t_list[i]
		w2t = w2t_list[i]

		value_dfdt = eval(func_dfdt)*dt + value_dfdt
		value_ddfddwt = 0.5*(eval(func_ddfddw1t)+2*s*eval(func_ddfdw1tdw2t)+eval(func_ddfddw2t))*dt + value_ddfddwt
		value_dfdwt = (eval(func_dfdw1t)+eval(func_dfdw2t))*np.random.normal(0,math.sqrt(dt)) + value_dfdwt
	
	return value_dfdt+value_ddfddwt+value_dfdwt

x,y1,y2,y3 = [],[],[],[]
summation = 0
[x,y1,y2,y3] = GenerateY(1000,0.001,0.5)
a = np.corrcoef(y1,y3)
print a[1][0]

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.show()
