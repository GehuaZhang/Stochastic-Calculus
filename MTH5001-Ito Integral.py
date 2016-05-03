# -*- coding: utf-8 -*

import random
import math
import numpy as np
import pylab as pl
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
 
def StochasticIntegral(function,a,b,dt):
	#f(t,W(t))
	func = compile(function,"",'eval')	
	a=float(a)
	b=float(b)
	n=int((b-a)/dt)
	WP = 0
	t = a
	length = 0
	for i in range(int(a/dt)):
		WP = WP + np.random.normal(0,math.sqrt(dt))
	a1 = eval(func)
	for i in range(n):
		t = dt*i+a
		WP = WP+np.random.normal(0,math.sqrt(dt))
		length = length + eval(func)
		if i == n-1:
			an = eval(func)
	result = (2*length-a1-an)*dt/2
	return result

def ItoDiffIntegral(miuT,sigmaT,z0,time,dt):
	#dZ(t)=miu(t)dt+sigma(t)dW(t)
	#Calculate integral of dZ(t)
	time = float(time)
	n = int(time/dt)
	miuFunc = compile(miuT,"",'eval')
	sigmaFunc = compile(sigmaT,"",'eval')
	miuResult,sigmaResult = 0,0
	for i in range(n):
		t = i*dt
		miu_begin = eval(miuFunc)
		sigmaValue = eval(sigmaFunc)
		t = (i+1)*dt
		miu_end = eval(miuFunc)
		t = (i+0.5)*dt
		miu_middle = eval(miuFunc)
		singleMiuValue = (miu_begin+miu_end+4*miu_middle)/6
		singleSigmaValue = sigmaValue*np.random.normal(0,math.sqrt(dt))
		miuResult = miuResult + singleMiuValue*dt
		sigmaResult = sigmaResult + singleSigmaValue
	print 'dZ = ('+str(miuT)+')*dt + ('+str(sigmaT)+')*dW(t)'
	return z0 + miuResult + sigmaResult

def ItoFormula_1(function,a,b,dt):
	func = compile(function,"",'eval')
	a=float(a)
	b=float(b)
	n=int((b-a)/dt)
	WP = 0
	t = a
	length = 0
	for i in range(int(a/dt)):
		WP = WP + np.random.normal(0,math.sqrt(dt))
	a1 = eval(func)
	for i in range(n):
		t = dt*i+a
		WP = WP+np.random.normal(0,math.sqrt(dt))
		length = length + eval(func)
		if i == n-1:
			an = eval(func)
	result = (2*length-a1-an)*dt/2
	return result
	print 'dX = '+str(miuT)+'*dt + '+str(sigmaT)+'*dWt'

def ItoFormula_2(function,a,b,dt):
	func = compile(function,"",'eval')
	a=float(a)
	b=float(b)
	n=int((b-a)/dt)	
	WP = 0
	t = a
	for i in range(int(a/dt)):
		WP = WP+np.random.normal(0,math.sqrt(dt))
	part_1 = eval(func)
	f = sym.sympify(function)
	WP = sym.Symbol('WP')
	df = str(sym.diff(f,WP))
	ddf = str(sym.diff(df,WP))
	part_2 = 0
	part_3 = 0
	result_2 = 0
	WP = 0
	for i in range(0,n):
		t=dt*((a/dt)+i)
		WP = WP+np.random.normal(0,math.sqrt(dt))
		k2 = eval(df,valueDic)
		k3 = eval(ddf,valueDic)
		part_2 = part_2 + k2*(WT[i+1]-WT[i])
		part_3 = part_3 + k3*dt
	result_2 = part_1 + part_2 + part_3
	print 'ito formula result: '+str(result_2)

def ItoIsoLeftSide(function,a,b,dt,N):
	func = compile(function,"",'eval')	
	n=int(float(b-a)/dt)
	result = 0
	for j in range(N):	
		integralValue = 0
		for i in range(n):
			t=dt*(i+1)+a
			funcValue = eval(func)
			integralValue = integralValue + funcValue*(np.random.normal(0,math.sqrt(dt)))
		result = result + (integralValue)**2
	return result/N

def ItoIsoRightSide(function,a,b,dt):
	func = compile(function,"",'eval')	
	result = 0
	n=int(float(b-a)/dt)
	for i in range(0,n):
		t=dt*i+a
		funcValue = eval(func)
		result = result + dt*(funcValue)**2
	return result

def CallFunction(a):
	if a == '1':
		function1 = 't**3-WP'
		a = 1
		b = 2
		dt = 0.001
		print 'Stochastic Intergal:'
		print 'Function is: '+function1
		print 'Simulation for intergal: '+str(StochasticIntegral(function1,a,b,dt))+'\n'
	if a ==	'2':
		function2 = '2*t**3+100'
		a = 1
		b = 2
		dt = 0.001
		N = 3000
		print 'Ito Isometry:'
		print 'Function is: '+function2
		print 'LeftSide: '+str(ItoIsoLeftSide(function2,a,b,dt,N))
		print 'RightSide: '+str(ItoIsoRightSide(function2,a,b,dt))+'\n'
	if a == '3':
		print ItoDiffIntegral('4*t**3-2*t','5*t+1',0,1,0.001)

print '1: Stochastic Intergal f(t,W(t)) \n2: Ito Isometry \n3: Has Ito Differetials intergal \n'
a = raw_input('Which One?')
CallFunction(a)
