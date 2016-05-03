import scipy.optimize 
from scipy.stats import norm
import math

def BSMinusPrice(P,T,K,S0,r,sigma):
	P = float(P)
	T = float(T)
	K = float(K)
	S0 = float(S0)
	r = float(r)
	sigma = float(sigma)
	d1 = (math.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
	d2 = d1 - sigma*math.sqrt(T)
	BS = S0*norm.cdf(d1)-math.exp((-r)*T)*K*norm.cdf(d2)
	return BS-P

def Vega(P,T,K,S0,r,sigma):
	P = float(P)
	T = float(T)
	K = float(K)
	S0 = float(S0)
	r = float(r)
	sigma = float(sigma)
	d1 = (math.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
	vega = S0*norm.pdf(d1)*math.sqrt(T)
	return vega

def func(x):
	return BSMinusPrice(7,0.5,98,100,0.1,x)

def dfunc(x):
	return Vega(7,0.5,98,100,0.1,x)

def Secant(f,x0,x1,TOL=0.001,NMAX=100):
	n=1
	while n<=NMAX:
		x2 = x1 - f(x1)*((x1-x0)/(f(x1)-f(x0)))
		if abs(x2-x1) < TOL:
			return x2
		else:
			x0 = x1
			x1 = x2
	return False

def Newton(f,df,x0,TOL=0.001,NMAX=100):
	n=1
	while n<=NMAX:
		x1 = x0 - f(x0)/df(x0)
		if abs(x0-x1) < TOL:
			return x1
		else:
			x0 = x1
	return False


print "Bisection Method: "
print "sigma = "+str(scipy.optimize.bisect(func,0.001,1))
print "Newton Method: "
print "sigma = "+str(Newton(func,dfunc,1))
print "Secant Method: "
print "sigma = "+str(Secant(func,1,1.2))


