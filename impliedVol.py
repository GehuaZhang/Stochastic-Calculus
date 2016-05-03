import scipy.optimize 
from scipy.stats import norm
import math

def PriceMinusBS(P,T,K,S0,r,sigma):
	P = float(P)
	T = float(T)
	K = float(K)
	S0 = float(S0)
	r = float(r)
	sigma = float(sigma)
	d1 = (math.log(S0/K)+r+0.5*sigma**2)/sigma
	d2 = d1 - sigma
	BS = S0*norm.cdf(d1)-math.exp(-r)*K*norm.cdf(d2)
	return P-BS

def func(x):
	return PriceMinusBS(60,1,100,100,0.01,x)

def secant(f,x0,x1, TOL=0.001, NMAX=100):
	n=1
	while n<=NMAX:
		x2 = x1 - f(x1)*((x1-x0)/(f(x1)-f(x0)))
		if x2-x1 < TOL:
			return x2
		else:
			x0 = x1
			x1 = x2
	return False

print "Bisection Method: "
print "sigma = "+str(scipy.optimize.bisect(func,1,3))
print "Secant Method: "
print "sigma = "+str(secant(func,1,1.2))

