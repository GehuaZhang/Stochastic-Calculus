import numpy as np
import math
import pylab as plt
import scipy.optimize 
from scipy.stats import norm

dt = 10**(-2)

T = [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50]
F0 = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
K = [0.04,0.0425,0.045,0.0475,0.05,0.05,0.0525,0.055,0.0575,0.06,0.04,0.0425,0.045,0.0475,0.05,0.05,0.0525,0.055,0.0575,0.06]
sigma0 = [0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
alpha = [2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,1.50,1.50,1.50,1.50,1.50,1.50,1.50,1.50,1.50,1.50]
beta = [0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30]
rho = [0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10]
Option = ['call','call','call','call','call','put','put','put','put','put','call','call','call','call','call','put','put','put','put','put']
dW_list_1,dZ_list_1,dW_list_2,dZ_list_2 = [],[],[],[]
ImpliedVol_list_1,strike_list_1,ImpliedVol_list_2,strike_list_2 = [],[],[],[]

def Generate_dW_dZ(rho,T,dt):
	n = int(T/dt)
	x1=[]
	x2=[]
	for i in range(n):
		dx1 = np.random.normal(0,math.sqrt(dt))
		x1.append(dx1)
		x2.append(rho*dx1+math.sqrt(1-rho**2)*np.random.normal(0,math.sqrt(dt)))
	return [x1,x2]

def SABR_Price(T,F0,K,sigma0,alpha,beta,rho,dt,Option,dW,dZ):
	n = int(T/dt)
	#[dW,dZ] = Generate_dW_dZ(rho,T,dt)
	sigma = sigma0
	F = F0
	for i in range(n):
		F = max(F+sigma*F**beta*dW[i],0.0)
		sigma = sigma*math.exp(alpha*dZ[i]-alpha**2*dt/2)
	if Option == 'call':
		return max(F-K,0.0)
	elif Option == 'put':
		return max(K-F,0.0)

def SABR_MC_Price(T,F0,K,sigma0,alpha,beta,rho,dt,Option,dW,dZ):
	MC_sum=0
	for i in range(40000):
		dw = dW[i]
		dz = dZ[i]
		MC_sum = MC_sum + SABR_Price(T,F0,K,sigma0,alpha,beta,rho,dt,Option,dw,dz)
	MC_price = MC_sum/40000
	return MC_price

def BS_Price(T,F0,K,Option,sigma):
	d1 = (math.log(F0/K)+(0.5*sigma**2)*T)/(sigma*math.sqrt(T))
	d2 = d1 - sigma*math.sqrt(T)
	if Option == 'call':
		BS = F0*norm.cdf(d1)-K*norm.cdf(d2)
	elif Option == 'put':
		BS = K*norm.cdf(-d2)-F0*norm.cdf(-d1)
	return BS

def func(x,T,F0,K,Option,SABR_price_instance):
	return BS_Price(T,F0,K,Option,x)-SABR_price_instance

def ImpliedVol(T,F0,K,sigma0,alpha,beta,rho,dt,Option,dW,dZ):
	SABR_price_instance = SABR_MC_Price(T,F0,K,sigma0,alpha,beta,rho,dt,Option,dW,dZ)
	if Option == 'call':
		if SABR_price_instance > F0-K:
			return scipy.optimize.newton(func,0.1,args=(T,F0,K,Option,SABR_price_instance,))
		else:
			return 0.0
	elif Option == 'put':
		if SABR_price_instance > K-F0:
			return scipy.optimize.newton(func,0.1,args=(T,F0,K,Option,SABR_price_instance,))
		else:
			return 0.0

for i in range(40000):
	[dW_1,dZ_1] = Generate_dW_dZ(0.30,0.25,dt)
	[dW_2,dZ_2] = Generate_dW_dZ(0.10,0.50,dt)
	dW_list_1.append(dW_1)
	dZ_list_1.append(dZ_1)
	dW_list_2.append(dW_2)
	dZ_list_2.append(dZ_2)

for i in range(10):
	ImpliedVol_list_1.append(ImpliedVol(T[i],F0[i],K[i],sigma0[i],alpha[i],beta[i],rho[i],dt,Option[i],dW_list_1,dZ_list_1))
	strike_list_1.append(K[i])
for i in range(10,20):
	ImpliedVol_list_2.append(ImpliedVol(T[i],F0[i],K[i],sigma0[i],alpha[i],beta[i],rho[i],dt,Option[i],dW_list_2,dZ_list_2))
	strike_list_2.append(K[i])

plt.plot(strike_list_1,ImpliedVol_list_1)
plt.plot(strike_list_2,ImpliedVol_list_2)
plt.show()