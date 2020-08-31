# -*- coding: utf-8 -*-
"""
Date: 18/02/2020

Description: The SEIR epidemic model with pulse vaccination.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
import sys

# -----------------------------------------------------------------------------
#                                LaTex
# -----------------------------------------------------------------------------

matplotlib.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font = {'weight': 'normal', 'size': 12}  # Graph number's fontsize
plt.rc('font', **font)
plt.rc('legend', fontsize=11)  # Legend's fontsize

# -----------------------------------------------------------------------------
#                       Parameter Declaration 
# -----------------------------------------------------------------------------

# Initial number of population
N0 = 4e7

# Initial proportion of susceptible, exposed, infectious and recovered ind.
S0, E0, I0, R0 = 1.0 - (1.0/N0), 0, 1.0/N0, 0

# Demographic parameters 
A, mu = 0.005, 0.005

# Epidemiologic parameters 
alpha, beta, gamma, sigma = 0.01, 0.57, 0.067, 0.13

# Vaccination proportion
p = 0.60

# Pulse vaccination period 
T = 30

# Time interval limits
t0, tmax = 0, 1000
# Time subintervals' step number
ss = 350

# -----------------------------------------------------------------------------
#                       Differential equations 
# -----------------------------------------------------------------------------
def deriv(y, t, A, alpha, beta, gamma, mu, sigma):
    """
    Implements the differential equations in which the SEIR model is based.
    
    Args: 
        y (tuple): tuple containing S, E, I and R variables
        t (Numpy.ndarray): grid of time points in [t0, tmax] interval
        A (float): proportion of new individuals per unit time
        mu (float): natural death rate
        alpha (float): disease-related death rate
        beta (float): contact rate
        gamma (float): recovery rate
        sigma (float): inverse latent period
        
    Returns:
        dSdt (float): derivative of S in time t
        dEdt (float): derivative of E in time t
        dIdt (float): derivative of I in time t
        dRdt (float): derivative of R in time t
    """   
    S, E, I, R = y
    dSdt = A - (beta*I+mu)*S 
    dEdt = beta*S*I - (mu+sigma)*E
    dIdt = sigma*E - (mu+gamma+alpha)*I
    dRdt = gamma*I - mu*R   
    return dSdt, dEdt, dIdt, dRdt

def integrate(y0, t): 
    """
    Function that integrates the SEIR equations over the given time interval.
    
    Args: 
        y0 (tuple): tuple containing S, E, I and R variablesi initial values
        t (Numpy.ndarray): grid of time points in [t0, tmax] interval

    Returns:
        S (Numpy.ndarray): solution of S in [t0, tmax] interval
        E (Numpy.ndarray): solution of E in [t0, tmax] interval
        I (Numpy.ndarray): solution of I in [t0, tmax] interval
        R (Numpy.ndarray): solution of R in [t0, tmax] interval
        N (Numpy.ndarray): solution of N in [t0, tmax] interval
    """ 
    ret = odeint(deriv, y0, t, args=(A, alpha, beta, gamma, mu, sigma))
    S, E, I, R = ret.T
    N = S + E + I + R
    return [S, E, I, R, N]

# -----------------------------------------------------------------------------
#              Integrate over the different time subintervals
# -----------------------------------------------------------------------------

# First time interval 
y0 = S0, E0, I0, R0
t = np.linspace(t0, T, ss)
[S, E, I, R, N] = integrate(y0, t)

# Time interval number: IT MUST BE BIGGER THAN OR EQUAL TO 1
nf  = (tmax-t0)/T 
n = int(np.floor(nf))

# Middle time intervals
for i in range(1,n):
    
    length = len(N)
    S0 = (1-p)*S[length-1]
    R0 = R[length-1] + p*S[length-1]    
    y0 = S0, E[length-1], I[length-1], R0
    t = np.linspace(i*T, (i+1)*T, ss)
    
    [subS, subE, subI, subR, subN] = integrate(y0, t)
    
    S = np.append(S, subS)
    E = np.append(E, subE)
    I = np.append(I, subI)
    R = np.append(R, subR)
    N = np.append(N, subN)

# Last time interval
if nf!=n:
    
    S0 = (1-p)*S[length+ss-1]
    R0 = R[length+ss-1] + p*S[length+ss-1]
    y0 = S0, E[length+ss-1], I[length+ss-1], R0
    t = np.linspace(n*T, tmax, ss)
    
    [subS, subE, subI, subR, subN] = integrate(y0, t)

    S = np.append(S, subS)
    E = np.append(E, subE)
    I = np.append(I, subI)
    R = np.append(R, subR)
    N = np.append(N, subN)   
    
    
# Whole time interval (for future plots)
length = len(N)
t = np.linspace(t0, tmax, length)
print("Infectious prop:" , I[len(I)-1])

# Critical susceptible proportion
Rep = A*beta*sigma / ( (mu)*(mu+sigma)*(mu+gamma+alpha) )
print("Reprodutive number:", Rep)
print("Inverse reproductive numer:", 1.0/Rep)
line=[]
for inst in t:
    line.append(1.0/Rep)

# -----------------------------------------------------------------------------
#                            Plot the data
# -----------------------------------------------------------------------------

fig = plt.figure(facecolor='w',figsize=(7,4))
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S, 'tomato', alpha=0.6, lw=1.2, label='Susceptibles')
#ax.plot(t, E, 'tomato', alpha=0.7, lw=1.2, label='Expuestos')
ax.plot(t, I, 'r', alpha=0.8, lw=1.2, label='Infecciosos')  
#ax.plot(t, R, 'grey', alpha=0.6, lw=1.2, label='Recuperados')  
ax.plot(t, line, "black", alpha=0.8, linestyle="dashdot", lw=1.0, label='$S_c$')

ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))     
ax.set_xlabel('Tiempo (días)', fontsize=13.5, labelpad=6)
ax.set_ylabel('Proporción de individuos', fontsize=13.5, labelpad=10)

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='silver', lw=0.5, ls='-')
legend = ax.legend(loc=1)
legend.get_frame().set_alpha(0.9)

for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.savefig('422.png', dpi=600)
plt.show()