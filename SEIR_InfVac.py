# -*- coding: utf-8 -*-
"""
Date: 16/02/2020

Description: The SEIR epidemic model with pediatric vaccination.
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
q = 0.0

# Time interval limits
t0, tmax = 0, 400
# A grid of time points (in days)
t = np.linspace(t0, tmax, 1000)

# -----------------------------------------------------------------------------
#                       Differential equations 
# -----------------------------------------------------------------------------

def deriv(y, t, A, mu, alpha, beta, gamma, sigma, q):
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
        q (float): vaccination proportion
        
    Returns:
        dSdt (float): derivative of S in time t
        dEdt (float): derivative of E in time t
        dIdt (float): derivative of I in time t
        dRdt (float): derivative of R in time t
    """    
    S, E, I, R = y
    dSdt = (1-q)*A - (beta*I+mu)*S
    dEdt = beta*S*I - (mu+sigma)*E
    dIdt = sigma*E - (mu+gamma+alpha)*I
    dRdt = gamma*I - mu*R + q*A
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
    ret = odeint(deriv, y0, t, args=(A, mu, alpha, beta, gamma, sigma, q))
    S, E, I, R = ret.T
    N = S + E + I + R
    return [S, E, I, R, N]

# Initial conditions
y0 = S0, E0, I0, R0
# Integrate the SEIR equations over the time grid
[S, E, I, R, N] = integrate(y0, t)

# -----------------------------------------------------------------------------
#                         R0 and stability
# -----------------------------------------------------------------------------

R0 = (1-q)*A*beta*sigma / ( mu*(mu+sigma)*(mu+gamma+alpha) )
print("Reprodutive number:", R0)

Sdf = (1-q)*A/mu
Rdf = mu*q*A/(mu*mu)

Send = (1-q)*A/(R0*mu)
Eend = (1-q)*A*(1-(1/R0))/(mu+sigma)
Iend = mu*(R0-1)/beta
Rend = (q*A + gamma*Iend)/mu

if R0 < 1.0:
    print("The disease free equilibrium is stable.")
    print("S0 analytical:", Sdf)
    print("S0 numerical:", S[len(S)-1])
    print("E0 analytical:", 0.0)
    print("E0 numerical:", E[len(S)-1])
    print("I0 analytical:", 0.0)
    print("I0 numerical:", I[len(S)-1])
    print("R0 analytical:", Rdf)
    print("R0 numerical:", R[len(S)-1])
    
else:
    print("The endemic equilibrium is stable.")
    print("S0 analytical:", Send)
    print("S0 numerical:", S[len(S)-1])
    print("E0 analytical:", Eend)
    print("E0 numerical:", E[len(S)-1])
    print("I0 analytical:", Iend)
    print("I0 numerical:", I[len(S)-1])
    print("R0 analytical:", Rend)
    print("R0 numerical:", R[len(S)-1])
    
# -----------------------------------------------------------------------------
#                            Plot the data
# -----------------------------------------------------------------------------

fig = plt.figure(facecolor='w' )
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S, 'tomato', alpha=0.4, lw=1.2, label='Susceptibles')
ax.plot(t, E, 'tomato', alpha=0.7, lw=1.2, label='Expuestos')
ax.plot(t, I, 'r', alpha=0.8, lw=1.2, label='Infecciosos')  
ax.plot(t, R, 'grey', alpha=0.6, lw=1.2, label='Recuperados')   
 
ax.set_xlabel('Tiempo (días)', fontsize=13.5, labelpad=6)
ax.set_ylabel('Proporción de individuos', fontsize=13.5, labelpad=10)

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_xticks([0, 100, 200, 300, 400])
ax.grid(b=True, which='major', c='silver', lw=0.5, ls='-')
legend = ax.legend(loc=1)
legend.get_frame().set_alpha(0.9)

for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.savefig('r0df', dpi=600)
plt.show()