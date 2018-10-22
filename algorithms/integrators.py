#!/usr/bin/python3

#====================================================================
# Integrate function x*sin(x) numerically using different methods
#====================================================================

from matplotlib import pyplot as plt
import numpy as np

xmin = 0
xmax = 2*np.pi
nx = 50
x = np.linspace(xmin, xmax, nx)
dx = x[1] - x[0]

def dydx(y,x):
    #  return 2*y*x
    return np.cos(x)*y
def analytical(x):
    return np.exp(np.sin(x))
analy = analytical(x)


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title(r'Integrating $\frac{dy}{dx} = \cos(x)y$')
ax2.set_title(r'Relative difference $\frac{y-y_{analytical}}{y_{analytical}}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim(xmin, xmax)




#============================
# EXPLICIT EULER
#============================
y = np.zeros(nx)
y[0] = 1
for n in range(nx-1):
    y[n+1] = y[n] + dydx(y[n],x[n])*dx
ax1.plot(x, y, label='explicit euler')
ax2.plot(x, y/analy-1, label='explicit euler')



#============================
# IMPLICIT EULER
#============================
y = np.zeros(nx)
y[0] = 1
for n in range(nx-1):
    y[n+1] = y[n] + dydx(analytical(x[n+1]), x[n+1])*dx
ax1.plot(x, y, label='implicit euler')
ax2.plot(x, y/analy-1, label='implicit euler')
xmax = 2



#============================
# IMPLICIT MIDPOINT EULER
#============================
y = np.zeros(nx)
y[0] = 1
for n in range(nx-1):
    y[n+1] = y[n] + dydx(0.5*(analytical(x[n+1])+y[n]), 0.5*(x[n]+x[n+1]))*dx
ax1.plot(x, y, label='implicit midpoint')
ax2.plot(x, y/analy-1, label='implicit midpoint')




#============================
# RUNGE KUTTA 2
#============================
y = np.zeros(nx)
y[0] = 1
for n in range(nx-1):
    k1 = dydx(y[n],x[n])*dx
    k2 = dydx(y[n]+k1, x[n+1])*dx
    y[n+1] = y[n] +0.5*(k1+k2)
ax1.plot(x, y, label='Runge Kutta 2')
ax2.plot(x, y/analy-1, label='Runge Kutta 2')




#============================
# RUNGE KUTTA 4
#============================
y = np.zeros(nx)
y[0] = 1
for n in range(nx-1):
    k1 = dydx(y[n],x[n])*dx
    k2 = dydx(y[n]+k1/2, x[n]+dx/2)*dx
    k3 = dydx(y[n]+k2/2, x[n]+dx/2)*dx
    k4 = dydx(y[n]+k3, x[n]+dx)*dx
    y[n+1] = y[n]+(k1/6 + k2/3 + k3/3 + k4/6)
ax1.plot(x, y, label='Runge Kutta 4')
ax2.plot(x, y/analy-1, label='Runge Kutta 4')






x = np.linspace(xmin, xmax, 1000)
ax1.plot(x, analytical(x), label='analytical solution')

ax1.legend()
ax2.legend()

plt.show()
