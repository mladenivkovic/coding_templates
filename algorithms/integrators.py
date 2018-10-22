#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(321)
ax1.set_title(r'Integrating $\frac{dy}{dx} = \cos(x)y$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2 = fig.add_subplot(322)
ax2.set_title(r'Relative difference $\frac{y-y_{analytical}}{y_{analytical}}$')
ax3 = fig.add_subplot(323)
ax3.set_title(r'Integrating $\frac{d^2x}{dt^2} = w^2x$')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax4 = fig.add_subplot(324)
ax4.set_title(r'Relative difference $\frac{y-y_{analytical}}{y_{analytical}}$')
ax5 = fig.add_subplot(325)
ax5.set_xlabel('t')
ax5.set_ylabel('v')
ax6 = fig.add_subplot(326)
ax6.set_title(r'Relative difference $\frac{y-y_{analytical}}{y_{analytical}}$')


#====================================================================
# Integrate dy/dx = cos(x)*y numerically using different methods
#====================================================================


xmin = 0
xmax = 2*np.pi
nx = 50
x = np.linspace(xmin, xmax, nx)
dx = x[1] - x[0]

def dydx(y,x):
    return np.cos(x)*y

def analytical(x):
    return np.exp(np.sin(x))

analy = analytical(x)







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
ax1.set_xlim(xmin, xmax)







#======================================
# Integrate d^2 x/dt^2 = w^2*x
#======================================

w = 1
v0 = w
tmin = 0
tmax = 50
nt = 50
t = np.linspace(tmin, tmax, nt)
dt = t[1]-t[0]

def analytic2_x(t):
    return np.exp(w*t)

def analytic2_v(t):
    return w*np.exp(w*t)

def a(x):
    return w**2*x

analy2_x = analytic2_x(t)
analy2_v = analytic2_v(t)



#==============================
# Strömer-Verlet
#==============================

x = np.zeros(nt)
v = np.zeros(nt)
x[0] = 1
x[1] = x[0] + v0*dt + 0.5*a(x[0])*dt**2
v[0] = v0
for n in range(1, nt-1):
    x[n+1] = 2*x[n]-x[n-1] + a(x[n])*dt**2
    v[n] = (x[n+1]-x[n-1])/(2*dt)
v[-1] = (x[-1]-x[-3])/(2*dt)

ax3.plot(t, x, label='Strömer-Verlet')
ax4.plot(t, np.abs(x/analy2_x-1), label='Strömer-Verlet')
ax5.plot(t, v, label='Strömer-Verlet')
ax6.plot(t, np.abs(v/analy2_v-1), label='Strömer-Verlet')



#==============================
# Velocity-Verlet
#==============================

x = np.zeros(nt)
v = np.zeros(nt)
x[0] = 1
v[0] = v0
for n in range(nt-1):
    x[n+1] = x[n] + v[n]*dt + 0.5*a(x[n])*dt**2
    v[n+1] = v[n] + 0.5*(a(x[n])+a(x[n+1]))*dt


ax3.plot(t, x, label='Velocity-Verlet')
ax4.plot(t, np.abs(x/analy2_x-1), label='Velocity-Verlet')
ax5.plot(t, v, label='Velocity-Verlet')
ax6.plot(t, np.abs(v/analy2_v-1), label='Velocity-Verlet')




#==============================
# Leapfrog
#==============================

x = np.zeros(nt)
v = np.zeros(nt)
x[0] = 1
v[0] = v0
for n in range(nt-1):
    vhalf = v[n] + a(x[n]) * dt/2
    x[n+1] = x[n] + vhalf*dt
    v[n+1] = vhalf + a(x[n+1])*dt/2


ax3.semilogy(t, x, label='Leapfrog')
ax4.semilogy(t, np.abs(x/analy2_x-1), label='Leapfrog')
ax5.semilogy(t, v, label='Leapfrog')
ax6.semilogy(t, np.abs(v/analy2_v-1), label='Leapfrog')




t = np.linspace(tmin, tmax, 1000)
ax3.semilogy(t, analytic2_x(t), label='analytical solution')
ax5.semilogy(t, analytic2_v(t), label='analytical solution')
ax3.legend()
ax4.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()

plt.show()
