"""
Numerical Software Lab - Project 4

@author: Getuar Rexhepi

email: grexhepi@constructor.university

17 April, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.integrate


#a)
#Function that solves the second order differential equation
#arguments a,b,c,omega, initial values with default values
#a vector of time t, and default values for y axis

def ODE2(a,b,c,omega,t, x0=1, v0=0, startXV=-10, endXV =10):
    def f(y, t, params):
        #The second order ODE was split in two First order ODE's
        x,v = y 
        a, b, c, omega = params 
        derivs = [v, c*np.cos(omega*t) + a*(1-x)*v - b*x**2]
        return derivs
    #initial condition vector
    y0 = [x0, v0]
    #parameters vector
    params = [a, b, c, omega]
    #unpacking the values for the starting 
    #and ending time
    tStart = t[0]
    tStop = t[-1]    
    #using odeint to solve the differential equation
    psoln = odeint(f, y0, t, args=(params,))
    #plotting both solutions on the same graph
    plt.plot(t, psoln[:,0], label='x(t)')
    plt.plot(t, psoln[:,1], label='v(t)')
    #labeling the axis
    plt.xlabel('Time (s)')
    plt.ylabel('X / V')
    #limiting the y axis so the solutions can be better seen
    plt.ylim(startXV, endXV)
    #showing the legen of the graph
    plt.legend()
    #putting a title for the graph
    plt.title('Solutions of Second Order Differential Equation')       
    
    #saving the time vector and the x values
    #in a csv file as two rows
    np.savetxt('OutputData.csv', (t,psoln[:,0]), delimiter=',')
    #plot is exported as a pdf figure and not shown here
    plt.savefig('SolutionsofODE.pdf')
    plt.close()
    
#testing the function with some random parameters    
a=1
b=2
c=3
omega=1
#time was choosen like this
#because of the function odeint,
#if more values then excesive work warning
time = np.arange (0.,3.05,0.05)
#the function is called here with all parameters
ODE2(a,b,c,omega,t=time, x0=1, v0=0, startXV=-20,endXV =5)
#nice pdf graph 

#b)
#defining three functions to calculate the volume and moments of
#inertia of a torus of uniform density, unit mass, 
#average radius R and cross-sectional radius r

#calculating volume with numerical integration
def V(r,R):
    #function that will be integrated
    #2 was put inside of the integral for convenience
    f = lambda z, rho, theta: 2*rho
    #g,h,j,k are limits of integration
    g = lambda rho: R - r
    h = lambda rho: R + r
    j = lambda theta, rho: 0
    k = lambda theta, rho: np.sqrt(r**2 - (rho - R)**2)
    #triple integral is calculated and returned
    #the first integral limits from 0 to 2*pi
    return scipy.integrate.tplquad(f, 0, 2 * np.pi, g, h, j, k)

#calculating inerita with numerical integration
def Iz(r,R):
    #function that will be integrated
    #the outside factor was put inside of the integral 
    #for convenience, together with rho**3
    f = lambda z, rho, theta: (2/Volume[0])*rho**3
    #g,h,j,k are limits of integration,
    #using lambda notation
    g = lambda rho: R - r
    h = lambda rho: R + r
    j = lambda theta, rho: 0
    k = lambda theta, rho: np.sqrt(r**2 - (rho - R)**2)
    #triple integral is calculated and returned
    #the first integral limits from 0 to 2*pi
    return scipy.integrate.tplquad(f, 0, 2 * np.pi, g, h, j, k) 

#calculating others inerita with numerical integration
def Ix(r,R):
    #the outside factor was put inside of the integral 
    #for convenience
    #f is the function that will be integrated
    f = lambda z, rho, theta: (2/Volume[0])*((rho**2)*np.sin(theta)**2+z**2)*rho
    #limits of integration using lambda notation
    g = lambda rho: R - r
    h = lambda rho: R + r
    j = lambda theta, rho: 0
    k = lambda theta, rho: np.sqrt(r**2 - (rho - R)**2)
    #triple integral is calculated and returned
    #the first integral limits from 0 to 2*pi
    return scipy.integrate.tplquad(f, 0, 2 * np.pi, g, h, j, k)
    
#numerically determining each value
R = 2
r = 1
Volume = V(r,R)

Inertia1 = Iz(r,R)

Inertia2 = Ix(r,R)

#determing exact values
#using the formulas provided
def Vex (r,R):
    return 2*np.pi**2*2*1**2
def Izex (r,R):
    return R**2+3*r**2/4
def Ixyex (r,R):
    return R**2/2 + 5*r**2/8

#printing the input as well as output

print ('For radius R=',R,'and cross-sectional radius r=',r,'the calculated values are: ')

print ('V =', Volume[0])
print ('Iz =', Inertia1[0])
print ('Ix = Iy =', Inertia2[0])

print ('For radius R=',R,'and cross-sectional radius r=',r,'the exact values are: ')

print ('V =', Vex(1,2))
print ('Iz =', Izex(1,2))
print ('Ix = Iy =', Ixyex(1,2))

#comparing the values
print ('Difference for Volume V:',Volume[0] - Vex(1,2))
print ('Difference for Inertia z:',Inertia1[0] - Izex(1,2))
print ('Difference for Inertia x,y:',Inertia2[0] - Ixyex(1,2))

#very small difference




