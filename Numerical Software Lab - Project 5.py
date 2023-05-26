"""
Numerical Software Lab - Project 5

@author: Getuar Rexhepi

email: grexhepi@constructor.university

10 May, 2023
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.linalg
from scipy.optimize import newton, brentq
from scipy import fftpack

#a)
#Generate 1000 random integers between 1 and 10 inclusively
random_numbers = np.random.randint(1, 11, 1000)

#Count the number of occurrences of each integer using np.bincount
counts = np.bincount(random_numbers)

#Plotting a bar chart using the counts of each integer
plt.bar(range(1, 11), counts[1:])

#Axes labels
plt.xlabel('Random Number')
plt.ylabel('Count')

#x-axis tick marks to be the integers from 1 to 10
plt.xticks(range(1, 11))

#y-axis tick marks to be multiples of 10 from 0 to 130
plt.yticks(range(0, 140, 10))

#Setting the title of the plot
plt.title('Occurrence of Random Numbers between 1-10')

#Display the plot
plt.show()

#b)
#Create a 10x10 matrix of random integers between -10 and 10
Matrix = np.random.randint(-10, 10, size=(10, 10))

#Compute the eigenvalues and eigenvectors of the matrix
lam, evec = scipy.linalg.eig(Matrix)

#Convert the eigenvalues and eigenvectors to numpy arrays
lam1 = np.array(lam)
evec1 = np.array(evec)

#Reshape the eigenvalues and eigenvectors into column vectors
#so they are better organized
#this was not required it is just my preference
col_lam = np.reshape(lam, (-1,1))
col_evec = np.reshape(evec, (-1, 1))

#Save vectors of eigenvalues and eigenvectors to a text file
np.savetxt('Eigen.txt', (col_lam, col_evec), fmt='%s')

#c)
#Defining the function
#N is the size of data points
def noisy_func(N, ampltd, freq, G_width, G_ampltd):
    
    # Generating time values
    time = np.linspace(0,4*np.pi, N)
    
    # Generating cosine wave
    cos = ampltd*np.cos(freq*time)
    
    # Adding Gaussian noise to the cosine wave
    gaussian = np.random.normal(loc=0, scale=G_width, size=N)
    noisy_data = cos + G_ampltd*gaussian
    
    # Defining a function to fit the noisy data
    def func(t, A, f):
        return A*np.cos(f*t)

    # Fitting the function to the noisy data
    params, params_covariance = optimize.curve_fit(func, time, noisy_data, p0=[ampltd, freq])
    
    # Extracting the amplitude and frequency from the fit
    Ampltd_fit, freq_fit = params
    
    # Plotting the noisy data and the fit
    plt.plot(time, noisy_data, label='Noisy Data')
    plt.plot(time, func(time, *params), label='Fit')
    plt.title("Noisy Cosine Function and line of fit")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    #Defining functions to calculate the first and second derivative of the fitted curve
    #the Newton conjugate gradient trust-region algorithm required first and second derivative
    def fit_curve(t):
        return Ampltd_fit * np.cos(freq_fit * t)
    def derivat1(t):
        return -freq_fit * Ampltd_fit * np.sin(freq_fit * t)
    def derivat2(t):
        return -freq_fit**2 * Ampltd_fit * np.cos(freq_fit * t)

    #Minimizing the fitted curve using two different optimization methods
    #first using the Newton conjugate gradient trust-region algorithm and then I chose
    #the BFGS Algorithm
    minimize1 = optimize.minimize(fit_curve, x0=0, method='trust-ncg', jac=derivat1, hess=derivat2)
    minimize2 = optimize.minimize(fit_curve, x0=0, method='BFGS')

    #Printing the minimum values obtained by the two optimization methods
    #and comparing their results
    print ('Minimization using Trust-ncg: ', minimize1.fun)
    print ('Minimization using BFGS:', minimize2.fun)
    print ('Comparing the two methods Trust/ncg - BFGS = ', minimize1.fun-minimize2.fun)

    #Returning the time and noisy data arrays
    return time, noisy_data

#Calling the function with some sample arguments
f = noisy_func(100, 1, np.pi/4, 0.5, 0.6)
#printing the noisy data and times in the main script
print(f[0])
print(f[1])


#d)
#Define the function
def f(x):
    #try-except block to handle the case when x=0 and f(x) is undefined
    #used from C++ Logic
    try:
        return x * np.sin(3/x) + 0.25
    except ZeroDivisionError:
        # If x = 0, return 0 
        return 0

#Initialize an empty list to store the roots found using brentq
roots_brentq = []

#Initialize x1 to be -1
x1 = -1

#WhileLoop through x values from -1 to 1 with a step size of 0.02
#to catch all the roots
while x1 < 1:
    x2 = x1 + 0.02
    #check if the function values at x1 and x2 have opposite signs, indicating a root
    if f(x1) * f(x2) < 0:
        #use brentq to find the root between x1 and x2
        root = brentq(f, x1, x2)
        #check if the root has already been found, and add it to the list
        if root not in roots_brentq:
            roots_brentq.append(root)
    x1 = x2

#generate an array of x values for plotting the function and roots
x_range = np.linspace(-1, 1, 1000)

#initialize an empty list to store the roots found using newton
roots_newton = []

#set the tolerance for determining whether two roots are the same
#according to the documentation of the function
tolerance = 1e-8

#Loop through the x range and use newtons method to find any roots
for x0 in x_range:
    try:
        # Use newton to find the root starting from x0
        root_newton = newton(f, x0, maxiter=100)
        #check if the root is already in the list, and add it if it isn't
        is_duplicate = False
        #I would get 30 values without this
        for root in roots_newton:
            if abs(root - root_newton) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            roots_newton.append(root_newton)
    except RuntimeError:
        #If newtons method fails to converge do nothing and move on
        pass

#create a figure with two subplots for plotting the function and roots
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,9))

#plot f(x) on the first subplot
ax[0].plot(x_range, f(x_range), label='f(x)')

#plot the roots found using brentq on the first subplot
#using red dots
for root1 in roots_brentq:
    ax[0].plot(root1, f(root), 'r.', label='brentq roots')

#plot f(x) on the second subplot
ax[1].plot(x_range, f(x_range), label='f(x)')

#plot the roots found using newton on the second subplot
#using red dots
for root2 in roots_newton:
    ax[1].plot(root2, f(root), 'r.', label='newton roots')

#set the labels and titles for the subplots
ax[0].set_xlabel('x')
ax[0].set_ylabel('f(x)')
ax[0].set_title('Roots found using Brentq')
ax[1].set_xlabel('x')
ax[1].set_ylabel('f(x)')
ax[1].set_title('Roots found using Newton')

#showing the plot
plt.show()


#d)
#load data from output.csv from last project
#solution of differential equation
data = np.loadtxt('OutputData.csv', delimiter=',')

# Extract time and x(t) data
time = data[0]
x = data[1]

#perform Fouriier transform
#using the right frequency spacing
freq = np.fft.fftfreq(len(time), d=time[1] - time[0])
x_freq = np.fft.fft(x)

#plot results in different colors
plt.plot(freq, np.real(x_freq), color='red', label='real part')
plt.plot(freq, np.imag(x_freq), color='blue', label='imaginary part')
plt.legend()
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Fourier transform of x(t)')
plt.show()

#f)
#load data from nsl23_5.dat file
data = np.loadtxt('nsl23_5.dat')
#extracting the 5th and 7th column
v1 = data[:, 4]
v2 = data[:, 6]

#define linear function
def fit_linear(x, a, b):
    return a * x + b

#define quadratic function
def fit_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

#define fifth degree function
def fit_fifth(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

#define sinusoidal function
def fit_sin(a, b, c, d, x):
    return a*np.sin(b - x) + c * x**2 + d

#fit linear function to the data
params1, params_covariance1 = scipy.optimize.curve_fit(fit_linear, v1, v2)
#fit quadratic function to the data
params2, params_covariance2 = scipy.optimize.curve_fit(fit_quadratic, v1, v2)
#fit fifth order function to the data
params3, params_covariance3 = scipy.optimize.curve_fit(fit_fifth, v1, v2)
#fit sinusoidal function to the data
params4, params_covariance4 = scipy.optimize.curve_fit(fit_sin, v1, v2)

#get fitted values for each function
fit1 = fit_linear(v1, *params1)
fit2 = fit_quadratic(v1, *params2)
fit3 = fit_fifth(v1, *params3)
fit4 = fit_sin(v1, *params4)

#perform FFT of original data
v2_fourier = fftpack.fft(v2)
freq = fftpack.fftfreq(v2.size, d=v1[1]-v1[0])

#perform FFT of sinusoidal fit
sin_fourier = fftpack.fft(fit4)

#Fourieer back transform of original data
v2_ifft = fftpack.ifft(v2_fourier)

#fourier back transform of sinusoidal fit
v2_sin_ifft = fftpack.ifft(sin_fourier)

#plotting results
fig, axs = plt.subplots(4, 2, figsize=(10, 14))

#plot original data and linear fit
axs[0, 0].plot(v1, v2, label='Original Data')
axs[0, 0].plot(v1, fit1, color='yellow', label='Linear Fit')
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('x(t)')
axs[0, 0].set_title('Linear Fit of Original Data')
axs[0, 0].legend()

# Plot original data and quadratic fit
axs[0, 1].plot(v1, v2, label='Original Data')
axs[0, 1].plot(v1, fit2, color='red', label='Quadratic Fit')
axs[0, 1].set_xlabel('t')
axs[0, 1].set_ylabel('f(t)')
axs[0, 1].set_title('Quadratic Fit of Original Data')
axs[0, 1].legend()

#plot original data and fifth degree fit
axs[1, 0].plot(v1, v2, label='Original Data')
axs[1, 0].plot(v1, fit3, color='green', label='Fifth Degree Fit')
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('f(t)')
axs[1, 0].set_title('Fifth Degree polynomial Fit of Original Data')
axs[1, 0].legend()

#plot original data and sinusoidal fit
axs[1, 1].plot(v1, v2, label='Original Data')
axs[1, 1].plot(v1, fit4, color='black', label='Sinusoidal Fit')
axs[1, 1].set_xlabel('t')
axs[1, 1].set_ylabel('f(t)')
axs[1, 1].set_title('Sinusoidal Fit of Original Data')
axs[1, 1].legend()

#plot Fourier transform of original data
axs[2, 0].plot(freq, np.abs(v2_fourier))
axs[2, 0].set_xlabel('Frequency')
axs[2, 0].set_ylabel('Amplitude')
axs[2, 0].set_title('FFT of Original Data')

#plot Fourier transform of sinusoidal fit
axs[2, 1].plot(freq, np.abs(sin_fourier))
axs[2, 1].set_xlabel('Frequency')
axs[2, 1].set_ylabel('Amplitude')
axs[2, 1].set_title('FFT of Sinusoidal Fit')

#plot inverse Fourier transform of original data
#to show it is exactly the same as above original data
axs[3, 0].plot(v1, v2_ifft.real)
axs[3, 0].set_xlabel('t')
axs[3, 0].set_ylabel('f(t)')
axs[3, 0].set_title('Inverse FFT of Original Data')

#plot inverse Fourier transform of sinusoidal fit
#to show it is the same as the above sinusoidal fit
axs[3, 1].plot(v1, v2_sin_ifft.real)
axs[3, 1].set_xlabel('t')
axs[3, 1].set_ylabel('f(t)')
axs[3, 1].set_title('Inverse FFT of Sinusoidal Fit')

#showing the results
plt.tight_layout()
plt.show()

