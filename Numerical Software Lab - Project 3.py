"""
Numerical Software Lab - Project 3

@author: Getuar Rexhepi

email: grexhepi@constructor.university

28 March, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import spherical_jn, spherical_yn


#a)
#A function that as input gets an array of coefficients, and the variable x
#and returns a polynomial which will be used to integrate in the next fnct.
def polynomial_function(x, p):
    return np.polyval(p, x)

#A function that as input gets an array of coefficients and the starting 
#and ending point of integration, and returns the numerical value and 
#its error in two seperate variables (result and error)
def integrate_polynomial(p, a, b):
    result, error = quad(polynomial_function, a, b, args=(p,))
    return result, error

#a testing array of coefficients 
#and testing starting and ending points
p = [3,2,1]
a = 1
b = 5
integrate_polynomial (p,a,b)

#printing the input and output for testing purposes
print ('The integral of polynomial with coefficients:',p,'from',a,'to',b,'is:',integrate_polynomial(p,a,b)[0],
      'with an error of:',integrate_polynomial(p,a,b)[1])

#b)
#Function that plots sperical bessel functions in 3D
#and as input takes its kind(or model), the x_range,
#the y_range and the filename
def spherical_bessel(model, n, x_range, y_range, filename):
    #creating arrays for x and y values, using the x and y
    #range, 50 linspaced points (enough for our testing)
    x_values = np.linspace(x_range[0], x_range[1], 50)
    y_values = np.linspace(y_range[0], y_range[1], 50)
    #np.meshgrid(x_values, y_values) takes two arrays and 
    #creates two arrays, mesh_x and mesh_y, that correspond 
    #to all pairs of values from the x and y_values arrays.    
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    #this grid of points is useful for evaluating the function
    #mesh_r computes the square root of the sum of squares of x and y
    #and stores it as an element of the array
    mesh_r = np.sqrt(mesh_x**2 + mesh_y**2)

    #if the spherical bessel function is of kind
    # jn mesh_z (the output) is computed using 
    #spherical_jn
    #else if spherical bessel function is of kind
    # yn mesh_z (the output) is computed using 
    #spherical_yn
    if model == 'jn':
        mesh_z = spherical_jn(n, mesh_r)
    elif model == 'yn':
        mesh_z = spherical_yn(n, mesh_r)
    #if something else is given than print a warning message
    else:
        print ("Kind can be only jn or yn! Please check again.")

    #plotting the 3D figure    
    fig = plt.figure();
    #adding a subplot, projecting should be 3d
    ax = fig.add_subplot(111, projection='3d')
    #Creating a surface plot, and I used a color map to make 
    #the characteristics of the figure visible
    ax.plot_surface(mesh_x, mesh_y, mesh_z, cmap='jet')
    #setting the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(model + "_" + str(n) + "(r)")
    #setting the axis tittle using string concatenation
    #showing the kind and the order
    ax.set_title("Spherical Bessel Function " + model + "_" + str(n) + "(r)")
    #defining the number of ticks
    num_ticks = 5
    
    #creating 5 ticks for the x and y axis (equispaced)
    x_ticks = np.linspace(x_range[0], x_range[1], num_ticks)
    y_ticks = np.linspace(y_range[0], x_range[1], num_ticks)
    
    #creating 4 ticks for z axis from minimum to maximum such that
    #we are sure that everytime we are covering all points in range
    z_ticks = np.linspace(mesh_z.min(), mesh_z.max(), num_ticks-1)

    #setting ticks for respective axes
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    
    #saving the plot as a pdf, and not showing it on the screen
    plt.savefig(filename +'.pdf')
    
    plt.close()


#testing array for x and y range
x = (-10,10)
y = (-10,10)
#testing our function using a jn bessel function
#of order 0, the aforementioned x and y arrays and
#saving it as output.pdf
spherical_bessel('jn', 0, x, y, 'output')


#no input and output testing for the second function,
#since it was required to not show the plot, and export it
# as pdf!