
"""
Numerical Software Lab - Project 2

@author: Getuar Rexhepi

email: grexhepi@constructor.university

16 March, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

#a) 
#Function that calculates the scalar triple product of 3 vectors, and it works for 3D vectors
def scalar_triple(v1,v2,v3):
    #create a variable that holds the result of the scalar triple product calculated
    #using the formula that can be found in any calculus book
    result = v1[0]*(v2[1]*v3[2]-v2[2]*v3[1])+v1[1]*(v2[2]*v3[0]-v2[0]*v3[2])+v1[2]*(v2[0]*v3[1]-v2[1]*v3[0])
    #rerturn the result stored in the variable
    return result

#Function that calculates the vector triple product of 3 vectors, and it works stritly for vectors of 3 dimensions
def vector_triple(v1,v2,v3):
    #the function has 3 arguments (vectors)
    #first it creates another vector that is a result of the cross product of the second and third vector
    #using indices we can simulate a cross product
    v23= [v2[1]*v3[2]-v3[1]*v2[2],-(v2[0]*v3[2]-v3[0]*v2[2]),v2[0]*v3[1]-v3[0]*v2[1]]
    #using the upper vector, we find the final result by cross product of the first vector and vector(2x3)
    result = [v1[1]*v23[2]-v23[1]*v1[2],-(v1[0]*v23[2]-v23[0]*v1[2]),v1[0]*v23[1]-v23[0]*v1[1]]
    #the result is stored in a variable which is finally returned by the function
    return result

#using 3 arbitrary vectors of dimension 3 for testing purposes
v = [1,2,3]
u = [3,2,1]
w = [4,1,2]

#printing the input vectors and the output result (scalar in the first case and a vector in the second)
print ("The triple scalar product of vectors :",v,u,w,"is : ",scalar_triple (v,u,w))
print ("The triple vector product of vectors :",v,u,w,"is : ",vector_triple (v,u,w))

#b)
#Function that takes a file name, title, x_label and y_label as input with default values for the latter 3
def data_read (filename, title="Input data", x_label="x axis", y_label="y axis"):
    #it reads the data form the file and outputs the x and y values in two seperate arrays
    #no skiprows because we don't have a header and no rows will be skipped
    data = np.loadtxt (filename, skiprows = 0)
    #storing the x and y values in two seperate arrays
    x = data[:, 0]
    y = data[:, 1]
    
    #computing the maximum and minimum for the two arrays as we need it later for the modifying the axes
    x_min = np.min (x)
    y_min = np.min (y)
    x_max = np.max (x)
    y_max = np.max (y)
    
    #computing the 5% margin and using it for plotting later
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    
    #creating the figure and axis
    fig, ax = plt.subplots()
    #setting the figure title
    ax.set_title(title)
    #setting the figure labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    #plotting the figure as x versus y values
    ax.plot(x, y)
    #setting the x and y axis limit and using the 5% margin at the both sides
    ax.set_xlim (x_min - margin_x, x_max + margin_x)
    ax.set_ylim (y_min - margin_y, y_max + margin_y)
    
    #saving the output figure as pdf
    plt.savefig("output.pdf")
    plt.close()
    #renaming the filename, using replace for the new csv file
    csv_filename = filename.replace(".dat", "_new.csv")
    #saving the csv file with 1 digit to the right of the decimal point for both columns
    np.savetxt(csv_filename, list(zip(x, y)), fmt='%.1f',delimiter =',')    
    
    #returning the x and y array
    return x,y

#testing the output using the file provided in moodle
data_read ('test_data2.dat')

