"""
Numerical Software Lab - Project 1

@author: Getuar Rexhepi

email: grexhepi@constructor.university

27 February, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

#a)
#Generate 201 points for x from -10 to 10
x = np.linspace(-10, 10, 201)

#Calculate y = cos(x)
y = np.cos(x)

#Calculating the taylor series fn for n = 1,3,5,7,9,11
#defining the values for n using range function
n_values = range(1,13,2)
#initializng an empty array for fN's, which will be used to store all 6 arrays
fn_arrays = []
#iterating through all n values (6)
for N in n_values:
    #initializing the array for the function of taylor series and filling with 0's for 201 points
    fn = np.zeros(len(x))
    #iterating through all the points
    for i in range(len(x)):
        #resetting the value inside the series at 0 for every iteration
        function_fn = 0
        #iterating through all N's + 1, because N starts from 0,
        #to compute the sum over the terms
        for n in range(N+1):
            #computing the taylor polynomial inside the series for nth term using the formula
            taylor = (-1)**n*x[i]**(2*n)/np.math.factorial(2*n)
            #adding the taylor polynomial for every n, simulating a series operation
            function_fn += taylor
        #assigning the value computed, to the taylor series      
        fn[i] = function_fn
    #appending the array of nth term to the array of the fN's
    fn_arrays.append(fn)
    
#b)
#plotting the cos(x) independently from taylor series
plt.plot (x, y, label ="cos(x)", color ='blue')
#creating an array of possible distinct colors for the graphs of taylor approximations
colors = ['red','orange','black','green','magenta','yellow']
#creating an array of possible lines for the graphs of taylor approximations
lines_styles = ['-', '--', '-.', ':', '-', '--']

#iterating through all 6 values to plot the graphs
for i in range(0, len(n_values)):
    #plotting the taylor approximations, and using i to iterate for colors, styles of lines and the n values
    plt.plot (x, fn_arrays[i],color = colors[i],linestyle = lines_styles[i],label=f'N={n_values[i]}')

#limiting the y-axis from -10 to 10    
plt.ylim (-10,10)
#labeling x- and y-axis
plt.xlabel('x')
plt.ylabel('cos(x)')
#defining a title for the graph
plt.title('Taylor expansion of cos(x)')
#adding a horizontal and vertical line to identify the axes
plt.axhline(color="gray", zorder=-1)
plt.axvline(color="gray", zorder=-1)
#displaying the legend of the graph in the figure
plt.legend()
#saving the figure as a pdf
plt.savefig('taylor_expansion.pdf')
plt.close()

#c)
#creating an array of 3 arbitrary sequences of A,C,G,T
sequences = ["ACGTGVATGATGGACT","GTGTACACCAGTGTGA","AAATTTCCCGGGTTTA"]
#initializing the count for A's and C's
n=m=0
#initializing a count variable for sequence number
j=1

#for loop to iterate through each element in sequence
for i in sequences:
    #iterate though all characters in each element (word)
    for c in i:
        #if a character is A increase count n
        if c=='A':
            n+=1
        #if a character is C increase count m
        elif c=='C':
            m+=1
    #print the number of occurrences for each letter
    print ("The number of C's on the sequence",j,":", m)
    print ("The number of A's on the sequence",j,":", n)
    #increasing the count variable for the sequence number
    j+=1
    #resetting the count for A's and C's, for the next word in the loop
    n=0
    m=0



