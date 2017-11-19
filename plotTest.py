#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  plotTest.py
#       Author @  xshi
#  Change date @  11/1/2017 3:02 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import random as rng
import scipy as sp
import openmdao as om

# 2D
plt.figure(1)
data = np.loadtxt(r"dataSet\PMLSdata\01HIVseries\HIVseries.csv",delimiter=',')
x = data[:,0]
y = data[:,1]
ey = np.random.rand(len(x))
ex = np.random.rand(len(y))
#plt.plot(x,y,'r--')
plt.errorbar(x,y,yerr=ey,xerr=ex)
#plt.scatter(x,y)
ax = plt.gca()
ax.set_xlabel('$\mu$')
ax.set_title('x,y')
plt.ylabel("y")
plt.legend(['aaa','a'])

# 3D
fig = plt.figure(2)  # create a new figure
ax = Axes3D(fig)    # create 3D plotter attached to figure
t = np.linspace(0, 5*np.pi, 501)
ax.plot(np.cos(t),np.sin(t),t)


# 2 figures at once
plt.figure(3)
x = np.linspace(0, 1, 51)
y1 = np.exp(x)
y2 = x**2
plt.plot(x, y1, x, y2)



#
plt.figure(4)
num_curves = 3
x = np.linspace(0, 1, 51)
y = np.zeros( (x.size, num_curves) )
for n in range(num_curves):
    y[:, n] = np.sin((n+1) * x * 2 * np.pi)
plt.plot(x, y)

ax=plt.gca()
lines = ax.get_lines()
plt.setp(lines[0],color='r')

plt.setp(lines[1],color='g')

plt.setp(lines[2],color='b')
plt.legend(['1','2','3'])
plt.show()

plt.close('all')
# subplot
t = np.linspace(0, 1, 101)
plt.figure()
plt.subplot(2, 2, 1); plt.hist(np.random.random(20)) # Upper left
plt.subplot(2, 2, 2); plt.plot(t, t**2, t, t**3 - t) # Upper right
plt.subplot(2, 2, 3); plt.plot(np.random.random(20), np.random.random(20), 'r*') # Lower left
plt.subplot(2, 2, 4); plt.plot(t*np.cos(10*t), t*np.sin(10*t)) # Lower right


# save
fig = plt.gcf() # Get current figure object.
fig.canvas.get_supported_filetypes()
plt.savefig("greatest_figure_ever.pdf")

# hist
data = rng(100)
plt.hist(data)
log2bins = np.logspace(-8, 0, num=9, base=2)
log2bins[0] = 0.0 # Set first bin edge to zero instead of 1/256.
plt.hist(data, bins=log2bins)
# bar
counts, bin_edges, _ = plt.hist(data)
bin_size = bin_edges[1] - bin_edges[0]
new_widths = bin_size * counts / counts.max()
plt.bar(bin_edges[:-1], counts, width=new_widths, color=['r','g','b'])

#animation
""" Make a movie out of the steps of a two-dimensional random walk. """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import random as rand

# Set number of steps for each random walk.
num_steps = 100

# Create an empty figure of the desired size.
bound = 20
fig = plt.figure()      # must have figure object for movie
ax = plt.axes(xlim=(-bound, bound), ylim=(-bound, bound))

# Create a line and a point with no data.  They will be updated during each
# frame of the animation.
(my_line,) = ax.plot([], [], lw=2)              # line to show path
(my_point,) = ax.plot([], [], 'ro', ms=9)       # dot to show current position

# Generate the random walk data.
x_steps = 2*(rand(num_steps) < 0.5) - 1     # generate random steps +/- 1
y_steps = 2*(rand(num_steps) < 0.5) - 1
x_coordinate = x_steps.cumsum()             # sum steps to get position
y_coordinate = y_steps.cumsum()

# This function will generate each frame of the animation.
# It adds all of the data through frame n to a line
# and moves a point to the nth position of the walk.
def get_step(n, x, y, this_line, this_point):
    this_line.set_data(x[:n+1], y[:n+1])
    this_point.set_data(x[n], y[n])

# Call the animator and create the movie.
my_movie = animation.FuncAnimation(fig, get_step, frames=num_steps, \
                    fargs=(x_coordinate,y_coordinate,my_line,my_point) )

# Save the movie in the current directory.
# *** THIS WILL CAUSE AN ERROR UNLESS FFMPEG OR MENCODER IS INSTALLED. ***
# my_movie.save('random_walk.mp4', fps=30)
