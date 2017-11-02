#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  basicOp.py
#       Author @  xshi
#  Change date @  10/31/2017 8:08 AM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import matplotlib.pyplot as plt
import numpy as np, matplotlib.pyplot as pl
from scipy.integrate import odeint
# Import ODE to integrate:
def F(y, t):
    """
    Return derivatives for second-order ODE y'' = -y.
    """
    dy = [0, 0]  # Create a list to store derivatives.
    dy[0] = y[1]  # Store first derivative of y(t).
    dy[1] = -y[0]  # Store second derivative of y(t).
    return dy

# Create array of time values to study:
t_min = 0; t_max = 10; dt = 0.1
t = np.arange(t_min, t_max+dt, dt)
initial_conditions = [ (1.0, 0.0), (0.0, 1.0) ]
plt.figure() # Create figure; add plots later.
for y0 in initial_conditions:
    y = odeint(F, y0, t)
    plt.plot(t, y[:, 0], linewidth=2)

skip = 5
t_test = t[::skip]
plt.plot(t_test, np.cos(t_test), 'bo') # Exact solution for y0 = (1,0)
plt.plot(t_test, np.sin(t_test), 'go') # Exact solution for y0 = (0,1)
