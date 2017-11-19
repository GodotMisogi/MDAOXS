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


photo = plt.imread('bwCat.tif')
photo.shape
photo.dtype
plt.imshow(photo)
plt.set_cmap('gray') # Use grayscale for black and white image.
plt.axis('off') # Get rid of axes and tick marks.
fig = plt.gcf() # Get current figure object.
fig.set_facecolor('white') # Set background color to white.

new_cat = (photo<photo.mean())
plt.imshow(new_cat)