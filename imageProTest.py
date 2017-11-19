#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  imageProTest.py
#       Author @  xshi
#  Change date @  11/2/2017 3:26 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sim
image = plt.imread('bwCat.tif')
v = np.arange(-25, 26)
X, Y = np.meshgrid(v, v)
gauss_filter = np.exp(-0.5*(X**2/2 + Y**2/45))

my_filter = np.array( [ [0, -1, 0], [-1, 4, -1], [0, -1, 0] ] )
combined_filter = sim.convolve(gauss_filter,my_filter)
new_image = sim.convolve(image,combined_filter)
#image = sim.uniform_filter(image)
#np.sum(image-new_image)
plt.figure()
plt.imshow(new_image)
new_image = sim.convolve(image,my_filter)
plt.imshow(new_image, vmin=0, vmax=0.5*image.max())