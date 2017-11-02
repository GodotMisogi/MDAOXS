#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  fctDefTest.py
#       Author @  xshi
#  Change date @  11/1/2017 6:33 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import openmdao as om
from numpy.random import random as rng

def randomWalk(steps):
    """
    :param steps: total walk steps
    :return: trajectory of random walk starts from (0,0)
    """
    x = np.array([0])
    y = np.array([0])
    for i in range(steps):
        dx = (rng()>0.5) * 2 + -1
        dy = (rng()>0.5) * 2 + -1
        x = np.hstack((x, x[-1]+dx))
        y = np.hstack((y, y[-1]+dy))
    #print(len(x))
    return np.hstack((x,y))

