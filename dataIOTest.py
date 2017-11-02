#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  dataIOTest.py
#       Author @  xshi
#  Change date @  11/1/2017 3:03 PM
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

data = np.loadtxt(r"dataSet\PMLSdata\01HIVseries\HIVseries.csv",delimiter=',')
x = data[:,0]
y = data[:,1]