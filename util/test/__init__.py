#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  timing.py
#       Author @  xshi
#  Change date @  11/19/2017 7:58 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
from contextlib import contextmanager
import time
@contextmanager
def timeblock ( label ):
    start = time.perf_counter ()
    try:
        yield
    finally:
        end = time.perf_counter ()
        print ('{}:{} '. format (label , end - start ))