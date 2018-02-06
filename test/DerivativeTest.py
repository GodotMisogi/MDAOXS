#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  DerivativeTest.py.py
#       Author @  xshi
#  Change date @  11/19/2017 5:42 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import numpy as np
#import matplotlib.pyplot as plt
import scipy as sp
from openmdao.api import Problem, Group, IndepVarComp
from test.ExplicitComp.ParaboloidFD import ParaboloidFD
# for performance profiling
from openmdao.devtools import iprofile as tool

# OR for memory profiling
# from openmdao.devtools import iprof_mem as tool

# OR for call tracing
# from openmdao.devtools import itrace as tool

model = Group()


ivc = IndepVarComp()
ivc.add_output('x', 3.0)
ivc.add_output('y', -4.0)
model.add_subsystem('des_vars', ivc)
model.add_subsystem('parab_comp', ParaboloidFD())

model.connect('des_vars.x', 'parab_comp.x')
model.connect('des_vars.y', 'parab_comp.y')

prob = Problem(model)
prob.setup()
tool.start()
prob.run_model()
tool.stop()
print(prob['parab_comp.f_xy'])

prob['des_vars.x'] = 5.0
prob['des_vars.y'] = -2.0
prob.run_model()
print(prob['parab_comp.f_xy'])
