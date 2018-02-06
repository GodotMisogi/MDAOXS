#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  prob_driver_paraboloid.py
#       Author @  xshi
#  Change date @  11/18/2017 10:52 AM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
from openmdao.api import Problem, \
    ScipyOptimizer, ExecComp, IndepVarComp

# We'll use the component that was defined in the last tutorial
from paraboloid import Paraboloid

# build the model
prob = Problem()
indeps = prob.model.add_subsystem('indeps', IndepVarComp())
indeps.add_output('x', 3.0)
indeps.add_output('y', -4.0)

prob.model.add_subsystem('parab', Paraboloid())

# define the component whos output will be constrained
prob.model.add_subsystem('const', ExecComp('g = x + y'))

prob.model.connect('indeps.x', ['parab.x', 'const.x'])
prob.model.connect('indeps.y', ['parab.y', 'const.y'])

# setup the optimization
prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'COBYLA'

prob.model.add_design_var('indeps.x', lower=-50, upper=50)
prob.model.add_design_var('indeps.y', lower=-50, upper=50)
prob.model.add_objective('parab.f_xy')

# to add the constraint to the model
prob.model.add_constraint('const.g', lower=0, upper=10.)
# prob.model.add_constraint('const.g', equals=0.)

prob.setup()
prob.run_driver()

# minimum value
print(prob['parab.f_xy'])
# location of the minimum
print(prob['indeps.x'])
print(prob['indeps.y'])

