#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  Paraboloid.py
#       Author @  xshi
#  Change date @  11/19/2017 5:44 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
"""
Introduction
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from openmdao.core.explicitcomponent import ExplicitComponent

class Paraboloid(ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0


if __name__ == "__main__":
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.core.indepvarcomp import IndepVarComp

    model = Group()
    ivc = IndepVarComp()
    ivc.add_output('x', 3.0)
    ivc.add_output('y', -4.0)
    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('parab_comp', Paraboloid())

    model.connect('des_vars.x', 'parab_comp.x')
    model.connect('des_vars.y', 'parab_comp.y')

    prob = Problem(model)
    prob.setup()
    prob.run_model()
    print(prob['parab_comp.f_xy'])

    prob['des_vars.x'] = 5.0
    prob['des_vars.y'] = -2.0
    prob.run_model()
    print(prob['parab_comp.f_xy'])