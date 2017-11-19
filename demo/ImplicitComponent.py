#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  ImplicitComponent.py
#       Author @  xshi
#  Change date @  11/18/2017 5:55 PM
#        Email @  xshi@kth.se
#  Description @  master thesis project
# ********************************************************************
from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp, NewtonSolver, ScipyOptimizer

class ImpWithInitial(ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0.)

        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def solve_nonlinear(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    # user-provided partial derivatives
    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']

        partials['x', 'a'] = x ** 2
        partials['x', 'b'] = x
        partials['x', 'c'] = 1.0
        partials['x', 'x'] = 2 * a * x + b

        self.inv_jac = 1.0 / (2 * a * x + b)

    # implement partial derivative in a matrix-free way
    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        if mode == 'fwd':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                if 'a' in d_inputs:
                    d_residuals['x'] += x ** 2 * d_inputs['a']
                if 'b' in d_inputs:
                    d_residuals['x'] += x * d_inputs['b']
                if 'c' in d_inputs:
                    d_residuals['x'] += d_inputs['c']
        elif mode == 'rev':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                if 'a' in d_inputs:
                    d_inputs['a'] += x ** 2 * d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += x * d_residuals['x']
                if 'c' in d_inputs:
                    d_inputs['c'] += d_residuals['x']

    # Solves a linear system where the matrix is
    # d_residuals/d_outputs or its transpose.
    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']

    # set the initial guess
    def guess_nonlinear(self, inputs, outputs, resids):
        # Solution at 1 and 3. Default value takes us to -1 solution. Here
        # we set it to a value that will tke us to the 3 solution.
        outputs['x'] = 5.0

if __name__ == '__main__':
    prob = Problem()
    model = prob.model = Group()

    model.add_subsystem('pa', IndepVarComp('a', 1.0))
    model.add_subsystem('pb', IndepVarComp('b', 1.0))
    model.add_subsystem('pc', IndepVarComp('c', 1.0))
    model.add_subsystem('comp2', ImpWithInitial())
    model.connect('pa.a', 'comp2.a')
    model.connect('pb.b', 'comp2.b')
    model.connect('pc.c', 'comp2.c')

    model.nonlinear_solver = NewtonSolver()
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.options['max_sub_solves'] = 1
    #model.linear_solver = ScipyOptimizer()

    prob.setup(check=False)

    prob['pa.a'] = 1.
    prob['pb.b'] = -4.
    prob['pc.c'] = 3.

    prob.run_model()

    print(prob['comp2.x'])