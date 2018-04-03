import numpy as np

from openmdao.api import Problem, ScipyOptimizeDriver
import scipy.io as si
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

E = 1.
L = 1.
b = 0.1
volume = 0.01

num_elements = 50
num_nodes = num_elements + 1
prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()

prob.run_driver()
K = prob['states_comp.K']
#print(K)
si.savemat('KK.mat',mdict={'K':K})
d = prob['displacements_comp.d']
si.savemat('d.mat',mdict={'d':d})
force_vector = np.zeros(2 * num_nodes)
force_vector[-2] = -1.
force_vector = np.hstack((force_vector,np.zeros(2)))
si.savemat('f.mat',mdict={'f':force_vector})
print(prob['inputs_comp.h'])