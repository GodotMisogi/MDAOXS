import numpy as np

from openmdao.api import Problem, ExplicitComponent, Group, IndepVarComp, PETScVector
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs

from openmdao.utils.mpi import MPI

if not MPI:
    raise unittest.SkipTest()

rank = MPI.COMM_WORLD.rank
size = 15


class DistribComp(ExplicitComponent):
    def __init__(self, size):
        super(DistribComp, self).__init__()
        self.size = size
        self.distributed = True

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * -3.0

    def setup(self):
        comm = self.comm
        rank = comm.rank

        # this results in 8 entries for proc 0 and 7 entries for proc 1 when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(sizes[rank], float))


class Summer(ExplicitComponent):

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        # this results in 8 entries for proc 0 and 7 entries for proc 1
        # when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(self.comm.size, self.size)
        start = offsets[rank]
        end = start + sizes[rank]

        # NOTE: you must specify src_indices here for the input. Otherwise,
        #       you'll connect the input to [0:local_input_size] of the
        #       full distributed output!
        self.add_input('invec', np.ones(sizes[self.comm.rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('out', 0.0)

    def compute(self, inputs, outputs):
        data = np.zeros(1)
        data[0] = np.sum(self._inputs['invec'])
        total = np.zeros(1)
        self.comm.Allreduce(data, total, op=MPI.SUM)
        self._outputs['out'] = total[0]


p = Problem(model=Group())
top = p.model
top.add_subsystem("indep", IndepVarComp('x', np.zeros(size)))
top.add_subsystem("C2", DistribComp(size))
top.add_subsystem("C3", Summer(size))

top.connect('indep.x', 'C2.invec')
top.connect('C2.outvec', 'C3.invec')

p.setup(vector_class=PETScVector)

p['indep.x'] = np.ones(size)

p.run_model()

print(p['C3.out'])