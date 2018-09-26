from openmdao.api import ImplicitComponent, Group, IndepVarComp, Problem
import numpy as np
# from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import gmres as gmres_solve
from demo.planeFrame2D.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss

def computeKandF(force_file="/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"):
    SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz'

    info = np.loadtxt(force_file,delimiter=',',skiprows=1)
    num_elements = info.__len__()
    num_nodes = info.__len__()

    np_mesh = np.zeros(shape=(num_elements,2,3),dtype=object)
    __MIN_NODE_ID = np.min(info[:,0])


    ###### assemble np_mesh
    for element_ID in range(num_elements):
        for local_node_ID in range(2):
            node_ID = int(info[(element_ID+local_node_ID)%num_nodes,0]) - __MIN_NODE_ID
            x,y = info[(element_ID+local_node_ID)%num_nodes,1:3]
            np_mesh[element_ID,local_node_ID,:] = x,y,node_ID


    E = 72E06      # modulus of elasticity
    I = 5E-05        # moment of inertia
    A = 0.1         # cross-sectional area
    L = 1           # length
    try:
        K = ss.load_npz('/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz')
    except:
        print('cant find K, GENERATING')
        ##### generate sparse pattern
        template_nonzero_matrix = np.zeros(shape=(2, ASSEMBLE_ENTRIES * num_elements))
        for element_ID in range(num_elements):
            i = int(np_mesh[element_ID,0,2]+1)
            j = int(np_mesh[element_ID,1,2]+1)
            generate_template(template_nonzero_matrix,i,j,element_ID)

        K = csc_matrix((np.zeros(shape=(ASSEMBLE_ENTRIES*num_elements)),(template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(3 * num_nodes, 3 * num_nodes))

        ##### generate global matrix
        for element_ID in range(num_elements):
            x1,y1 = np_mesh[element_ID,0,0:2]
            x2,y2 = np_mesh[element_ID,1,0:2]
            i = int(np_mesh[element_ID, 0, 2]+1)
            j = int(np_mesh[element_ID, 1, 2]+1)
            L = PlaneFrameElementLength(x1,y1,x2,y2)
            C,S = PlaneFrameElementCS(x1,y1,x2,y2,L)
            k = PlaneFrameElementStiffness(E,A,I,L,C,S)
            PlaneFrameAssemble(K, k, i, j)
        ss.save_npz('K_planeFrame2D.mat', K)


    ### FORCE
    # SPEED_OF_SOUND=295.1m/s
    force_file = np.loadtxt(force_file,delimiter=',',skiprows=1)
    force_dict = {}
    force_dict['GLOBALIDX'] = np.array(force_file[:,0],dtype=int) - __MIN_NODE_ID
    force_dict['X'] = force_file[:,1]
    force_dict['Y'] = force_file[:,2]
    force_dict['PRESS'] = force_file[:,3]
    force_dict['PRESSCO'] = force_file[:,4]
    force_dict['MACHNUM'] = force_file[:,5]


    force = np.zeros(shape=(3*num_nodes,1))
    ## compute dimensional force
    for force_ID in range(len(force_file)):
        current_point_id =  int(force_file[force_ID,0] - __MIN_NODE_ID)
        pre_point_id = int(force_file[force_ID-1,0] - __MIN_NODE_ID)
        post_point_id = int(force_file[(force_ID+1)%len(force_file),0] - __MIN_NODE_ID)
        x0,y0 = force_file[force_ID-1,1:3]
        x1,y1 = force_file[force_ID,1:3]
        x2,y2 = force_file[(force_ID+1)%len(force_file),1:3]
        #fx,fy = force_file[force_ID,3] * computeN_VEC(x0,y0,x1,y1,x2,y2)
        fx,fy = computeN_VEC(x0,y0,x1,y1,x2,y2)
        force[3*current_point_id:3*current_point_id+2,0] = fx,fy


    #### assume that U_y is fixed
    """
    KU=F
    U = [U1_x,U1_y,theta_1,...,UN_x,UN,y,theta_N]
    F = [F1_x,F1_y,M1,...,FN_x,FN_y,MN]
    """
    discard_row = set(np.arange(0,num_nodes*3,3))
    saved_row = list(set(np.arange(0,3*num_nodes)) - discard_row)
    #saved_row = np.arange(1,3*num_nodes,3)
    A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]
    F = force[saved_row]
    return A,F



class LinearSysComp(ImplicitComponent):
    """
    A Simple Implicit Component representing a Linear System Kd=f

    """

    def initialize(self):
        self.metadata.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        size = num_elements
        self.add_input('K', shape=(size, size))
        self.add_input('f', shape=size)
        self.add_output('d', shape=size)
        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('d', 'K', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['d'] = np.dot(inputs['K'], outputs['d']) - inputs['f']

    def linearize(self, inputs, outputs, partials):
        num_elements = self.metadata['num_elements']
        size = num_elements
        # self.lu = lu_factor(inputs['K'])
        self.K = inputs['K']
        partials['d', 'K'] = np.outer(np.ones(size), outputs['d']).flatten()
        partials['d', 'd'] = inputs['K']

    def solve_nonlinear(self, inputs, outputs):
        # self.lu = lu_factor(inputs['K'])
        # outputs['d'] = lu_solve(self.lu, inputs['f'])
        self.K = inputs['K']
        outputs['d'] = gmres_solve(inputs['K'], inputs['f'])[0]

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            # d_outputs['d'] = lu_solve(self.lu, d_residuals['d'], trans=0)
            d_outputs['d'] = gmres_solve(self.K, d_residuals['d'], trans=0)
        else:
            # d_residuals['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)
            d_residuals['d'] = gmres_solve(self.K, d_outputs['d'], trans=1)

class LsGroup(Group):

    def initialize(self):
        self.metadata.declare('num_elements', int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements+1
        inputs_comp = IndepVarComp()
        K,F = computeKandF()
        K = K.toarray()
        inputs_comp.add_output('K', shape=(2*num_nodes,2*num_nodes), val=K)
        inputs_comp.add_output('F',shape=(2*num_nodes),val=F)
        self.add_subsystem('inputs_comp', inputs_comp)

        d_comp = LinearSysComp(num_elements=num_elements)
        self.add_subsystem('d_comp', d_comp)

        self.connect('inputs_comp.K', 'd_comp.K')
        self.connect('inputs_comp.F', 'd_comp.F')

num_elements = 10
prob = Problem(model=LsGroup(num_elements=num_elements))
prob.setup()
prob.run_driver()
