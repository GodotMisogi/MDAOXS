from tools.SU2IO import read
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import numpy as np



###################################################
##########          configure           ###########
###################################################
FORCE_FILE = "/Users/gakki/Dropbox/thesis/data/surface_flow.csv"
SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz'

GAMMA_VALUE= 1.4
AOA= 1.25
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 273.15
GAS_CONSTANT= 287.87
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_DIMENSIONALIZATION= 'DIMENSIONAL'
E = 210E06      # modulus of elasticity
I = 5E-05        # moment of inertia
A = 0.01         # cross-sectional area
L = 0           # length
# Marker of the Euler boundary (NONE = no marker)
MARKER_EULER= 'airfoil'
# Marker of the far field (NONE = no marker)
MARKER_FAR=  'farfield'

info = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
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


###################################################
##########          function            ###########
###################################################
ASSEMBLE_ENTRIES = 36


def PlaneFrameElementLength(x1,y1,x2,y2):
    """
    %PlaneFrameElementLength      This function returns the length of the
    #                             plane frame element whose first node has
    %                             coordinates (x1,y1) and second node has
    %                             coordinates (x2,y2).
    """
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def PlaneFrameElementCS(x1,y1,x2,y2,L):
    v = np.array([x2,y2]) - np.array([x1,y1])
    C = v[0]/L
    S = v[1]/L
    return C,S

def PlaneFrameElementStiffness(E,A,I,L,C,S):
    """
    %PlaneFrameElementStiffness   This function returns the element
    %                             stiffness matrix for a plane frame
    %                             element with modulus of elasticity E,
    %                             cross-sectional area A, moment of
    %                             inertia I, length L, and angle
    %                             theta (in degrees).
    %                             The size of the element stiffness
    %                             matrix is 6 x 6.
    """
    #x = theta*np.pi/180
    #C = np.cos(x)
    #S = np.sin(x)
    w1 = A*C*C + 12*I*S*S/(L*L)
    w2 = A*S*S + 12*I*C*C/(L*L)
    w3 = (A-12*I/(L*L))*C*S
    w4 = 6*I*S/L
    w5 = 6*I*C/L
    return E/L*np.array([[w1,w3,-w4,-w1,-w3,-w4],
                         [w3,w2,w5,-w3,-w2,w5],
                         [-w4,w5,4*I,w4,-w5,2*I],
                         [-w1,-w3,w4,w1,w3,w4],
                         [-w3,-w2,-w5,w3,w2,-w5],
                         [-w4,w5,2*I,w4,-w5,4*I]])


def PlaneFrameAssemble(K, k, i, j):
    """
    %PlaneFrameAssemble This function assembles the element stiffness
    % matrix k of the plane frame element with nodes
    % i and j into the global stiffness matrix K.
    % This function returns the global stiffness
    % matrix K after the element stiffness matrix
    % k is assembled.
    """
    K[3 * i - 2 - 1, 3 * i - 2 - 1] = K[3 * i - 2 - 1, 3 * i - 2 - 1] + k[1 - 1, 1 - 1]
    K[3 * i - 2 - 1, 3 * i - 1 - 1] = K[3 * i - 2 - 1, 3 * i - 1 - 1] + k[1 - 1, 2 - 1]
    K[3 * i - 2 - 1, 3 * i - 1] = K[3 * i - 2 - 1, 3 * i - 1] + k[1 - 1, 3 - 1]
    K[3 * i - 2 - 1, 3 * j - 2 - 1] = K[3 * i - 2 - 1, 3 * j - 2 - 1] + k[1 - 1, 4 - 1]
    K[3 * i - 2 - 1, 3 * j - 1 - 1] = K[3 * i - 2 - 1, 3 * j - 1 - 1] + k[1 - 1, 5 - 1]
    K[3 * i - 2 - 1, 3 * j - 1] = K[3 * i - 2 - 1, 3 * j - 1] + k[1 - 1, 6 - 1]
    K[3 * i - 1 - 1, 3 * i - 2 - 1] = K[3 * i - 1 - 1, 3 * i - 2 - 1] + k[2 - 1, 1 - 1]
    K[3 * i - 1 - 1, 3 * i - 1 - 1] = K[3 * i - 1 - 1, 3 * i - 1 - 1] + k[2 - 1, 2 - 1]
    K[3 * i - 1 - 1, 3 * i - 1] = K[3 * i - 1 - 1, 3 * i - 1] + k[2 - 1, 3 - 1]
    K[3 * i - 1 - 1, 3 * j - 2 - 1] = K[3 * i - 1 - 1, 3 * j - 2 - 1] + k[2 - 1, 4 - 1]
    K[3 * i - 1 - 1, 3 * j - 1 - 1] = K[3 * i - 1 - 1, 3 * j - 1 - 1] + k[2 - 1, 5 - 1]
    K[3 * i - 1 - 1, 3 * j - 1] = K[3 * i - 1 - 1, 3 * j - 1] + k[2 - 1, 6 - 1]
    K[3 * i - 1, 3 * i - 2 - 1] = K[3 * i - 1, 3 * i - 2 - 1] + k[3 - 1, 1 - 1]
    K[3 * i - 1, 3 * i - 1 - 1] = K[3 * i - 1, 3 * i - 1 - 1] + k[3 - 1, 2 - 1]
    K[3 * i - 1, 3 * i - 1] = K[3 * i - 1, 3 * i - 1] + k[3 - 1, 3 - 1]
    K[3 * i - 1, 3 * j - 2 - 1] = K[3 * i - 1, 3 * j - 2 - 1] + k[3 - 1, 4 - 1]
    K[3 * i - 1, 3 * j - 1 - 1] = K[3 * i - 1, 3 * j - 1 - 1] + k[3 - 1, 5 - 1]
    K[3 * i - 1, 3 * j - 1] = K[3 * i - 1, 3 * j - 1] + k[3 - 1, 6 - 1]
    K[3 * j - 2 - 1, 3 * i - 2 - 1] = K[3 * j - 2 - 1, 3 * i - 2 - 1] + k[4 - 1, 1 - 1]
    K[3 * j - 2 - 1, 3 * i - 1 - 1] = K[3 * j - 2 - 1, 3 * i - 1 - 1] + k[4 - 1, 2 - 1]
    K[3 * j - 2 - 1, 3 * i - 1] = K[3 * j - 2 - 1, 3 * i - 1] + k[4 - 1, 3 - 1]
    K[3 * j - 2 - 1, 3 * j - 2 - 1] = K[3 * j - 2 - 1, 3 * j - 2 - 1] + k[4 - 1, 4 - 1]
    K[3 * j - 2 - 1, 3 * j - 1 - 1] = K[3 * j - 2 - 1, 3 * j - 1 - 1] + k[4 - 1, 5 - 1]
    K[3 * j - 2 - 1, 3 * j - 1] = K[3 * j - 2 - 1, 3 * j - 1] + k[4 - 1, 6 - 1]
    K[3 * j - 1 - 1, 3 * i - 2 - 1] = K[3 * j - 1 - 1, 3 * i - 2 - 1] + k[5 - 1, 1 - 1]
    K[3 * j - 1 - 1, 3 * i - 1 - 1] = K[3 * j - 1 - 1, 3 * i - 1 - 1] + k[5 - 1, 2 - 1]
    K[3 * j - 1 - 1, 3 * i - 1] = K[3 * j - 1 - 1, 3 * i - 1] + k[5 - 1, 3 - 1]
    K[3 * j - 1 - 1, 3 * j - 2 - 1] = K[3 * j - 1 - 1, 3 * j - 2 - 1] + k[5 - 1, 4 - 1]
    K[3 * j - 1 - 1, 3 * j - 1 - 1] = K[3 * j - 1 - 1, 3 * j - 1 - 1] + k[5 - 1, 5 - 1]
    K[3 * j - 1 - 1, 3 * j - 1] = K[3 * j - 1 - 1, 3 * j - 1] + k[5 - 1, 6 - 1]
    K[3 * j - 1, 3 * i - 2 - 1] = K[3 * j - 1, 3 * i - 2 - 1] + k[6 - 1, 1 - 1]
    K[3 * j - 1, 3 * i - 1 - 1] = K[3 * j - 1, 3 * i - 1 - 1] + k[6 - 1, 2 - 1]
    K[3 * j - 1, 3 * i - 1] = K[3 * j - 1, 3 * i - 1] + k[6 - 1, 3 - 1]
    K[3 * j - 1, 3 * j - 2 - 1] = K[3 * j - 1, 3 * j - 2 - 1] + k[6 - 1, 4 - 1]
    K[3 * j - 1, 3 * j - 1 - 1] = K[3 * j - 1, 3 * j - 1 - 1] + k[6 - 1, 5 - 1]
    K[3 * j - 1, 3 * j - 1] = K[3 * j - 1, 3 * j - 1] + k[6 - 1, 6 - 1]


def generate_template(temp_global, i, j, element_num):
    # update 36 col in temp_global
    xlist = [3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1,3*i-1,3*i-1,3*i-1,3*i-1,3*i-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1,3*j-1,3*j-1,3*j-1,3*j-1,3*j-1]
    ylist = [3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1]
    temp_global[:,ASSEMBLE_ENTRIES*element_num:ASSEMBLE_ENTRIES*(element_num+1)] = np.vstack((xlist,ylist))

def computeN_VEC(x0,y0,x1,y1,x2,y2):
    #A = LinearTriangleElementArea(x0,y0,x1,y1,x2,y2)
    #L0 = PlaneFrameElementLength(x0,y0,x1,y1)
    #L1 = PlaneFrameElementLength(x1,y1,x2,y2)
    #L2 = PlaneFrameElementLength(x2,y2,x0,y0)
    v = np.array([x2,y2]) - np.array([x0,y0])
    if v[0] == 0:
        return normalizeVector([0,-1/v[1]])
    if v[1] == 0:
        return normalizeVector([-1/v[0],0])
    return normalizeVector([1,(-1-v[0])/v[1]])

def normalizeVector(v):

    norm = np.linalg.norm(v)
    assert(norm != 0)
    return v/norm





###################################################
##########          global matrix       ###########
###################################################
try:
    K = ss.load_npz(SAVED_K_FILE)
except FileNotFoundError as e:
    print("cant find existing K info")
    ##### generate sparse pattern
    template_nonzero_matrix = np.zeros(shape=(2, ASSEMBLE_ENTRIES * num_elements))
    for element_ID in range(num_elements):
        i = int(np_mesh[element_ID, 0, 2] + 1)
        j = int(np_mesh[element_ID, 1, 2] + 1)
        generate_template(template_nonzero_matrix, i, j, element_ID)

    K = csc_matrix((np.zeros(shape=(ASSEMBLE_ENTRIES * num_elements)),
                    (template_nonzero_matrix[0, :], template_nonzero_matrix[1, :])),
                   shape=(3 * num_nodes, 3 * num_nodes))

    ##### generate global matrix
    for element_ID in range(num_elements):
        x1, y1 = np_mesh[element_ID, 0, 0:2]
        x2, y2 = np_mesh[element_ID, 1, 0:2]
        i = int(np_mesh[element_ID, 0, 2] + 1)
        j = int(np_mesh[element_ID, 1, 2] + 1)
        L = PlaneFrameElementLength(x1, y1, x2, y2)
        C, S = PlaneFrameElementCS(x1, y1, x2, y2, L)
        k = PlaneFrameElementStiffness(E, A, I, L, C, S)
        PlaneFrameAssemble(K, k, i, j)
    ss.save_npz(SAVED_K_FILE, K)

print(K.shape)



###################################################
##########            force             ###########
###################################################

force_file = np.loadtxt(FORCE_FILE, delimiter=',', skiprows=1)
force_dict = {}
force_dict['GLOBALIDX'] = np.array(force_file[:,0],dtype=int)
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



###################################################
##########          fixed U_y           ###########
###################################################
#discard_row = set(np.arange(1,num_nodes*3,3))
#
# saved_row = list(set(np.arange(0,3*num_nodes)) - discard_row)
saved_row = np.arange(0,3*num_nodes,3)
A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]
F = force[saved_row,0]
print(ssl.cgs(A,F))