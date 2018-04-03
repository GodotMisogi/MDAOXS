from demo.planeFrame2D.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
FORCE_FILE = "/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"
SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz'

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


E = 210E06      # modulus of elasticity
I = 5E-05        # moment of inertia
A = 0.1         # cross-sectional area
L = 0           # length
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
force_file = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
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
print(ssl.cgs(A,F))