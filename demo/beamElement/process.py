from demo.beamElement.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.io as si

FORCE_FILE = "/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"
SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz'

info = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
num_elements = info.__len__()
num_nodes = info.__len__()

np_mesh = np.zeros(shape=(num_elements,2,3),dtype=object)
__MIN_NODE_ID = np.min(info[:,0])

force_dict = {}
force_dict['GLOBALIDX'] = np.array(info[:,0],dtype=int) - __MIN_NODE_ID
force_dict['X'] = info[:,1]
force_dict['Y'] = info[:,2]
force_dict['PRESS'] = info[:,3]
force_dict['PRESSCO'] = info[:,4]
force_dict['MACHNUM'] = info[:,5]


new_info = sorted(zip(force_dict['X'],force_dict['Y'],force_dict['GLOBALIDX']))


###### assemble np_mesh
for element_ID in range(num_elements):
    for local_node_ID in range(2):
        node_ID = int(new_info[(element_ID+local_node_ID)%num_nodes][2])
        x,y = new_info[(element_ID+local_node_ID)%num_nodes][0:2]
        np_mesh[element_ID,local_node_ID,:] = x,y,node_ID


E = 1.
L = np.zeros(shape=(num_elements,1))
I = 0.1


try:
    assert(1==0)
    K = ss.load_npz('/Users/gakki/PycharmProjects/MDAOXS/K_beam1D.mat.npz')
except:
    print('cant find K, GENERATING')
    ##### generate sparse pattern
    template_nonzero_matrix = np.zeros(shape=(2, ASSEMBLE_ENTRIES * num_elements + 4))
    for element_ID in range(num_elements):
        i = int(element_ID)+1
        j = int((element_ID+1)%num_nodes)+1
        generate_template(template_nonzero_matrix,i,j,element_ID)

    # bc
    template_nonzero_matrix[0, ASSEMBLE_ENTRIES*num_elements] = 2 * num_nodes
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES*num_elements] = 0

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES*num_elements+1] = 2 * num_nodes + 1
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES*num_elements+1] = 1

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES * num_elements + 2] = 0
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES * num_elements + 2] = 2 * num_nodes

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES * num_elements + 3] = 1
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES * num_elements + 3] = 2 * num_nodes + 1

    K = csc_matrix((np.zeros(shape=(ASSEMBLE_ENTRIES*num_elements+4)),(template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(2 * num_nodes+2, 2 * num_nodes+2))

    ##### generate global matrix
    for element_ID in range(num_elements):
        x1, y1 = np_mesh[element_ID,0,0:2]
        x2, y2 = np_mesh[element_ID,1,0:2]
        i = int(element_ID) + 1
        j = int((element_ID + 1) % num_nodes) + 1
        L[element_ID] = abs(x2-x1)
        k = BeamElementStiffness(E,I,L[element_ID])
        BeamAssemble(K, k, i, j)
    K[2 * num_nodes, 0] = 1
    K[2 * num_nodes + 1 ,1] = 1
    K[0, 2*num_nodes] = 1
    K[1, 2*num_nodes+1] = 1
    ss.save_npz('K_beam1D.mat', K)




### FORCE
# SPEED_OF_SOUND=295.1m/s
force_file = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
force = np.zeros(shape=(2*num_nodes+2,1))

## compute dimensional force

for force_ID in range(len(force_file)):
    current_point_id =  int(force_file[force_ID,0] - __MIN_NODE_ID)
    pre_point_id = int(force_file[force_ID-1,0] - __MIN_NODE_ID)
    post_point_id = int(force_file[(force_ID+1)%len(force_file),0] - __MIN_NODE_ID)
    x0,y0 = force_file[force_ID-1,1:3]
    x1,y1 = force_file[force_ID,1:3]
    x2,y2 = force_file[(force_ID+1)%len(force_file),1:3]
    LR = BeamElementLength(x1, y1, x2, y2)
    LL = BeamElementLength(x0, y0, x1, y1)
    #fx,fy = force_file[force_ID,3] * computeN_VEC(x0,y0,x1,y1,x2,y2)
    fx,fy = abs(force_file[force_ID,3] *  computeN_VEC(x0,y0,x1,y1,x2,y2))
    w = -fy

    if force_ID >= 399:
        w = -w

    f = -w * (LL+LR)/2
    m = w * (LL+LR)**2 / 12
    for i in range(num_nodes):
        if new_info[i][2] == current_point_id:
            idx = i
            break

    force[2*idx:2*idx + 2,0] = f, m


#### assume that U_y is fixed
"""
KU=F
U = [U1_x,U1_y,theta_1,...,UN_x,UN,y,theta_N]
F = [F1_x,F1_y,M1,...,FN_x,FN_y,MN]
"""
#discard_row = set([0,1,2*num_nodes-2,2*num_nodes-1])
discard_row = set()
saved_row = list(set(np.arange(0,2*num_nodes+2)) - discard_row)
#saved_row = np.arange(1,3*num_nodes,3)

A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]

F = force[saved_row]
#A = K
#F = force
d = np.array(np.linalg.solve(A.todense(),F)).reshape(-1,1)
error = np.matmul(A.todense(),d)-F
l2_error = np.sqrt(error.transpose()*error)
print('error = '+ str(l2_error))
Uy = np.arange(0,2*num_nodes-4,2)
dd = np.vstack(([0],d[Uy]))
dd = np.vstack((dd,[0]))

#print(d)
import matplotlib.pyplot as plt
plt.scatter(np.array(new_info)[:,0],dd,c='red',s=0.1,label='displacement')

changed_info = np.array(new_info).copy()
for i in range(num_nodes):
    if i !=0 and i!=num_nodes-1:
        changed_info[i,1] = new_info[i][1] + dd[i]

plt.scatter(changed_info[:,0],changed_info[:,1], s=0.1, c="blue",label='changed')
plt.hold(True)
plt.scatter(force_dict['X'],force_dict['Y'], c="orange", s=0.1,label='original')
plt.xlim()
plt.ylim()
plt.legend()
plt.savefig(fname='20180728')

si.savemat('A.mat',mdict={'A':A.todense()})
si.savemat('F.mat',mdict={'F':F})
si.savemat('d.mat',mdict={'d':d})
