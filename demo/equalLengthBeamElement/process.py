from demo.equalLengthBeamElement.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.io as si
import scipy.sparse.linalg as ssl
FORCE_FILE = "/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"
SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_planeFrame2D.mat.npz'

info = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
num_elements = info.__len__()
num_nodes = info.__len__()
info[:,0] = np.arange(num_elements)

turning_point_id = num_elements//2

force_dict = {}
force_dict['GLOBALIDX'] = np.array(info[:,0],dtype=int)
force_dict['X'] = info[:,1]
force_dict['Y'] = info[:,2]
force_dict['PRESS'] = info[:,3]
force_dict['PRESSCO'] = info[:,4]
force_dict['MACHNUM'] = info[:,5]

new_info = sorted(zip(force_dict['X'],force_dict['Y'],force_dict['GLOBALIDX']))


E = 210
I = 0.1
N = 200
NNode = N+1
L = np.ones(shape=(N,))/N
xi = np.cumsum(L) - L[0]
xi = np.hstack((xi,1))
###### assemble np_mesh
divide_list = {}
divide_list['TOP'] = [0]
divide_list['BOT'] = [num_nodes-1]


np_mesh = np.zeros(shape=(N,2,3),dtype=object)

for element_ID in range(N):
    for local_node_ID in range(2):
        x = xi[element_ID+local_node_ID]
        np_mesh[element_ID,local_node_ID,:] = x,0,element_ID + local_node_ID
    count_top = min(range(turning_point_id), key=lambda i: abs(force_dict['X'][i] - x - L[0]/2))
    count_bot = min(range(turning_point_id,num_nodes), key=lambda i: abs(force_dict['X'][i] - x - L[0]/2))
    divide_list['TOP'].append(count_top)
    divide_list['BOT'].append(count_bot)
divide_list['TOP'].append(turning_point_id)
divide_list['BOT'].append(turning_point_id)

try:
    assert(1==0)
    K = ss.load_npz('/Users/gakki/PycharmProjects/MDAOXS/K_beam1D.mat.npz')
except:
    print('cant find K, GENERATING')
    ##### generate sparse pattern
    template_nonzero_matrix = np.zeros(shape=(2, ASSEMBLE_ENTRIES * NNode + 4))
    for element_ID in range(N):
        i = int(element_ID)+1
        j = int(element_ID+1)+1
        generate_template(template_nonzero_matrix,i,j,element_ID)

    # bc
    template_nonzero_matrix[0, ASSEMBLE_ENTRIES*NNode] = 2 * NNode
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES*NNode] = 0

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES*NNode+1] = 2 * NNode + 1
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES*NNode+1] = 1

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES * NNode + 2] = 0
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES * NNode + 2] = 2 * NNode

    template_nonzero_matrix[0, ASSEMBLE_ENTRIES * NNode + 3] = 1
    template_nonzero_matrix[1, ASSEMBLE_ENTRIES * NNode + 3] = 2 * NNode + 1

    K = csc_matrix((np.zeros(shape=(ASSEMBLE_ENTRIES*NNode+4)),(template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(2 * NNode+2, 2 * NNode+2))

    ##### generate global matrix
    for element_ID in range(N):
        i = int(element_ID) + 1
        j = int(element_ID + 1) + 1
        k = BeamElementStiffness(E,I,L[0])
        BeamAssemble(K, k, i, j)
    K[2 * NNode, 0] = 1
    K[2 * NNode + 1 ,1] = 1
    K[0, 2*NNode] = 1
    K[1, 2*NNode+1] = 1
    #ss.save_npz('K_beam1D.mat', K)

### FORCE
# SPEED_OF_SOUND=295.1m/s
force = np.zeros(shape=(2*NNode+2,1))
## compute dimensional force
for force_ID in range(N+1):
    # top force on node
    f_top = 0
    f_bot = 0
    m_top = 0
    m_bot = 0
    for idx in range(divide_list['TOP'][force_ID],divide_list['TOP'][force_ID+1]-1):
        x0, y0 = force_dict['X'][idx],force_dict['Y'][idx]
        x1, y1 = force_dict['X'][idx+1],force_dict['Y'][idx+1]
        LL = BeamElementLength(x0, y0, x1, y1)
        _, fy = LL * abs(force_dict['PRESS'][force_ID] * computeN_VEC(x0,y0,x1,y1))

        w = fy
        # downwards <--> negative position
        f_top += - w
        m_top += - 1/3 * w * ((x1 - xi[force_ID]) ** 2 - (x0-xi[force_ID]) ** 2)
    for idx in range(divide_list['BOT'][force_ID+1]-1,divide_list['BOT'][force_ID]):
        x0, y0 = force_dict['X'][idx],force_dict['Y'][idx]
        x1, y1 = force_dict['X'][idx+1],force_dict['Y'][idx+1]
        LL = BeamElementLength(x0, y0, x1, y1)
        _, fy = LL * abs(force_dict['PRESS'][force_ID] *  computeN_VEC(x0,y0,x1,y1))
        w = - fy
        # downwards <--> negative position
        f_bot += - w
        m_bot += - 1/3 * w * ((x0 - xi[force_ID]) ** 2 - (x1-xi[force_ID]) ** 2)
    force[2*force_ID:2*force_ID + 2,0] = f_top+f_bot, m_top+m_bot


#### assume that U_y is fixed
"""
KU=F
U = [U1_x,U1_y,theta_1,...,UN_x,UN,y,theta_N]
F = [F1_x,F1_y,M1,...,FN_x,FN_y,MN]
"""
#discard_row = set([0,1,2*num_nodes-2,2*num_nodes-1])
discard_row = set()
saved_row = list(set(np.arange(0,2*NNode+2)) - discard_row)
#saved_row = np.arange(1,3*num_nodes,3)

A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]
F = force[saved_row]
#A = K
#F = force
d = np.array(np.linalg.solve(A.todense(),F)).reshape(-1,1)
error = np.matmul(A.todense(),d)-F
l2_error = np.sqrt(error.transpose()*error)
print('error = '+ str(l2_error))
Uy = np.arange(0,2*NNode,2)
dd = d[Uy]


si.savemat('K.mat',mdict={'K':K.todense()})
si.savemat('A.mat',mdict={'A':A.todense()})
si.savemat('F.mat',mdict={'F':F})
si.savemat('d.mat',mdict={'d':d})
si.savemat('dd.mat',mdict={'dd':dd})

# #print(d)
import matplotlib.pyplot as plt
plt.scatter(np.arange(NNode)/NNode,dd,s=0.1,label='displacement')
plt.scatter(np.arange(NNode)/NNode,force[::2][:-1],s=0.1,label='force')
plt.legend()
# plt.scatter(np.array(new_info)[:,0],dd,c='red',s=0.1,label='displacement')
#
# changed_info = np.array(new_info).copy()
# for i in range(NNode):
#     changed_info[i,1] = new_info[i][1] + dd[i]
#
# plt.scatter(changed_info[:,0],changed_info[:,1], s=0.1, c="blue",label='changed')
# plt.hold(True)
# plt.scatter(force_dict['X'],force_dict['Y'], c="orange", s=0.1,label='original')
# plt.xlim()
plt.ylim()
# plt.legend()
plt.savefig(fname='20180403')
