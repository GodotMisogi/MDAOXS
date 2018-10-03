from demo.planeFrame2D.functionality import *
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

turning_point_id = 398

force_dict = {}
force_dict['GLOBALIDX'] = np.array(info[:,0],dtype=int)
force_dict['X'] = info[:,1]
force_dict['Y'] = info[:,2]
force_dict['PRESS'] = info[:,3]
force_dict['PRESSCO'] = info[:,4]
force_dict['MACHNUM'] = info[:,5]

new_info = sorted(zip(force_dict['X'],force_dict['Y'],force_dict['GLOBALIDX']))

NElem = 20
NNode = NElem + 1
turning_point_id = 398
divide_by_id_interval = turning_point_id/NElem

xi = np.cumsum(np.ones(NElem)/NElem) - 1/NElem
xi = np.hstack((xi,1))

###### assemble np_mesh
divide_list = {}
divide_list['TOP'] = [0]
divide_list['BOT'] = [num_nodes-1]
y = [0]

np_mesh = np.zeros(shape=(NElem,2,3),dtype=object)
__MIN_NODE_ID = np.min(info[:,0])

for node_id in range(1,NNode-1):
    x = xi[node_id]
    count_top = min(range(turning_point_id), key=lambda i: abs(force_dict['X'][i] - x))
    count_bot = min(range(turning_point_id,num_nodes), key=lambda i: abs(force_dict['X'][i] - x))
    y.append((force_dict['Y'][count_top]+force_dict['Y'][count_bot])/2)
    divide_list['TOP'].append(count_top)
    divide_list['BOT'].append(count_bot)
divide_list['TOP'].append(turning_point_id)
divide_list['BOT'].append(turning_point_id)
y.append(force_dict['Y'][turning_point_id])

#computing K
E = 210000000
A = 0.02
I = 0.00005
K = np.zeros(shape=(3*NNode,3*NNode))

for node_id in range(1,NElem):
    x1 = xi[node_id]
    x2 = xi[node_id+1]
    y1 = y[node_id]
    y2 = y[node_id+1]
    L = PlaneFrameElementLength(x1,x2,y1,y2)
    C,S = PlaneFrameElementCS(x1,x2,y1,y2,L)
    k = PlaneFrameElementStiffness(E,A,I,L,C,S)
    PlaneFrameAssemble(K,k,node_id,node_id+1)



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
    LR = PlaneFrameElementLength(x1, y1, x2, y2)
    LL = PlaneFrameElementLength(x0, y0, x1, y1)
    # fx,fy = force_file[force_ID,3] * computeN_VEC(x0,y0,x1,y1,x2,y2)
    fx, fy = abs(force_file[force_ID, 3] * computeN_VEC(x0, y0, x1, y1, x2, y2))
    f = np.sqrt(fx**2+fy**2)
    w = f

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

#A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]

#F = force[saved_row]


from util.plot import *
plt.figure()
plt = twoDPlot(force_dict['X'],force_dict['Y'],plotstyle='scatter',label='the original shape',s=0.1)
plt = twoDPlot(xi,y,label='the plane frame model',marker='x',c='red')
plt = twoDPlot(xi,np.zeros(shape=(len(xi),1)),label='the beam model',marker='o',c='green')
plt.legend(loc=1)
plt.xlabel('x')
plt.ylabel('y')
finalizePlot(plt,title='the modelling for NACA6a203 airfoil',savefig=True,fname='model_xdistance.eps')