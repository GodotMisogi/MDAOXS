from tools.SU2IO import read
import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse as ss
from demo.twoDFEM.functionality import *
import scipy.sparse.linalg as ssl
import scipy.sparse.linalg as ssl
MESH_FILE = "/Users/gakki/Dropbox/thesis/data/mesh_NACA0012_inv.su2"
FORCE_FILE = "/Users/gakki/Dropbox/thesis/data/surface_flow.csv"
SAVED_K_FILE = '/Users/gakki/PycharmProjects/MDAOXS/K_STD_2d.mat.npz'
GAMMA_VALUE= 1.4
AOA= 1.25
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 273.15
GAS_CONSTANT= 287.87
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_DIMENSIONALIZATION= 'DIMENSIONAL'
# Marker of the Euler boundary (NONE = no marker)
MARKER_EULER= 'airfoil'
# Marker of the far field (NONE = no marker)
MARKER_FAR=  'farfield'


mesh = read(MESH_FILE)
num_elements = mesh['NELEM']
num_nodes = mesh['NPOIN']

E = 210E06
NU = 0.3
t = 0.025
p = 1

try:
    K = ss.load_npz(SAVED_K_FILE)
except FileNotFoundError as e:
    print("cant find k")
    # assemble template first
    template_nonzero_matrix = np.zeros(shape=(2, 36 * num_elements))
    for element_id in range(num_elements):
        i = mesh['ELEM'][element_id][1] + 1
        j = mesh['ELEM'][element_id][2] + 1
        m = mesh['ELEM'][element_id][3] + 1
        generate_template(template_nonzero_matrix, i, j, m, element_id)

    K=csc_matrix((np.zeros(shape=(36*num_elements)), (template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(2 * num_nodes, 2 * num_nodes))
    for element_id in range(num_elements):
        assert(5 == mesh['ELEM'][element_id][0])
        i = mesh['ELEM'][element_id][1]
        j = mesh['ELEM'][element_id][2]
        m = mesh['ELEM'][element_id][3]
        xi,yi = mesh['POIN'][i][0:2]
        xj,yj = mesh['POIN'][j][0:2]
        xm,ym = mesh['POIN'][m][0:2]
        k = LinearTriangleElementStiffness(E,NU,t,xi,yi,xj,yj,xm,ym,p)
        LinearTriangleAssemble(K, k, i+1, j+1, m+1)
    ss.save_npz(SAVED_K_FILE,K)

print(K.shape)


### FORCE

force_file = np.loadtxt(FORCE_FILE, delimiter=',', skiprows=1)
force_dict = {}
force_dict['GLOBALIDX'] = np.array(force_file[:,0],dtype=int)
force_dict['X'] = force_file[:,1]
force_dict['Y'] = force_file[:,2]
force_dict['PRESS'] = force_file[:,3]
force_dict['PRESSCO'] = force_file[:,4]
force_dict['MACHNUM'] = force_file[:,5]


force = np.zeros(shape=(2*num_nodes,1))
## compute dimensional force
for force_ID in range(len(force_file)):
    current_point_id =  int(force_file[force_ID,0])
    pre_point_id = int(force_file[force_ID-1,0])
    post_point_id = int(force_file[(force_ID+1)%len(force_file),0])
    x0,y0 = mesh['POIN'][pre_point_id][0:2]
    x1,y1 = mesh['POIN'][current_point_id][0:2]
    x2,y2 = mesh['POIN'][post_point_id][0:2]
    #fx,fy = force_file[force_ID,3] * computeN_VEC(x0,y0,x1,y1,x2,y2)
    fx,fy = computeN_VEC(x0,y0,x1,y1,x2,y2)
    force[2*current_point_id:2*current_point_id+2,0] = fx,fy


discard_row = set(np.arange(1,num_nodes*2,2))
saved_row = list(set(np.arange(0,2*num_nodes)) - discard_row)
A = K[np.array(saved_row)[:,np.newaxis], np.array(saved_row)]
F = force[saved_row,0]


print(ssl.spsolve(A,F))
"""
### Visualize
import matplotlib.pyplot as plt
x = []
y = []
for i in range(force_file[:,0].__len__()):
    x.append(mesh['POIN'][force_dict['GLOBALIDX'][i]][0])
    y.append(mesh['POIN'][force_dict['GLOBALIDX'][i]][1])


change_x = x.copy()
change_y = y.copy()
for i in range(force_file[:,0].__len__()):
    change_x[i] += force_dict['X'][i]
    change_y[i] += force_dict['Y'][i]


#plt.scatter(x,y,s=0.1,label='force points')
#plt.hold(True)
plt.scatter(force_dict['X'],force_dict['Y'],s=30, label="force point")

#plt.scatter(change_x, change_y, s=0.1,label='movement?')
plt.scatter(np.array(mesh['POIN'])[:,0],np.array(mesh['POIN'])[:,1],label='CFD mesh', s=3)
#plt.hold(True)
plt.legend()
#plt.xlim(xmax=1,xmin=0)
#plt.ylim(ymax=0.04,ymin=-0.03)
plt.xlim(xmax=force_dict['X'][100]+0.01,xmin=force_dict['X'][100]-0.01)
plt.ylim(ymax=force_dict['Y'][100]+0.01,ymin=force_dict['Y'][100]-0.01)
_,firstID, secondID, thirdID,_ = mesh['ELEM'][force_dict['GLOBALIDX'][100]]
element_X = []
element_Y = []
element_X.append(mesh['POIN'][firstID][0])
element_Y.append(mesh['POIN'][firstID][0])
element_X.append(mesh['POIN'][secondID][0])
element_Y.append(mesh['POIN'][secondID][0])
element_X.append(mesh['POIN'][thirdID][0])
element_Y.append(mesh['POIN'][thirdID][0])
#plt.show(dpi=300)
#plt.plot(element_X,element_Y)
plt.savefig('filename.png', dpi = 300)

###############################################
############## demo in page 223  ##############
###############################################
num_elements = 2
num_nodes = 4
mesh2 = {}
mesh2['MARKS']={}
mesh2['MARKS']['Section 1']={}
mesh2['MARKS']['Section 1']['ELEM'] = [[5, 0, 2, 3],[5, 0, 1, 2]]
mesh2['POIN'] = [[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.25,0.0],[0.0,0.25,0.0]]

E = 210E06
NU = 0.3
t = 0.025

template_nonzero_matrix = np.zeros(shape=(2,36*num_elements))
for element_id in range(num_elements):
    i = mesh2['MARKS']['Section 1']['ELEM'][element_id][1] + 1
    j = mesh2['MARKS']['Section 1']['ELEM'][element_id][2] + 1
    m = mesh2['MARKS']['Section 1']['ELEM'][element_id][3] + 1
    generate_template(template_nonzero_matrix,i,j,m,element_id)



K = csc_matrix((np.zeros(shape=(36*num_elements)), (template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(2 * num_nodes, 2 * num_nodes))

for element_id in range(num_elements):
    assert(5 == mesh['MARKS']['Section 1']['ELEM'][element_id][0])
    i = mesh2['MARKS']['Section 1']['ELEM'][element_id][1]
    j = mesh2['MARKS']['Section 1']['ELEM'][element_id][2]
    m = mesh2['MARKS']['Section 1']['ELEM'][element_id][3]
    xi,yi = mesh2['POIN'][i][0:2]
    xj,yj = mesh2['POIN'][j][0:2]
    xm,ym = mesh2['POIN'][m][0:2]
    k = LinearTriangleElementStiffness(E,NU,t,xi,yi,xj,yj,xm,ym,p)
    LinearTriangleAssemble(K, k, i+1, j+1, m+1)

"""
