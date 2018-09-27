from demo.beamElement.functionality import *

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

x_id = np.linspace(0,turning_point_id,NNode).astype(int)
xi = force_dict['X'][x_id]

###### assemble np_mesh
divide_list = {}
divide_list['TOP'] = [0]
divide_list['BOT'] = [num_nodes-1]
y = [0]

np_mesh = np.zeros(shape=(NElem,2,3),dtype=object)

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



import matplotlib.pyplot as plt
plt.scatter(force_dict['X'],force_dict['Y'],s=0.1,label='shape')
plt.plot(xi,y,'x-',label='plane frame',c='red')
plt.plot(xi,np.zeros(shape=(len(xi),1)),'o-',label='beam',c='green')
plt.legend()
plt.title('equal point-distance')
plt.savefig('point-distance.png')
plt.show()
