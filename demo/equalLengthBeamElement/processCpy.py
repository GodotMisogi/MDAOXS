from demo.equalLengthBeamElement.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.io as si
import scipy.sparse.linalg as ssl



def computeForce(force_file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv', N=200):

    info = np.loadtxt(force_file,delimiter=',',skiprows=1)
    num_nodes = info.__len__()
    info[:,0] = np.arange(num_nodes)

    turning_point_id = num_nodes//2

    xBot = info[:,1][:turning_point_id]
    yBot = info[:,2][:turning_point_id]
    xTop = info[:,1][turning_point_id:][::-1]
    yTop = info[:,2][turning_point_id:][::-1]
    force_dict = {}
    force_dict['GLOBALIDX'] = np.array(info[:,0],dtype=int)
    force_dict['X'] = info[:,1]
    force_dict['Y'] = info[:,2]
    force_dict['PRESS'] = info[:,3]
    force_dict['PRESSCO'] = info[:,4]
    force_dict['MACHNUM'] = info[:,5]

    NNode = N+1
    L = np.ones(shape=(N,))/N
    xi = np.cumsum(L) - L[0]
    xi = np.hstack((xi,1))
    ###### assemble np_mesh
    divide_list = {}
    divide_list['TOP'] = [num_nodes-1]
    divide_list['BOT'] = [0]

    for element_ID in range(N):
        for local_node_ID in range(2):
            x = xi[element_ID+local_node_ID]
        count_bot = min(range(turning_point_id), key=lambda i: abs(force_dict['X'][i] - x - L[0]/2))
        count_top = min(range(turning_point_id,num_nodes), key=lambda i: abs(force_dict['X'][i] - x - L[0]/2))
        divide_list['TOP'].append(count_top)
        divide_list['BOT'].append(count_bot)
    divide_list['TOP'].append(turning_point_id)
    divide_list['BOT'].append(turning_point_id)
    ### FORCE
    # SPEED_OF_SOUND=295.1m/s
    force = np.zeros(shape=(2*NNode))
    force_copy = np.zeros(shape=(2*NNode))
    ## compute dimensional force
    for force_ID in range(NNode):
        # top force on node
        f_top = 0
        f_bot = 0
        m_top = 0
        m_bot = 0
        for idx in range(divide_list['TOP'][force_ID+1],divide_list['TOP'][force_ID]):
            x0, y0 = force_dict['X'][idx+1],force_dict['Y'][idx+1]
            x1, y1 = force_dict['X'][idx],force_dict['Y'][idx]
            LL = BeamElementLength(x0, y0, x1, y1)
            _, fy = LL * abs(5.6*1.01325E5*force_dict['PRESS'][idx] * computeN_VEC(x0,y0,x1,y1))

            w = fy

            # downwards <--> negative position
            f_top += - w

            m_top += - 1/3 * w * ((x1 - xi[force_ID]) ** 2 - (x0-xi[force_ID]) ** 2)
        for idx in range(divide_list['BOT'][force_ID],divide_list['BOT'][force_ID+1]):
            x0, y0 = force_dict['X'][idx],force_dict['Y'][idx]
            x1, y1 = force_dict['X'][idx+1],force_dict['Y'][idx+1]
            LL = BeamElementLength(x0, y0, x1, y1)
            print(LL)
            _, fy = LL * abs(5.6*1.01325E5*force_dict['PRESS'][idx] * computeN_VEC(x0,y0,x1,y1))
            w = - fy
            # downwards <--> negative position
            f_bot += - w
            m_bot += - 1/3 * w * ((x0 - xi[force_ID]) ** 2 - (x1-xi[force_ID]) ** 2)
        force_copy[2*force_ID:2*force_ID + 2] = f_top+f_bot, 0
    for force_ID in range(NNode):
        if force_ID == 0 or force_ID == NNode-1:
            force[2*force_ID] = force_copy[2*force_ID]
        else:
            force[2*force_ID] = (force_copy[2*(force_ID-1)]+force_copy[2*force_ID])

    return force

if __name__ == '__main__':
    force_file = '/Users/gakki/Dropbox/thesis/surface_flow_sort.csv'
    info = np.loadtxt(force_file, delimiter=',', skiprows=1)
    num_nodes = info.__len__()
    info[:, 0] = np.arange(num_nodes)

    turning_point_id = num_nodes // 2

    force_dict = {}
    force_dict['GLOBALIDX'] = np.array(info[:, 0], dtype=int)
    force_dict['X'] = info[:, 1]
    force_dict['Y'] = info[:, 2]
    force_dict['PRESS'] = info[:, 3]
    force_dict['PRESSCO'] = info[:, 4]
    force_dict['MACHNUM'] = info[:, 5]


    F = computeForce(N=20)
    force = F[::2]
    import matplotlib.pyplot as plt
    plt.scatter(range(int(len(force_dict['PRESS'])/2)),force_dict['PRESS'][:turning_point_id],label='PBOT')
    plt.scatter(range(int(len(force_dict['PRESS']) / 2)), force_dict['PRESS'][turning_point_id:][::-1], label='PTOP')
    plt.legend()
    plt.show()

    xBot = force_dict['X'][:turning_point_id]
    yBot = force_dict['Y'][:turning_point_id]
    xTop = force_dict['X'][turning_point_id:]
    yTop = force_dict['Y'][turning_point_id:]
    f_top = []
    f_bot = []
    LTop = []
    LBot = []
    for id in range(turning_point_id-1):
        x0, y0 = xBot[id + 1], yBot[id + 1]
        x1, y1 = xBot[id], yBot[id]
        LL = BeamElementLength(x0, y0, x1, y1)
        LBot.append(LL)
        _, fy = LL * abs(5.6 * 1.01325E5 * force_dict['PRESS'][id] * computeN_VEC(x0, y0, x1,y1))
        w = fy

        # downwards <--> negative position
        f_bot.append(w)
        x0, y0 = xTop[id], yTop[id]
        x1, y1 = xTop[id+1], yTop[id+1]
        LL = BeamElementLength(x0, y0, x1, y1)
        LTop.append(LL)
        _, fy = LL * abs(5.6 * 1.01325E5 * force_dict['PRESS'][id+turning_point_id] * computeN_VEC(x0, y0, x1, y1))
        w = -fy

        # downwards <--> negative position
        f_top.append(w)

    plt.scatter(range(len(f_bot)),f_bot,label='FBOT')
    plt.scatter(range(len(f_top)), f_top[::-1], label='FTOP')
    plt.scatter(range(len(f_bot)),np.array(f_bot)+np.array(f_top[::-1]),label='F')
    plt.legend()
    plt.show()
    print(f_top)
    plt.scatter(range(len(LBot)), LBot, label='LBOT')
    plt.scatter(range(len(LTop)), LTop[::-1], label='LTOP')
    plt.legend()
    plt.show()

    F = computeForce(N=20)
    force = F[::2]
    plt.scatter(range(len(force)),force,label='force')
    plt.legend()
    plt.show()

