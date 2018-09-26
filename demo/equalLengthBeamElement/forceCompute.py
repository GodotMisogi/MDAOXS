import numpy as np
from demo.equalLengthBeamElement.functionality import *
def computeForce(force_file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv', N=200):

    info = np.loadtxt(force_file,delimiter=',',skiprows=1)
    num_nodes = info.__len__()
    info[:,0] = np.arange(num_nodes)

    turning_point_id = num_nodes//2

    xBot = info[:,1][:turning_point_id]
    yBot = info[:,2][:turning_point_id]
    pBot = info[:,3][:turning_point_id]
    xTop = info[:,1][turning_point_id:][::-1]
    yTop = info[:,2][turning_point_id:][::-1]
    pTop = info[:,3][turning_point_id:][::-1]

    NNode = N + 1
    L = np.ones(shape=(N,)) / N
    xi = np.cumsum(L) - L[0]
    xi = np.hstack((xi, 1))
    ###### assemble np_mesh
    divide_list = {}
    divide_list['TOP'] = [0]
    divide_list['BOT'] = [0]
    for element_ID in range(N):
        for local_node_ID in range(2):
            x = xi[element_ID+local_node_ID]
        count_bot = min(range(turning_point_id), key=lambda i: abs(xBot[i] - x - L[0]/2))
        count_top = min(range(turning_point_id), key=lambda i: abs(xTop[i] - x - L[0]/2))
        divide_list['TOP'].append(count_top)
        divide_list['BOT'].append(count_bot)
    ### FORCE
    # SPEED_OF_SOUND=295.1m/s
    force = np.zeros(shape=(2 * NNode))
    force_copy = np.zeros(shape=(2 * NNode))
    fTop = []
    fBot = []
    ## compute dimensional force
    for force_ID in range(NNode-1):
        # top force on node
        f_top = 0
        f_bot = 0
        m_top = 0
        m_bot = 0
        for idx in range(divide_list['TOP'][force_ID], divide_list['TOP'][force_ID+1]):
            x0, y0 = xTop[idx], yTop[idx]
            x1, y1 = xTop[idx+1], yTop[idx+1]
            LL = BeamElementLength(x0, y0, x1, y1)
            _, fy = LL * abs(5.6 * 1.01325E5 * pTop[idx] * computeN_VEC(x0, y0, x1, y1))

            w = fy

            # downwards <--> negative position
            f_top += - w
            fTop.append(-w)
            m_top += - 1 / 3 * w * ((x1 - xi[force_ID]) ** 2 - (x0 - xi[force_ID]) ** 2)
        for idx in range(divide_list['BOT'][force_ID], divide_list['BOT'][force_ID + 1]):
            x0, y0 = xBot[idx], yBot[idx]
            x1, y1 = xBot[idx+1], yBot[idx + 1]
            LL = BeamElementLength(x0, y0, x1, y1)
            _, fy = LL * abs(5.6 * 1.01325E5 * pBot[idx] * computeN_VEC(x0, y0, x1, y1))
            w = - fy
            # downwards <--> negative position
            f_bot += - w
            fBot.append(-w)
            m_bot += - 1 / 3 * w * ((x0 - xi[force_ID]) ** 2 - (x1 - xi[force_ID]) ** 2)
        force_copy[2 * force_ID:2 * force_ID + 2] = f_top + f_bot, 0
    for force_ID in range(NNode):
        if force_ID == 0:
            force[2 * force_ID] = force_copy[2 * force_ID]/2.
            pass
        elif force_ID == NNode-1:
            force[2*force_ID] = force_copy[2*(force_ID-1)]/2.
        else:
            force[2 * force_ID] = (force_copy[2 * (force_ID - 1)] + force_copy[2 * force_ID])/2.


    return force

if __name__ == '__main__':

    F = computeForce(N=10)
    force = F[::2]
    print(force)
    import matplotlib.pyplot as plt
    plt.scatter(range(len(force)),force,label='force')
    plt.legend()
    plt.title(s='Shear Force Distribution with 11 Nodes')
    plt.savefig(fname='ShearForce')

