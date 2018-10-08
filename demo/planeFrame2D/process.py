from demo.planeFrame2D.functionality import *
from scipy.sparse import csc_matrix
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from openmdao.api import Problem
from demo.equalLengthBeamElement.functionality import *
from util.io.airfoilIO import *
from demo.equalLengthBeamElement.BeamModel import *
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
    y = [0]
    for element_ID in range(N):
        for local_node_ID in range(2):
            x = xi[element_ID+local_node_ID]
        count_bot = min(range(turning_point_id), key=lambda i: abs(xBot[i] - x - L[0]/2))
        count_top = min(range(turning_point_id), key=lambda i: abs(xTop[i] - x - L[0]/2))
        divide_list['TOP'].append(count_top)
        divide_list['BOT'].append(count_bot)
        y.append((yTop[count_top]+yBot[count_bot])/2)

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
        force_copy[2 * force_ID] = f_top + f_bot
    for force_ID in range(NNode):
        if force_ID == 0:
            force[2 * force_ID] = force_copy[2 * force_ID]/2.
            pass
        elif force_ID == NNode-1:
            force[2 * force_ID] = force_copy[2*(force_ID-1)]/2.
        else:
            force[2 * force_ID] = (force_copy[2 * (force_ID - 1)] + force_copy[2 * force_ID])/2.
    import matplotlib.pyplot as plt

    plt.plot(xi, y)
    plt.scatter(xTop,yTop)
    plt.scatter(xBot,yBot)
    print(y[-3])
    print(divide_list['TOP'][-3], divide_list['BOT'][-3])
    plt.show()

    return xi, y, force

if __name__ == '__main__':
    NElement = 20
    NNode = NElement + 1
    E = 210E9
    A = 0.01
    I = 0.00005
    K = np.zeros(shape=(3 * NNode, 3 * NNode))


    X, Y, F = computeForce(N=NElement)
    beamp = BeamModel(N=NElement)
    F = beamp.computeForce()
    #import scipy.io as sio
    #sio.savemat('XYF.mat',{'X':X,'Y':Y,'F':F})
    for element in range(NElement):
        x0, y0 = X[element], Y[element]
        x1, y1 = X[element+1], Y[element+1]
        LL = PlaneFrameElementLength(x0,y0,x1,y1)
        C,S = PlaneFrameElementCS(x0,y0,x1,y1,LL)
        k = PlaneFrameElementStiffness(E,A,I,LL,C,S)
        PlaneFrameAssemble(K, k, element, element + 1)
    #### assume that U_y is fixed
    """
    KU=F
    U = [U1_x,U1_y,theta_1,...,UN_x,UN,y,theta_N]
    F = [F1_x,F1_y,M1,...,FN_x,FN_y,MN]
    """
    #import scipy.io as sio
    #sio.savemat('K.mat',{'K':K})
    #np.save('K',K)
    # discard_row = set([0,1,2*num_nodes-2,2*num_nodes-1])
    discard_row = set([0,1,2, 3*(NNode)-3,3*(NNode)-2,3*(NNode)-1]).union(set(range(0,3*NNode,3)))
    saved_row = list(set(np.arange(0, 3 * NNode)) - discard_row)

    A = K[np.array(saved_row)[:, np.newaxis], np.array(saved_row)]
    F = F[2:-2]
    #sio.savemat('f.mat',{'f':F})
    print(F[::2])
    U = np.linalg.solve(A,F)

    d = U[::2]
    print(d)
    from util.plot import *
    plt.figure()
    plt = oneDPlot(d,'scatter',1,xlabel='x',ylabel='displacement')
    finalizePlot(plt, title='The displacement distribution along the plane frame with %d nodes'%NNode,savefig=True,fname='pf_d.eps',bbox_inches='tight')
    plt.figure()
    plt = oneDPlot(F[::2],'scatter',1,xlabel='x',ylabel='shear force in y direction')
    finalizePlot(plt, title='The shear force (in y direction) distribution alone the plane frame with %d nodes'%NNode,savefig=True,fname='pf_f.eps')