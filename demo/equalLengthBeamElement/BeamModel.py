import numpy as np
from demo.equalLengthBeamElement.functionality import *
from util.io.airfoilIO import *

class BeamModel:
    def __init__(self,force_file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv', N=200):
        self.num_element = N
        self.num_node = N+1
        self.airfoil = loadAirfoil(force_file)
        L = np.ones(shape=(N,)) / N
        self.xi = np.cumsum(L) - L[0]
        self.xi = np.hstack((self.xi, 1))
        self._yi = []
        self._divide_list = []

    @property
    def yi(self):
        if not self._yi:
            self._yi = np.zeros(self.num_node)
        return self._yi
    @property
    def divide_list(self):
        if not self._divide_list:
            xTop = self.airfoil['x_top']
            yTop = self.airfoil['y_top']
            pTop = self.airfoil['p_top']
            xBot = self.airfoil['x_bot']
            yBot = self.airfoil['y_bot']
            pBot = self.airfoil['p_bot']
            turning_point_id = self.airfoil['turning_point']
            N = self.num_element
            ###### assemble np_mesh
            divide_list = {}
            divide_list['TOP'] = [0]
            divide_list['BOT'] = [0]

            L = np.ones(shape=(N,)) / N
            xi = np.cumsum(L) - L[0]
            xi = np.hstack((xi, 1))

            for element_ID in range(N):
                for local_node_ID in range(2):
                    x = xi[element_ID + local_node_ID]
                count_bot = min(range(turning_point_id), key=lambda i: abs(xBot[i] - x))
                count_top = min(range(turning_point_id), key=lambda i: abs(xTop[i] - x))
                divide_list['TOP'].append(count_top)
                divide_list['BOT'].append(count_bot)
            self._divide_list = divide_list
        return self._divide_list



    def computeForce(self):
        xTop = self.airfoil['x_top']
        yTop = self.airfoil['y_top']
        pTop = self.airfoil['p_top']
        xBot = self.airfoil['x_bot']
        yBot = self.airfoil['y_bot']
        pBot = self.airfoil['p_bot']
        turning_point_id = self.airfoil['turning_point']
        f_top = []
        f_bot = []
        #LTop = []
        #LBot = []
        for id in range(turning_point_id - 1):
            x0, y0 = xBot[id + 1], yBot[id + 1]
            x1, y1 = xBot[id], yBot[id]
            LL = BeamElementLength(x0, y0, x1, y1)
            #LBot.append(LL)
            _, fy = LL * abs(5.6 * 1.01325E5 * pBot[id] * computeN_VEC(x1, y1, x0, y0))
            w = fy
            # downwards <--> negative position
            f_bot.append(w)

            x0, y0 = xTop[id], yTop[id]
            x1, y1 = xTop[id + 1], yTop[id + 1]
            LL = BeamElementLength(x0, y0, x1, y1)
            #LTop.append(LL)
            _, fy = LL * abs(5.6 * 1.01325E5 * pTop[id] * computeN_VEC(x0, y0, x1, y1))
            w = -fy

            # downwards <--> negative position
            f_top.append(w)

        f_bot.append(0)
        f_top.append(0)
        force = np.array([f_top[i] for i in self.divide_list['TOP']]) + np.array([f_bot[i] for i in self.divide_list['BOT']])
        #force = f_top,f_bot

        F = np.zeros(2*self.num_node)
        F[::2] = force
        return F

    def rebuildShape(self,d):
        d_list_top = self.divide_list['TOP']
        d_list_bot = self.divide_list['BOT']
        y_top = [self.airfoil['y_top'][i]+d[j] for j,i in enumerate(d_list_top)]
        y_bot = [self.airfoil['y_bot'][i]+d[j] for j,i in enumerate(d_list_bot)]
        return self.xi,y_top,y_bot

if __name__ == '__main__':
    from util.plot import *


    N = 20
    beam = BeamModel(N=20)
    force = beam.computeForce()
    plt.figure()
    plt = oneDPlot(force[::2], plotstyle='scatter', span=1, xlabel='x', ylabel='shear force')
    finalizePlot(plt, title='The shear force distribution along the beam with %d nodes' % (N + 1), savefig=True,
                 fname='beam_shear_f.eps')
