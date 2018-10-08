from demo.equalLengthBeamElement.BeamModel import BeamModel
from util.io.airfoilIO import loadAirfoil
import numpy as np
from demo.planeFrame2D.functionality import *
class PlaneFrame(BeamModel):
    def __init__(self,force_file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv', N=200,E=210E9, A=0.0001, I = 0.000005):
        super(PlaneFrame,self).__init__(force_file=force_file,N=N)
        self.num_element = N
        self.num_node = N+1
        self.airfoil = loadAirfoil(force_file)
        L = np.ones(shape=(N,)) / N
        self.xi = np.cumsum(L) - L[0]
        self.xi = np.hstack((self.xi, 1))
        self._yi = []
        self._divide_list = []
        self.E = E
        self.A = A
        self.I = I

    @property
    def yi(self):
        if not self._yi:
            yi = np.array([self.airfoil['y_top'][i] for i in self.divide_list['TOP']]) + \
                  np.array([self.airfoil['y_bot'][i] for i in self.divide_list['BOT']])
            yi = yi/2
            self._yi = yi
        return self._yi

    def assemable_K(self):
        X = self.xi
        Y = self.yi
        K = np.zeros(shape=(3 * self.num_node, 3 * self.num_node))
        for element in range(self.num_element):
            x0, y0 = X[element], Y[element]
            x1, y1 = X[element + 1], Y[element + 1]
            LL = PlaneFrameElementLength(x0, y0, x1, y1)
            C, S = PlaneFrameElementCS(x0, y0, x1, y1, LL)
            k = PlaneFrameElementStiffness(self.E, self.A, self.I, LL, C, S)
            PlaneFrameAssemble(K, k, element, element + 1)
        return K

    def computeDisplacement(self):
        NNode = self.num_node
        discard_row = {0, 1, 2, 3 * (NNode) - 3, 3 * (NNode) - 2, 3 * (NNode) - 1}.union(
            set(range(0, 3 * NNode, 3)))
        saved_row = list(set(np.arange(0, 3 * NNode)) - discard_row)
        K = self.assemable_K()
        A = K[np.array(saved_row)[:, np.newaxis], np.array(saved_row)]
        F = self.computeForce()[2:-2]
        U = np.linalg.solve(A, F)
        d = U[::2]
        return np.hstack((0,d,0))

if __name__ == '__main__':
    num_elements = 40
    num_nodes = num_elements+1
    planeF = PlaneFrame(N=num_elements)
    displacement = planeF.computeDisplacement()
    F = planeF.computeForce()[::2]
    from util.plot import *

    plt = oneDPlot(displacement, 'scatter', 1, xlabel='x', ylabel='displacement')
    finalizePlot(plt, title='The displacement distribution along the plane frame with %d nodes' % (num_nodes), savefig=True,
                 fname='pf_d.eps',bbox_inches='tight')
    plt.figure()
    plt = oneDPlot(F, 'scatter', 1, xlabel='x', ylabel='force')
    finalizePlot(plt, title='The shear force (in y direction) distribution along the plane frame with %d nodes' % (num_nodes), savefig=True,
                 fname='pf_f.eps')
    # REBUILD
    xi, y_top, y_bot = planeF.rebuildShape(displacement)
    plt.figure()
    plt = interpolatePlot(xi, y_top, plotstyple='scatter', N=396, label='rebuilt shape', c='orange', s=0.2)
    plt = interpolatePlot(xi, y_bot, plotstyple='scatter', N=396, c='orange', s=0.2)
    plt = twoDPlot(planeF.airfoil['x_top'], planeF.airfoil['y_top'], plotstyle='scatter', xlabel='x', ylabel='y',
                   label='original shape', c='blue', s=0.2, alpha=0.5)
    plt = twoDPlot(planeF.airfoil['x_bot'], planeF.airfoil['y_bot'], plotstyle='scatter', xlabel='x', ylabel='y', c='blue',
                   s=0.2, alpha=0.5)

    finalizePlot(plt, title='The result of the plane frame model with %d nodes' % (num_nodes), savefig=True,
                 fname='pf_result.eps')


