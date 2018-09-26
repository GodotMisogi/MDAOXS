from openmdao.api import ExplicitComponent, ImplicitComponent
import numpy as np
from scipy.linalg import lu_factor, lu_solve

class ArcLengthComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('aoa',types=float)
        self.metadata.declare('x')

    def setup(self):
        num_points = self.metadata['num_panel'] + 1

        self.add_input('y', shape=num_points)
        self.add_output('S', shape=num_points-1)

        rows1 = np.arange(num_points-1)
        cols1 = np.arange(num_points-1)
        rows2 = np.arange(num_points-1)
        cols2 = np.arange(1,num_points)
        self.declare_partials('S', 'y', rows=np.hstack((rows1,rows2)), cols=np.hstack((cols1,cols2)))

    def compute(self, inputs, outputs):
        num_points = self.metadata['num_panel'] + 1
        x = self.metadata['x']
        for i in range(num_points-1):
            outputs['S'][i] = np.sqrt((x[i+1]-x[i])**2+(inputs['y'][i+1]-inputs['y'][i])**2)

    def compute_partials(self, inputs, partials):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        for i in np.arange(num_panel):
            partials['S','y'][i] = - (inputs['y'][i+1]-inputs['y'][i])/np.sqrt((x[i+1]-x[i])**2+(inputs['y'][i+1]-inputs['y'][i])**2)

        for i in np.arange(num_panel):
            partials['S','y'][i+num_panel] = -partials['S','y'][i]


class ThetaComp(ExplicitComponent):
    """
            for i in range(M):
            IP1 = i + 1

            X[i] = 0.5 * (XB[i] + XB[IP1])
            Y[i] = 0.5 * (YB[i] + YB[IP1])

            S[i] = math.sqrt(pow(XB[IP1] - XB[i],2) + pow(YB[IP1] - YB[i],2))

            theta[i] = math.atan2(YB[IP1] - YB[i], XB[IP1] - XB[i])

            RHS[i] = math.sin(theta[i] - alpha)
    """
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_points = self.metadata['num_panel'] + 1

        self.add_input('y', shape=num_points)
        self.add_output('theta', shape=num_panel)

        rows1 = np.arange(num_points - 1)
        cols1 = np.arange(num_points - 1)
        rows2 = np.arange(num_points - 1)
        cols2 = np.arange(1, num_points)
        self.declare_partials('theta', 'y', rows=np.hstack((rows1, rows2)), cols=np.hstack((cols1, cols2)))

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        for i in range(num_panel):
            outputs['theta'][i] = np.arctan2((inputs['y'][i + 1] - inputs['y'][i]) /2, (x[i + 1] - x[i]) / 2)

    def compute_partials(self, inputs, partials):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        for i in np.arange(num_panel):
            tempX = (inputs['y'][i + 1] - inputs['y'][i])/2 / ((x[i + 1] - x[i]) / 2)
            partials['theta', 'y'][i] = - 1 / (1 + tempX**2) / 2 / ((x[i + 1] - x[i]) / 2)

        for i in np.arange(num_panel):
            partials['theta', 'y'][i + num_panel] = -partials['theta', 'y'][i]


class RHSComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('aoa')

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('theta', shape=num_panel)
        self.add_output('RHS', shape=num_panel+1)

        rows1 = np.arange(num_panel)
        cols1 = np.arange(num_panel)
        self.declare_partials('RHS', 'theta', rows=rows1, cols=cols1)

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        alpha = self.metadata['aoa']
        for i in range(num_panel):
            outputs['RHS'][i] = np.sin(inputs['theta'][i]-alpha)
        outputs['RHS'][-1] = 0

    def compute_partials(self, inputs, partials):
        num_panel = self.metadata['num_panel']
        alpha = self.metadata['aoa']
        for i in np.arange(num_panel):
            partials['RHS','theta'][i] = np.cos(inputs['theta'][i]-alpha)

class AComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_point = num_panel+1
        self.add_input('theta', shape=num_panel)
        self.add_input('y',shape=num_point)
        self.add_output('A', shape=(num_panel,num_panel))
        self.declare_partials('A', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        y = inputs['y']
        theta = inputs['theta']
        xm = np.zeros(num_panel)
        ym = np.zeros(num_panel)
        for i in range(num_panel):
            xm[i] = (x[i+1]+x[i])/2
            ym[i] = (y[i+1]+y[i])/2

        for i in range(num_panel):
            for j in range(num_panel):
                outputs['A'][i,j] = - (xm[i]-x[j]) * np.cos(theta[j]) - (ym[i]-y[j])*np.sin(theta[j])
    #
    # def compute_partials(self, inputs, partials):
    #     num_panel = self.metadata['num_panel']
    #     x = self.metadata['x']
    #     for i in np.arange(num_panel):
    #         for j in np.arange(num_panel):
    #             partials['A','theta'][i*num_panel+j,j] = -(x[i] - (x[j+1]+x[j])/2)  *np.sin(inputs['theta'][j]) - (inputs['y'][i]-(inputs['y'][j+1]+inputs['y'][j])/2)*np.cos(inputs['theta'][j])
    #             if i != j:
    #                 if i != j+1:
    #                     partials['A','y'][i*num_panel+j,i] = - np.sin(inputs['theta'][j])
    #                     partials['A', 'y'][i * num_panel + j, j] = np.sin(inputs['theta'][j])/2
    #                 else:
    #                     partials['A', 'y'][i * num_panel + j, i] = - np.sin(inputs['theta'][j]) / 2
    #                     partials['A', 'y'][i * num_panel + j, j] = np.sin(inputs['theta'][j]) / 2
    #             else:
    #                 partials['A', 'y'][i * num_panel + j, i] = - np.sin(inputs['theta'][j]) / 2

class BComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_point = num_panel+1
        self.add_input('y', shape=num_point)
        self.add_output('B', shape=(num_panel,num_panel))
        self.declare_partials('B', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        y = inputs['y']
        xm = np.zeros(num_panel)
        ym = np.zeros(num_panel)
        for i in range(num_panel):
            xm[i] = (x[i + 1] + x[i]) / 2
            ym[i] = (y[i + 1] + y[i]) / 2

        for i in range(num_panel):
            for j in range(num_panel):
                outputs['B'][i,j] = (xm[i]-x[j]) ** 2 + (ym[i]-y[j]) ** 2
    #
    # def compute_partials(self, inputs, partials):
    #     num_panel = self.metadata['num_panel']
    #     x = self.metadata['x']
    #     for i in np.arange(num_panel):
    #         for j in np.arange(num_panel):
    #             if i != j:
    #                 if i != j+1:
    #                     partials['B','y'][i*num_panel+j,i] = 2 * (inputs['y'][i]-(inputs['y'][j+1]+inputs['y'][j])/2)
    #                     partials['B', 'y'][i * num_panel + j, j] = - (inputs['y'][i]-(inputs['y'][j+1]+inputs['y'][j])/2)
    #                 else:
    #                     partials['B', 'y'][i * num_panel + j, i] = (inputs['y'][i]-inputs['y'][j])/2
    #                     partials['B', 'y'][i * num_panel + j, j] = - (inputs['y'][i]-(inputs['y'][j+1]+inputs['y'][j])/2)
    #             else:
    #                 partials['B', 'y'][i * num_panel + j, i] = (inputs['y'][i]-inputs['y'][j+1])/2

class CComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('theta', shape=num_panel)
        self.add_output('C', shape=(num_panel,num_panel))
        self.declare_partials('C', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['C'][i,j] = np.sin(inputs['theta'][i]-inputs['theta'][j])
    #
    # def compute_partials(self, inputs, partials):
    #     num_panel = self.metadata['num_panel']
    #     for i in np.arange(num_panel):
    #         for j in np.arange(num_panel):
    #             if i != j:
    #                 partials['C', 'theta'][i * num_panel + j, i] = np.cos(inputs['theta'][i]-inputs['theta'][j])
    #                 partials['C', 'theta'][i * num_panel + j, j] = -np.cos(inputs['theta'][i]-inputs['theta'][j])
    #             else:
    #                 partials['C', 'theta'][i * num_panel + j, i] = 0

class DComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('theta', shape=num_panel)
        self.add_output('D', shape=(num_panel,num_panel))
        self.declare_partials('D', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['D'][i,j] = np.cos(inputs['theta'][i]-inputs['theta'][j])
    #
    # def compute_partials(self, inputs, partials):
    #     num_panel = self.metadata['num_panel']
    #     for i in np.arange(num_panel):
    #         for j in np.arange(num_panel):
    #             if i != j:
    #                 partials['D', 'theta'][i * num_panel + j, i] = -np.sin(inputs['theta'][i]-inputs['theta'][j])
    #                 partials['D', 'theta'][i * num_panel + j, j] = np.sin(inputs['theta'][i]-inputs['theta'][j])
    #             else:
    #                 partials['D', 'theta'][i * num_panel + j, i] = 0

class EComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_point = num_panel+1
        self.add_input('theta', shape=num_panel)
        self.add_input('y',shape=num_point)
        self.add_output('E', shape=(num_panel,num_panel))
        self.declare_partials('E', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        x = self.metadata['x']
        y = inputs['y']
        xm = np.zeros(num_panel)
        ym = np.zeros(num_panel)
        for i in range(num_panel):
            xm[i] = (x[i + 1] + x[i]) / 2
            ym[i] = (y[i + 1] + y[i]) / 2
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['E'][i,j] = (xm[i]-x[j]) * np.sin(inputs['theta'][j]) - (ym[i]-y[j])*np.cos(inputs['theta'][j])
    #
    # def compute_partials(self, inputs, partials):
    #     num_panel = self.metadata['num_panel']
    #     x = self.metadata['x']
    #     for i in np.arange(num_panel):
    #         for j in np.arange(num_panel):
    #             partials['E','theta'][i*num_panel+j,j] = (x[i] - (x[j+1]+x[j])/2) * np.cos(inputs['theta'][j]) + (inputs['y'][i]-(inputs['y'][j+1]+inputs['y'][j])/2)*np.sin(inputs['theta'][j])
    #             if i != j:
    #                 if i != j+1:
    #                     partials['E','y'][i*num_panel+j,i] = - np.cos(inputs['theta'][j])
    #                     partials['E', 'y'][i * num_panel + j, j] = np.cos(inputs['theta'][j])/2
    #                 else:
    #                     partials['E', 'y'][i * num_panel + j, i] = - np.cos(inputs['theta'][j]) / 2
    #                     partials['E', 'y'][i * num_panel + j, j] = np.cos(inputs['theta'][j]) / 2
    #             else:
    #                 partials['E', 'y'][i * num_panel + j, i] = - np.cos(inputs['theta'][j]) / 2


class FComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('A', shape=(num_panel,num_panel))
        self.add_input('B',shape=(num_panel,num_panel))
        self.add_input('S',shape=num_panel)
        self.add_output('F', shape=(num_panel,num_panel))
        self.declare_partials('F', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        S = inputs['S']
        A = inputs['A']
        B = inputs['B']
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['F'][i,j] = np.log(1+(S[j]**2 + 2 * A[i,j]*S[j])/B[i,j])

class GComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('A', shape=(num_panel,num_panel))
        self.add_input('B',shape=(num_panel,num_panel))
        self.add_input('E', shape=(num_panel, num_panel))
        self.add_input('S',shape=num_panel)
        self.add_output('G', shape=(num_panel,num_panel))
        self.declare_partials('G', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        E = inputs['E']
        A = inputs['A']
        B = inputs['B']
        S = inputs['S']
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['G'][i,j] = np.arctan2(E[i,j]*S[j],(B[i,j]+A[i,j]*S[j]))


class PComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_point = num_panel + 1
        self.add_input('y', shape=num_point)
        self.add_input('theta',shape=num_panel)
        self.add_output('P', shape=(num_panel,num_panel))
        self.declare_partials('P', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        y = inputs['y']
        x = self.metadata['x']
        theta = inputs['theta']
        xm = np.zeros(num_panel)
        ym = np.zeros(num_panel)
        for i in range(num_panel):
            xm[i] = (x[i + 1] + x[i]) / 2
            ym[i] = (y[i + 1] + y[i]) / 2
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['P'][i,j] = (xm[i]-x[j])*np.sin(theta[i]-2*theta[j]) + (ym[i]-y[j])*np.cos(theta[i]-2*theta[j])



class QComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('x')

    def setup(self):
        num_panel = self.metadata['num_panel']
        num_point = num_panel + 1
        self.add_input('y', shape=num_point)
        self.add_input('theta',shape=num_panel)
        self.add_output('Q', shape=(num_panel,num_panel))
        self.declare_partials('Q', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        y = inputs['y']
        x = self.metadata['x']
        theta = inputs['theta']
        xm = np.zeros(num_panel)
        ym = np.zeros(num_panel)
        for i in range(num_panel):
            xm[i] = (x[i + 1] + x[i]) / 2
            ym[i] = (y[i + 1] + y[i]) / 2
        for i in range(num_panel):
            for j in range(num_panel):
                outputs['Q'][i,j] = (xm[i]-x[j])*np.cos(theta[i]-2*theta[j]) - (ym[i]-y[j])*np.sin(theta[i]-2*theta[j])

class CN2Comp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('A', shape=(num_panel,num_panel))
        self.add_input('C', shape=(num_panel,num_panel))
        self.add_input('D', shape=(num_panel,num_panel))
        self.add_input('E', shape=(num_panel,num_panel))
        self.add_input('F', shape=(num_panel,num_panel))
        self.add_input('G', shape=(num_panel,num_panel))
        self.add_input('Q', shape=(num_panel,num_panel))
        self.add_input('S', shape=num_panel)
        self.add_output('CN2', shape=(num_panel,num_panel))
        self.declare_partials('CN2', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        A = inputs['A']
        C = inputs['C']
        D = inputs['D']
        E = inputs['E']
        F = inputs['F']
        G = inputs['G']
        Q = inputs['Q']
        S = inputs['S']

        for i in range(num_panel):
            for j in range(num_panel):
                if i == j:
                    outputs['CN2'][i,j] = 1
                else:
                    outputs['CN2'][i,j] = D[i,j]+0.5*Q[i,j]*F[i,j]/S[j]-(A[i,j]*C[i,j] + D[i,j]*E[i,j]) * G[i,j]/S[j]

class CN1Comp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('C', shape=(num_panel,num_panel))
        self.add_input('D', shape=(num_panel,num_panel))
        self.add_input('F', shape=(num_panel,num_panel))
        self.add_input('G', shape=(num_panel,num_panel))
        self.add_input('CN2', shape=(num_panel,num_panel))
        self.add_output('CN1', shape=(num_panel,num_panel))
        self.declare_partials('CN1', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        C = inputs['C']
        D = inputs['D']
        F = inputs['F']
        G = inputs['G']
        CN2 = inputs['CN2']

        for i in range(num_panel):
            for j in range(num_panel):
                if i == j:
                    outputs['CN1'][i,j] = -1
                else:
                    outputs['CN1'][i,j] = 0.5*D[i,j]*F[i,j] + C[i,j]*G[i,j]-CN2[i,j]

class ANComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('CN1', shape=(num_panel,num_panel))
        self.add_input('CN2', shape=(num_panel,num_panel))
        self.add_output('AN', shape=(num_panel+1,num_panel+1))
        self.declare_partials('AN', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        CN1 = inputs['CN1']
        CN2 = inputs['CN2']
        for i in np.arange(num_panel):
            outputs['AN'][i,0] = CN1[i,0]
            for j in np.arange(1,num_panel):
                outputs['AN'][i,j] = CN1[i,j] + CN2[i,j-1]
            outputs['AN'][i,num_panel] = CN2[i,num_panel-1]
        outputs['AN'][num_panel,0] = 1
        outputs['AN'][num_panel,num_panel] = 1
        for j in np.arange(1,num_panel):
            outputs['AN'][num_panel,j] = 0

class GammaComp(ImplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        size = num_panel+1
        self.add_input('AN', shape=(num_panel+1,num_panel+1))
        self.add_input('RHS', shape=num_panel+1)
        self.add_output('gamma', shape=num_panel+1)
        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('gamma', 'AN', rows=rows, cols=cols)

        self.declare_partials('gamma', 'gamma')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['gamma'] = np.dot(inputs['AN'], outputs['gamma']) - inputs['RHS']

    def solve_nonlinear(self, inputs, outputs):
        self.lu = lu_factor(inputs['AN'])

        outputs['gamma'] = lu_solve(self.lu, inputs['RHS'])

    def linearize(self, inputs, outputs, partials):
        num_panel = self.metadata['num_panel']

        self.lu = lu_factor(inputs['AN'])

        partials['gamma', 'AN'] = np.outer(np.ones(num_panel+1), outputs['gamma']).flatten()
        partials['gamma', 'gamma'] = inputs['AN']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['gamma'] = lu_solve(self.lu, d_residuals['gamma'], trans=0)
        else:
            d_residuals['gamma'] = lu_solve(self.lu, d_outputs['gamma'], trans=1)

class CT2Comp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('A', shape=(num_panel,num_panel))
        self.add_input('C', shape=(num_panel,num_panel))
        self.add_input('D', shape=(num_panel,num_panel))
        self.add_input('E', shape=(num_panel,num_panel))
        self.add_input('F', shape=(num_panel,num_panel))
        self.add_input('G', shape=(num_panel,num_panel))
        self.add_input('P', shape=(num_panel,num_panel))
        self.add_input('S', shape=num_panel)
        self.add_output('CT2', shape=(num_panel,num_panel))
        self.declare_partials('CT2', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        A = inputs['A']
        C = inputs['C']
        D = inputs['D']
        E = inputs['E']
        F = inputs['F']
        G = inputs['G']
        P = inputs['P']
        S = inputs['S']

        for i in range(num_panel):
            for j in range(num_panel):
                if i == j:
                    outputs['CT2'][i,j] = 0.5 * np.pi
                else:
                    outputs['CT2'][i,j] = C[i,j] + 0.5*P[i,j]*F[i,j]/S[j] + (A[i,j]*D[i,j] - C[i,j]*E[i,j]) * G[i,j]/S[j]

class CT1Comp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('C', shape=(num_panel,num_panel))
        self.add_input('D', shape=(num_panel,num_panel))
        self.add_input('F', shape=(num_panel,num_panel))
        self.add_input('G', shape=(num_panel,num_panel))
        self.add_input('CT2', shape=(num_panel,num_panel))
        self.add_output('CT1', shape=(num_panel,num_panel))
        self.declare_partials('CT1', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        C = inputs['C']
        D = inputs['D']
        F = inputs['F']
        G = inputs['G']
        CT2 = inputs['CT2']

        for i in range(num_panel):
            for j in range(num_panel):
                if i == j:
                    outputs['CT1'][i,j] = 0.5 * np.pi
                else:
                    outputs['CT1'][i,j] = 0.5*C[i,j]*F[i,j] - D[i,j]*G[i,j]- CT2[i,j]

class ATComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']
        self.add_input('CT1', shape=(num_panel,num_panel))
        self.add_input('CT2', shape=(num_panel,num_panel))
        self.add_output('AT', shape=(num_panel+1,num_panel+1))
        self.declare_partials('AT', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        CT1 = inputs['CT1']
        CT2 = inputs['CT2']
        for i in np.arange(num_panel):
            outputs['AT'][i,0] = CT1[i,0]
            for j in np.arange(1,num_panel):
                outputs['AT'][i,j] = CT1[i,j] + CT2[i,j-1]
            outputs['AT'][i,num_panel] = CT2[i,num_panel-1]

class VelocityComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)
        self.metadata.declare('aoa',types=float)
    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('gamma', shape=num_panel+1)
        self.add_input('AT', shape=(num_panel+1, num_panel+1))
        self.add_input('theta', shape=num_panel)
        self.add_output('V', shape=num_panel)
        self.declare_partials('V', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        alpha = self.metadata['aoa']
        gamma = inputs['gamma']
        AT = inputs['AT']
        theta = inputs['theta']
        for i in np.arange(num_panel):
            outputs['V'][i] = np.cos(theta[i]-alpha)
            for j in np.arange(num_panel+1):
                outputs['V'][i] += AT[i,j]*gamma[j]

class LiftComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_panel', types=int)

    def setup(self):
        num_panel = self.metadata['num_panel']

        self.add_input('V', shape=num_panel)
        self.add_input('S', shape=num_panel)
        self.add_output('CL', shape=1)
        self.declare_partials('CL', '*',method='fd')

    def compute(self, inputs, outputs):
        num_panel = self.metadata['num_panel']
        outputs['CL'] = 0
        V = inputs['V']
        S = inputs['S']
        for i in np.arange(num_panel):
            outputs['CL'] += 2*V[i]*S[i]
        outputs['CL'] = (outputs['CL']-0.18)**2
