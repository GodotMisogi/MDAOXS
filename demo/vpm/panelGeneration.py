import numpy as np
import matplotlib.pyplot as plt
import math
from demo.vpm.vpmfunc import *
class Airfoil:

    def __init__(self,file_name,chord_length,num_samples,angle_of_attack,figID=None,plotColor=None):
        """
        The Airfoil class constructor.
        """
        assert (num_samples % 2 == 0)
        self.file_name = file_name
        self.chord = chord_length
        self.NUM_SAMPLES = num_samples
        self.angle_of_attack = angle_of_attack
        self.figure_to_plot = figID
        self.plotColor = plotColor
        self.aoa = math.radians(angle_of_attack)
        self.HASPLOT = {}
        if (figID != None):
            self.HASPLOT[str(self.Figure_To_Plot.number)] = False
        self.boundaryPoints_X = None
        self.boundaryPoints_Y = None
        self.full_coefficientLift = None
        self.pressure_coefficient = None

    def panelGeneration(self):
        """
        :param grid:
        :param N: N must be an even number
        :return: X,Y as the panel corner point
        """
        info = np.loadtxt(self.file_name, delimiter=',', skiprows=1)
        grid = {}
        grid['X'] = info[:, 1]
        grid['Y'] = info[:, 2]
        number_of_grid = len(grid['X'])
        X = []
        Y = []
        for i in np.linspace(number_of_grid//2-1,0,self.NUM_SAMPLES/2+1).astype(int):
            X.append(grid['X'][i])
            Y.append(grid['Y'][i])
        for i in np.linspace(number_of_grid-1,number_of_grid//2+1,self.NUM_SAMPLES/2+1).astype(int):
            if i == number_of_grid-1:
                continue
            X.append(grid['X'][i])
            Y.append(grid['Y'][i])
        self.boundaryPoints_X = X
        self.boundaryPoints_Y = Y

    def Get_PanelCoefficients(self,PLOT=False):
        XB = self.boundaryPoints_X
        YB = self.boundaryPoints_Y
        M = self.NUM_SAMPLES
        alpha = self.angle_of_attack
        FigID = self.figure_to_plot
        plotColor = self.plotColor

        alphad = alpha
        # convert alpha to radiance
        alpha *= 2*np.pi/360

        # The trailing edge requires an extra point
        MP1 = M+1

        """ TODO: Find better variable names for these arrays """
        X = np.zeros(M)
        Y = np.zeros(M)
        RHS = np.zeros(MP1)
        theta = np.zeros(M)
        S = np.zeros(M)

        for i in range(M):
            IP1 = i + 1

            X[i] = 0.5 * (XB[i] + XB[IP1])
            Y[i] = 0.5 * (YB[i] + YB[IP1])

            S[i] = math.sqrt(pow(XB[IP1] - XB[i],2) + pow(YB[IP1] - YB[i],2))

            theta[i] = math.atan2(YB[IP1] - YB[i], XB[IP1] - XB[i])

            RHS[i] = math.sin(theta[i] - alpha)
        CN1 = np.zeros((M,M))
        CN2 = np.zeros((M,M))
        CT1 = np.zeros((M,M))
        CT2 = np.zeros((M,M))
        An = np.zeros((MP1,MP1))
        At = np.zeros((MP1,MP1))
        self.A = []

        for i in range(M):
            for j in range(M):

                if (i == j):
                    CN1[i,j] = -1
                    CN2[i,j] = 1
                    CT1[i,j] = 0.5 * math.pi
                    CT2[i,j] = 0.5 * math.pi
                    self.A.append(0)
                else:
                    A = -1*(X[i]-XB[j])*math.cos(theta[j])-(Y[i]-YB[j])*math.sin(theta[j])
                    self.A.append(A)
                    B = (X[i]-XB[j])**2+(Y[i]-YB[j])**2
                    C = math.sin(theta[i]-theta[j])
                    D = math.cos(theta[i]-theta[j])
                    E = (X[i]-XB[j])*math.sin(theta[j])-(Y[i]-YB[j])*math.cos(theta[j])
                    F = math.log(1+S[j]*(S[j]+2*A)/B)
                    G = math.atan2((E*S[j]),(B+A*S[j]))
                    P = (X[i]-XB[j])*math.sin(theta[i]-2*theta[j])+(Y[i]-YB[j])*math.cos(theta[i]-2*theta[j])
                    Q = (X[i]-XB[j])*math.cos(theta[i]-2*theta[j])-(Y[i]-YB[j])*math.sin(theta[i]-2*theta[j])

                    CN2[i,j] = D + 0.5 * Q * F/S[j] - (A*C + D*E) * G/S[j]
                    CN1[i,j] = 0.5*D*F + C*G - CN2[i,j]
                    CT2[i,j] = C + 0.5*P*F/S[j] + (A*D - C*E) * G/S[j]
                    CT1[i,j] = 0.5*C*F - D*G - CT2[i,j]

        for i in range(M):
            An[i,0] = CN1[i,0]
            An[i,MP1-1] = CN2[i,M-1]
            At[i,0] = CT1[i,0]
            At[i,MP1-1] = CT2[i,M-1]

            for j in range(1,M):
                An[i,j] = CN1[i,j] + CN2[i,(j-1)]
                At[i,j] = CT1[i,j] + CT2[i,(j-1)]

        # Trailing edge conditions
        An[MP1-1,0] = 1
        An[MP1-1,MP1-1] = 1

        for j in range(1,M):
            An[MP1-1,j] = 0

        RHS[MP1-1] = 0

        gamma = np.linalg.solve(An,RHS)
        V = np.zeros(M)
        Cp = np.zeros(M)


        for i in range(M):
            V[i] = math.cos(theta[i] - alpha)

            for j in range(MP1):
                V[i] = V[i] + At[i,j]*gamma[j]
                Cp[i] = 1 - (V[i])**2

        cl = Get_LiftCoefficients(V,S,M)

        CpLower = Cp[0:int(M/2)]
        CpUpper = Cp[int(M/2):]
        Cp = Cp.astype(np.float)
        newcl = np.array(cl)
        newcl = newcl.astype(np.float)

        self.full_coefficientLift = newcl
        self.pressure_coefficient = Cp
        self.S = S
        self.CN1 = CN1
        self.CN2 = CN2
        self.An = An

if __name__ == '__main__':
    FORCE_FILE = "/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"
    AOA = 0
    MACH_NUMBER = 2.0
    FREESTREAM_PRESSURE = 7158.0701868
    FREESTREAM_TEMPERATURE = 216.65
    NUMBER_OF_PANEL = 12

    airfoil = Airfoil(FORCE_FILE,chord_length=1,num_samples=NUMBER_OF_PANEL,angle_of_attack=AOA)
    airfoil.panelGeneration()
    airfoil.Get_PanelCoefficients()

    info = np.loadtxt(FORCE_FILE, delimiter=',', skiprows=1)
    grid = {}
    grid['X'] = info[:, 1]
    grid['Y'] = info[:, 2]
    # plt.new_figure_manager(1)
    # plt.scatter(grid['X'], grid['Y'], s=0.1, label='Original Cl = ' + '{:.4f}'.format(float(airfoil.full_coefficientLift)))
    # plt.axis('equal')
    # plt.title('NACA64a203 Airfoil at ' + str(AOA) + ' angle of attack')
    # plt.legend()
    # plt.savefig('original')
    plt.new_figure_manager(2)
    plt.plot(airfoil.boundaryPoints_X, airfoil.boundaryPoints_Y,'-o', label='the panel model with %d panels'%(NUMBER_OF_PANEL),c='red')
    plt.scatter(grid['X'],grid['Y'],label='the original shape',c='blue',s=0.1)
    plt.legend(loc=1)
    plt.title('the vortex panel model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('vpm_model.eps')
