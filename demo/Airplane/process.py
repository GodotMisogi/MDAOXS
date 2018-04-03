from tools.SU2IO import read
from demo.Airplane.FEM3dComp import FEM3dComp
import numpy as np
from numpy.linalg import det as det
from scipy.sparse import csc_matrix
def _VandB(mesh, element_ID):
    """
    compute the volume of the linear Tetrahedron Elements
    :param element_ID:
    :return:
    :resource: (15.2),(15.3)
    """
    x1, y1, z1 = mesh[element_ID, 0, :3]
    x2, y2, z2 = mesh[element_ID, 1, :3]
    x3, y3, z3 = mesh[element_ID, 2, :3]
    x4, y4, z4 = mesh[element_ID, 3, :3]

    xyz = mesh[element_ID, :, :3]
    xyz = np.hstack([np.ones([4, 1]), xyz])
    V = det(xyz.astype('f8')) / 6

    mbeta1 = np.array([[1, y2, z2], [1, y3, z3], [1, y4, z4]])
    mbeta2 = np.array([[1, y1, z1], [1, y3, z3], [1, y4, z4]])
    mbeta3 = np.array([[1, y1, z1], [1, y2, z2], [1, y4, z4]])
    mbeta4 = np.array([[1, y1, z1], [1, y2, z2], [1, y3, z3]])

    mgamma1 = np.array([[1, x2, z2], [1, x3, z3], [1, x4, z4]])
    mgamma2 = np.array([[1, x1, z1], [1, x3, z3], [1, x4, z4]])
    mgamma3 = np.array([[1, x1, z1], [1, x2, z2], [1, x4, z4]])
    mgamma4 = np.array([[1, x1, z1], [1, x2, z2], [1, x3, z3]])

    mdelta1 = np.array([[1, x2, y2], [1, x3, y3], [1, x4, y4]])
    mdelta2 = np.array([[1, x1, y1], [1, x3, y3], [1, x4, y4]])
    mdelta3 = np.array([[1, x1, y1], [1, x2, y2], [1, x4, y4]])
    mdelta4 = np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]])

    beta1 = -1 * det(mbeta1)
    beta2 = det(mbeta2)
    beta3 = -1 * det(mbeta3)
    beta4 = det(mbeta4)

    gamma1 = det(mgamma1)
    gamma2 = -1 * det(mgamma2)
    gamma3 = det(mgamma3)
    gamma4 = -1 * det(mgamma4)

    delta1 = -1 * det(mdelta1)
    delta2 = det(mdelta2)
    delta3 = -1 * det(mdelta3)
    delta4 = det(mdelta4)

    B1 = np.array([[beta1, 0, 0],
                   [0, gamma1, 0],
                   [0, 0, delta1],
                   [gamma1, beta1, 0],
                   [0, delta1, gamma1],
                   [delta1, 0, beta1]])

    B2 = np.array([[beta2, 0, 0],
                   [0, gamma2, 0],
                   [0, 0, delta2],
                   [gamma2, beta2, 0],
                   [0, delta2, gamma2],
                   [delta2, 0, beta2]])

    B3 = np.array([[beta3, 0, 0],
                   [0, gamma3, 0],
                   [0, 0, delta3],
                   [gamma3, beta3, 0],
                   [0, delta3, gamma3],
                   [delta3, 0, beta3]])

    B4 = np.array([[beta4, 0, 0],
                   [0, gamma4, 0],
                   [0, 0, delta4],
                   [gamma4, beta4, 0],
                   [0, delta4, gamma4],
                   [delta4, 0, beta4]])

    B = np.hstack([B1, B2, B3, B4]) / (6 * V)
    return V, B

def generate_template(temp_global, i,j,m,n, element_num):
    # update 144 col in temp_global
    xlist = [3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 2 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1 - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * i - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 2 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1 - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * j - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 2 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1 - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * m - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 2 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1 - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1,3 * n - 1]
    ylist = [3 * i - 2 - 1,3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1, 3 * i - 2 - 1, 3 * i - 1 - 1, 3 * i - 1, 3 * j - 2 - 1, 3 * j - 1 - 1, 3 * j - 1, 3 * m - 2 - 1, 3 * m - 1 - 1, 3 * m - 1, 3 * n - 2 - 1, 3 * n - 1 - 1, 3 * n - 1]
    temp_global[:,144*element_num:144*(element_num+1)] = np.vstack((xlist,ylist))

def _K(K, k, i,j,m,n, element_id):


    K[3 * i - 2 - 1, 3 * i - 2 - 1] += k[1 - 1, 1 - 1]
    K[3 * i - 2 - 1, 3 * i - 1 - 1] += k[1 - 1, 2 - 1]
    K[3 * i - 2 - 1, 3 * i - 1] += k[1 - 1, 3 - 1]
    K[3 * i - 2 - 1, 3 * j - 2 - 1] += k[1 - 1, 4 - 1]
    K[3 * i - 2 - 1, 3 * j - 1 - 1] += + k[1 - 1, 5 - 1]
    K[3 * i - 2 - 1, 3 * j - 1] += k[1 - 1, 6 - 1]
    K[3 * i - 2 - 1, 3 * m - 2 - 1] += k[1 - 1, 7 - 1]
    K[3 * i - 2 - 1, 3 * m - 1 - 1] += k[1 - 1, 8 - 1]
    K[3 * i - 2 - 1, 3 * m - 1] += k[1 - 1, 9 - 1]
    K[3 * i - 2 - 1, 3 * n - 2 - 1] += k[1 - 1, 10 - 1]
    K[3 * i - 2 - 1, 3 * n - 1 - 1] += k[1 - 1, 11 - 1]
    K[3 * i - 2 - 1, 3 * n - 1] += k[1 - 1, 12 - 1]
    K[3 * i - 1 - 1, 3 * i - 2 - 1] += k[2 - 1, 1 - 1]
    K[3 * i - 1 - 1, 3 * i - 1 - 1] += k[2 - 1, 2 - 1]
    K[3 * i - 1 - 1, 3 * i - 1] += k[2 - 1, 3 - 1]
    K[3 * i - 1 - 1, 3 * j - 2 - 1] += k[2 - 1, 4 - 1]
    K[3 * i - 1 - 1, 3 * j - 1 - 1] += k[2 - 1, 5 - 1]
    K[3 * i - 1 - 1, 3 * j - 1] += k[2 - 1, 6 - 1]
    K[3 * i - 1 - 1, 3 * m - 2 - 1] += k[2 - 1, 7 - 1]
    K[3 * i - 1 - 1, 3 * m - 1 - 1] += k[2 - 1, 8 - 1]
    K[3 * i - 1 - 1, 3 * m - 1] += k[2 - 1, 9 - 1]
    K[3 * i - 1 - 1, 3 * n - 2 - 1] += k[2 - 1, 10 - 1]
    K[3 * i - 1 - 1, 3 * n - 1 - 1] += k[2 - 1, 11 - 1]
    K[3 * i - 1 - 1, 3 * n - 1] += k[2 - 1, 12 - 1]
    K[3 * i - 1, 3 * i - 2 - 1] += k[3 - 1, 1 - 1]
    K[3 * i - 1, 3 * i - 1 - 1] += k[3 - 1, 2 - 1]
    K[3 * i - 1, 3 * i - 1] += k[3 - 1, 3 - 1]
    K[3 * i - 1, 3 * j - 2 - 1] += k[3 - 1, 4 - 1]
    K[3 * i - 1, 3 * j - 1 - 1] += k[3 - 1, 5 - 1]
    K[3 * i - 1, 3 * j - 1] += k[3 - 1, 6 - 1]
    K[3 * i - 1, 3 * m - 2 - 1] += k[3 - 1, 7 - 1]
    K[3 * i - 1, 3 * m - 1 - 1] += k[3 - 1, 8 - 1]
    K[3 * i - 1, 3 * m - 1] += k[3 - 1, 9 - 1]
    K[3 * i - 1, 3 * n - 2 - 1] += k[3 - 1, 10 - 1]
    K[3 * i - 1, 3 * n - 1 - 1] += k[3 - 1, 11 - 1]
    K[3 * i - 1, 3 * n - 1] += k[3 - 1, 12 - 1]
    K[3 * j - 2 - 1, 3 * i - 2 - 1] += k[4 - 1, 1 - 1]
    K[3 * j - 2 - 1, 3 * i - 1 - 1] += k[4 - 1, 2 - 1]
    K[3 * j - 2 - 1, 3 * i - 1] += k[4 - 1, 3 - 1]
    K[3 * j - 2 - 1, 3 * j - 2 - 1] += k[4 - 1, 4 - 1]
    K[3 * j - 2 - 1, 3 * j - 1 - 1] += k[4 - 1, 5 - 1]
    K[3 * j - 2 - 1, 3 * j - 1] += k[4 - 1, 6 - 1]
    K[3 * j - 2 - 1, 3 * m - 2 - 1] += k[4 - 1, 7 - 1]
    K[3 * j - 2 - 1, 3 * m - 1 - 1] += k[4 - 1, 8 - 1]
    K[3 * j - 2 - 1, 3 * m - 1] += k[4 - 1, 9 - 1]
    K[3 * j - 2 - 1, 3 * n - 2 - 1] += k[4 - 1, 10 - 1]
    K[3 * j - 2 - 1, 3 * n - 1 - 1] += k[4 - 1, 11 - 1]
    K[3 * j - 2 - 1, 3 * n - 1] += k[4 - 1, 12 - 1]
    K[3 * j - 1 - 1, 3 * i - 2 - 1] += k[5 - 1, 1 - 1]
    K[3 * j - 1 - 1, 3 * i - 1 - 1] += k[5 - 1, 2 - 1]
    K[3 * j - 1 - 1, 3 * i - 1] += k[5 - 1, 3 - 1]
    K[3 * j - 1 - 1, 3 * j - 2 - 1] += k[5 - 1, 4 - 1]
    K[3 * j - 1 - 1, 3 * j - 1 - 1] += k[5 - 1, 5 - 1]
    K[3 * j - 1 - 1, 3 * j - 1] += k[5 - 1, 6 - 1]
    K[3 * j - 1 - 1, 3 * m - 2 - 1] += k[5 - 1, 7 - 1]
    K[3 * j - 1 - 1, 3 * m - 1 - 1] += k[5 - 1, 8 - 1]
    K[3 * j - 1 - 1, 3 * m - 1] += k[5 - 1, 9 - 1]
    K[3 * j - 1 - 1, 3 * n - 2 - 1] += k[5 - 1, 10 - 1]
    K[3 * j - 1 - 1, 3 * n - 1 - 1] += k[5 - 1, 11 - 1]
    K[3 * j - 1 - 1, 3 * n - 1] += k[5 - 1, 12 - 1]
    K[3 * j - 1, 3 * i - 2 - 1] += k[6 - 1, 1 - 1]
    K[3 * j - 1, 3 * i - 1 - 1] += k[6 - 1, 2 - 1]
    K[3 * j - 1, 3 * i - 1] += k[6 - 1, 3 - 1]
    K[3 * j - 1, 3 * j - 2 - 1] += k[6 - 1, 4 - 1]
    K[3 * j - 1, 3 * j - 1 - 1] += k[6 - 1, 5 - 1]
    K[3 * j - 1, 3 * j - 1] += k[6 - 1, 6 - 1]
    K[3 * j - 1, 3 * m - 2 - 1] += k[6 - 1, 7 - 1]
    K[3 * j - 1, 3 * m - 1 - 1] += k[6 - 1, 8 - 1]
    K[3 * j - 1, 3 * m - 1] += k[6 - 1, 9 - 1]
    K[3 * j - 1, 3 * n - 2 - 1] += k[6 - 1, 10 - 1]
    K[3 * j - 1, 3 * n - 1 - 1] += k[6 - 1, 11 - 1]
    K[3 * j - 1, 3 * n - 1] += k[6 - 1, 12 - 1]
    K[3 * m - 2 - 1, 3 * i - 2 - 1] += k[7 - 1, 1 - 1]
    K[3 * m - 2 - 1, 3 * i - 1 - 1] += k[7 - 1, 2 - 1]
    K[3 * m - 2 - 1, 3 * i - 1] += k[7 - 1, 3 - 1]
    K[3 * m - 2 - 1, 3 * j - 2 - 1] += k[7 - 1, 4 - 1]
    K[3 * m - 2 - 1, 3 * j - 1 - 1] += k[7 - 1, 5 - 1]
    K[3 * m - 2 - 1, 3 * j - 1] += k[7 - 1, 6 - 1]
    K[3 * m - 2 - 1, 3 * m - 2 - 1] += k[7 - 1, 7 - 1]
    K[3 * m - 2 - 1, 3 * m - 1 - 1] += k[7 - 1, 8 - 1]
    K[3 * m - 2 - 1, 3 * m - 1] += k[7 - 1, 9 - 1]
    K[3 * m - 2 - 1, 3 * n - 2 - 1] += k[7 - 1, 10 - 1]
    K[3 * m - 2 - 1, 3 * n - 1 - 1] += k[7 - 1, 11 - 1]
    K[3 * m - 2 - 1, 3 * n - 1] += k[7 - 1, 12 - 1]
    K[3 * m - 1 - 1, 3 * i - 2 - 1] += k[8 - 1, 1 - 1]
    K[3 * m - 1 - 1, 3 * i - 1 - 1] += k[8 - 1, 2 - 1]
    K[3 * m - 1 - 1, 3 * i - 1] += k[8 - 1, 3 - 1]
    K[3 * m - 1 - 1, 3 * j - 2 - 1] += k[8 - 1, 4 - 1]
    K[3 * m - 1 - 1, 3 * j - 1 - 1] += k[8 - 1, 5 - 1]
    K[3 * m - 1 - 1, 3 * j - 1] += k[8 - 1, 6 - 1]
    K[3 * m - 1 - 1, 3 * m - 2 - 1] += k[8 - 1, 7 - 1]
    K[3 * m - 1 - 1, 3 * m - 1 - 1] += k[8 - 1, 8 - 1]
    K[3 * m - 1 - 1, 3 * m - 1] += k[8 - 1, 9 - 1]
    K[3 * m - 1 - 1, 3 * n - 2 - 1] += k[8 - 1, 10 - 1]
    K[3 * m - 1 - 1, 3 * n - 1 - 1] += k[8 - 1, 11 - 1]
    K[3 * m - 1 - 1, 3 * n - 1] += k[8 - 1, 12 - 1]
    K[3 * m - 1, 3 * i - 2 - 1] += k[9 - 1, 1 - 1]
    K[3 * m - 1, 3 * i - 1 - 1] += k[9 - 1, 2 - 1]
    K[3 * m - 1, 3 * i - 1] += k[9 - 1, 3 - 1]
    K[3 * m - 1, 3 * j - 2 - 1] += k[9 - 1, 4 - 1]
    K[3 * m - 1, 3 * j - 1 - 1] += k[9 - 1, 5 - 1]
    K[3 * m - 1, 3 * j - 1] += k[9 - 1, 6 - 1]
    K[3 * m - 1, 3 * m - 2 - 1] += k[9 - 1, 7 - 1]
    K[3 * m - 1, 3 * m - 1 - 1] += k[9 - 1, 8 - 1]
    K[3 * m - 1, 3 * m - 1] += k[9 - 1, 9 - 1]
    K[3 * m - 1, 3 * n - 2 - 1] += k[9 - 1, 10 - 1]
    K[3 * m - 1, 3 * n - 1 - 1] += k[9 - 1, 11 - 1]
    K[3 * m - 1, 3 * n - 1] += k[9 - 1, 12 - 1]
    K[3 * n - 2 - 1, 3 * i - 2 - 1] += k[10 - 1, 1 - 1]
    K[3 * n - 2 - 1, 3 * i - 1 - 1] += k[10 - 1, 2 - 1]
    K[3 * n - 2 - 1, 3 * i - 1] += k[10 - 1, 3 - 1]
    K[3 * n - 2 - 1, 3 * j - 2 - 1] += k[10 - 1, 4 - 1]
    K[3 * n - 2 - 1, 3 * j - 1 - 1] += k[10 - 1, 5 - 1]
    K[3 * n - 2 - 1, 3 * j - 1] += k[10 - 1, 6 - 1]
    K[3 * n - 2 - 1, 3 * m - 2 - 1] += k[10 - 1, 7 - 1]
    K[3 * n - 2 - 1, 3 * m - 1 - 1] += k[10 - 1, 8 - 1]
    K[3 * n - 2 - 1, 3 * m - 1] += k[10 - 1, 9 - 1]
    K[3 * n - 2 - 1, 3 * n - 2 - 1] += k[10 - 1, 10 - 1]
    K[3 * n - 2 - 1, 3 * n - 1 - 1] += k[10 - 1, 11 - 1]
    K[3 * n - 2 - 1, 3 * n - 1] += k[10 - 1, 12 - 1]
    K[3 * n - 1 - 1, 3 * i - 2 - 1] += k[11 - 1, 1 - 1]
    K[3 * n - 1 - 1, 3 * i - 1 - 1] += k[11 - 1, 2 - 1]
    K[3 * n - 1 - 1, 3 * i - 1] += k[11 - 1, 3 - 1]
    K[3 * n - 1 - 1, 3 * j - 2 - 1] += k[11 - 1, 4 - 1]
    K[3 * n - 1 - 1, 3 * j - 1 - 1] += k[11 - 1, 5 - 1]
    K[3 * n - 1 - 1, 3 * j - 1] += k[11 - 1, 6 - 1]
    K[3 * n - 1 - 1, 3 * m - 2 - 1] += k[11 - 1, 7 - 1]
    K[3 * n - 1 - 1, 3 * m - 1 - 1] += k[11 - 1, 8 - 1]
    K[3 * n - 1 - 1, 3 * m - 1] += k[11 - 1, 9 - 1]
    K[3 * n - 1 - 1, 3 * n - 2 - 1] += k[11 - 1, 10 - 1]
    K[3 * n - 1 - 1, 3 * n - 1 - 1] += k[11 - 1, 11 - 1]
    K[3 * n - 1 - 1, 3 * n - 1] += k[11 - 1, 12 - 1]
    K[3 * n - 1, 3 * i - 2 - 1] += k[12 - 1, 1 - 1]
    K[3 * n - 1, 3 * i - 1 - 1] += k[12 - 1, 2 - 1]
    K[3 * n - 1, 3 * i - 1] += k[12 - 1, 3 - 1]
    K[3 * n - 1, 3 * j - 2 - 1] += k[12 - 1, 4 - 1]
    K[3 * n - 1, 3 * j - 1 - 1] += k[12 - 1, 5 - 1]
    K[3 * n - 1, 3 * j - 1] += k[12 - 1, 6 - 1]
    K[3 * n - 1, 3 * m - 2 - 1] += k[12 - 1, 7 - 1]
    K[3 * n - 1, 3 * m - 1 - 1] += k[12 - 1, 8 - 1]
    K[3 * n - 1, 3 * m - 1] += k[12 - 1, 9 - 1]
    K[3 * n - 1, 3 * n - 2 - 1] += k[12 - 1, 10 - 1]
    K[3 * n - 1, 3 * n - 1 - 1] += k[12 - 1, 11 - 1]
    K[3 * n - 1, 3 * n - 1] += k[12 - 1, 12 - 1]


mesh = read('/Users/gakki/Dropbox/thesis/Optimale17_1_1_Mesh1.su2')
num_elements = mesh['NELEM']
num_nodes = mesh['NPOIN']

np_mesh = np.zeros(shape=(num_elements,4,4),dtype=object)
for element_ID in range(num_elements):
    for local_node_ID in range(4):
        node_ID = mesh['ELEM'][element_ID][1:][local_node_ID]
        x,y,z = mesh['POIN'][node_ID]
        np_mesh[element_ID,local_node_ID,:] = x,y,z,node_ID

"""
K = VB'DB
"""
E = 1
NU = 1
"""
D is irrelevant to any position information (15.9)
shape=(6,6), dtype=float
"""
D = np.diag(np.array([1 - NU, 1 - NU, 1 - NU, (1 - 2 * NU) / 2, (1 - 2 * NU) / 2, (1 - 2 * NU) / 2]))
D[0, (1, 2)] = D[1, (0, 2)] = D[2, (0, 1)] = NU
D = E / ((1 + NU) * (1 - 2 * NU)) * D

# assemble template first
template_nonzero_matrix = np.zeros(shape=(2,144*num_elements))
for element_ID in range(num_elements):
    i = np_mesh[element_ID, 0, 3] + 1
    j = np_mesh[element_ID, 1, 3] + 1
    m = np_mesh[element_ID, 2, 3] + 1
    n = np_mesh[element_ID, 3, 3] + 1
    generate_template(template_nonzero_matrix,i,j,m,n,element_ID)

"""
K: global stiffness matrix
shape=(3*num_node,3*num_node)
"""
K=csc_matrix((np.zeros(shape=(144*num_elements)), (template_nonzero_matrix[0,:], template_nonzero_matrix[1,:])), shape=(3 * num_nodes, 3 * num_nodes))

for element_ID in range(num_elements):
    """
    B and V are dependent on the element_ID
    """
    V, B = _VandB(np_mesh, element_ID)
    """
    k: local stiffness matrix of element_on
    """
    i = np_mesh[element_ID, 0, 3] + 1
    j = np_mesh[element_ID, 1, 3] + 1
    m = np_mesh[element_ID, 2, 3] + 1
    n = np_mesh[element_ID, 3, 3] + 1
    k = np.matmul(np.matmul(V * np.transpose(B),D), B)
    _K(K, k, i,j,m,n, element_ID)


from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp, NonlinearBlockGS, PETScKrylov
prob = Problem()
model = prob.model = Group()

np.random.seed(0)
num_elements = 400
model.add_subsystem('pK', IndepVarComp('K', np.random.rand(num_elements,num_elements)))
model.add_subsystem('pf', IndepVarComp('f', np.eye(num_elements,1)))
model.add_subsystem('Kdeqf', FEM3dComp(num_elements=num_elements,num_nodes=num_nodes))
model.connect('pK.K', 'Kdeqf.K')
model.connect('pf.f', 'Kdeqf.f')

model.nonlinear_solver = NonlinearBlockGS()

model.linear_solver = PETScKrylov()

prob.setup()
prob.run_model()
prob['Kdeqf.d']
