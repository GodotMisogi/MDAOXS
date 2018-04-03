import numpy as np
from numpy.linalg import det as det

###########################################
def LinearTriangleElementArea(xi,yi,xj,yj,xm,ym):
    return (xi*(yj - ym) + xj*(ym-yi) + xm*(yi-yj))/2


def LinearTriangleElementStiffness(E,NU,t,xi,yi,xj,yj,xm,ym,p):
    A = LinearTriangleElementArea(xi,yi,xj,yj,xm,ym)
    betai = yj-ym
    betaj = ym-yi
    betam = yi-yj
    gammai = xm-xj
    gammaj = xi-xm
    gammam = xj-xi
    B = np.array([[betai,0,betaj,0,betam,0],
         [0,gammai,0,gammaj,0,gammam],
         [gammai,betai,gammaj,betaj,gammam,betam]])/ (2 * A)
    if p == 1:
        D = (E / (1 - NU * NU)) * \
            np.array([[1,NU,0],
             [NU,1,0],
             [0,0,(1-NU)/2]])
    elif p == 2:
        D = (E / (1 + NU) / (1 - 2 * NU)) * \
            np.array([[1 - NU,NU,0],
             [NU,1-NU,0],
             [0,0,(1-2*NU)/2]])
    return t * np.matmul(A*B.transpose(),np.matmul(D,B))

def LinearTriangleAssemble(K,k,i,j,m):
    K[2 * i-1-1, 2 * i-1-1] = K[2 * i-1-1, 2 * i-1-1] + k[1-1, 1-1]
    K[2 * i-1-1, 2 * i-1] = K[2 * i-1-1, 2 * i-1] + k[1-1, 2-1]
    K[2 * i-1-1, 2 * j-1-1] = K[2 * i-1-1, 2 * j-1-1] + k[1-1, 3-1]
    K[2*i-1-1,2*j-1] = K[2*i-1-1,2*j-1] + k[1-1,4-1]
    K[2*i-1-1,2*m-1-1] = K[2*i-1-1,2*m-1-1] + k[1-1,5-1]
    K[2*i-1-1,2*m-1] = K[2*i-1-1,2*m-1] + k[1-1,6-1]
    K[2*i-1,2*i-1-1] = K[2*i-1,2*i-1-1] + k[2-1,1-1]
    K[2*i-1,2*i-1] = K[2*i-1,2*i-1] + k[2-1,2-1]
    K[2*i-1,2*j-1-1] = K[2*i-1,2*j-1-1] + k[2-1,3-1]
    K[2*i-1,2*j-1] = K[2*i-1,2*j-1] + k[2-1,4-1]
    K[2*i-1,2*m-1-1] = K[2*i-1,2*m-1-1] + k[2-1,5-1]
    K[2*i-1,2*m-1] = K[2*i-1,2*m-1] + k[2-1,6-1]
    K[2*j-1-1,2*i-1-1] = K[2*j-1-1,2*i-1-1] + k[3-1,1-1]
    K[2*j-1-1,2*i-1] = K[2*j-1-1,2*i-1] + k[3-1,2-1]
    K[2*j-1-1,2*j-1-1] = K[2*j-1-1,2*j-1-1] + k[3-1,3-1]
    K[2*j-1-1,2*j-1] = K[2*j-1-1,2*j-1] + k[3-1,4-1]
    K[2*j-1-1,2*m-1-1] = K[2*j-1-1,2*m-1-1] + k[3-1,5-1]
    K[2*j-1-1,2*m-1] = K[2*j-1-1,2*m-1] + k[3-1,6-1]
    K[2*j-1,2*i-1-1] = K[2*j-1,2*i-1-1] + k[4-1,1-1]
    K[2*j-1,2*i-1] = K[2*j-1,2*i-1] + k[4-1,2-1]
    K[2*j-1,2*j-1-1] = K[2*j-1,2*j-1-1] + k[4-1,3-1]
    K[2*j-1,2*j-1] = K[2*j-1,2*j-1] + k[4-1,4-1]
    K[2*j-1,2*m-1-1] = K[2*j-1,2*m-1-1] + k[4-1,5-1]
    K[2*j-1,2*m-1] = K[2*j-1,2*m-1] + k[4-1,6-1]
    K[2*m-1-1,2*i-1-1] = K[2*m-1-1,2*i-1-1] + k[5-1,1-1]
    K[2*m-1-1,2*i-1] = K[2*m-1-1,2*i-1] + k[5-1,2-1]
    K[2*m-1-1,2*j-1-1] = K[2*m-1-1,2*j-1-1] + k[5-1,3-1]
    K[2*m-1-1,2*j-1] = K[2*m-1-1,2*j-1] + k[5-1,4-1]
    K[2*m-1-1,2*m-1-1] = K[2*m-1-1,2*m-1-1] + k[5-1,5-1]
    K[2*m-1-1,2*m-1] = K[2*m-1-1,2*m-1] + k[5-1,6-1]
    K[2*m-1,2*i-1-1] = K[2*m-1,2*i-1-1] + k[6-1,1-1]
    K[2*m-1,2*i-1] = K[2*m-1,2*i-1] + k[6-1,2-1]
    K[2*m-1,2*j-1-1] = K[2*m-1,2*j-1-1] + k[6-1,3-1]
    K[2*m-1,2*j-1] = K[2*m-1,2*j-1] + k[6-1,4-1]
    K[2*m-1,2*m-1-1] = K[2*m-1,2*m-1-1] + k[6-1,5-1]
    K[2*m-1,2*m-1] = K[2*m-1,2*m-1] + k[6-1,6-1]

def generate_template(temp_global, i,j,m, element_num):
    # update 144 col in temp_global
    xlist = [2 * i-1-1,2 * i-1-1,2 * i-1-1,2*i-1-1,2*i-1-1,2*i-1-1,2*i-1,2*i-1,2*i-1,2*i-1,2*i-1,2*i-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1,2*j-1,2*j-1,2*j-1,2*j-1,2*j-1,2*m-1-1,2*m-1-1,2*m-1-1,2*m-1-1,2*m-1-1,2*m-1-1,2*m-1,2*m-1,2*m-1,2*m-1,2*m-1,2*m-1]
    ylist = [2 * i-1-1,2 * i-1,2 * j-1-1,2*j-1,2*m-1-1,2*m-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*m-1-1,2*m-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*m-1-1,2*m-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*m-1-1,2*m-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*m-1-1,2*m-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*m-1-1,2*m-1]
    temp_global[:,36*element_num:36*(element_num+1)] = np.vstack((xlist,ylist))

#########################################
######          force       #############
#########################################
def LinearTriangleElementStresses(E,NU,xi,yi,xj,yj,xm,ym,p,u):
    A = LinearTriangleElementArea(xi,yi,xj,yj,xm,ym)
    betai = yj-ym
    betaj = ym-yi
    betam = yi-yj
    gammai = xm-xj
    gammaj = xi-xm
    gammam = xj-xi
    B = np.array([[betai, 0, betaj, 0, betam, 0],
                  [0, gammai, 0, gammaj, 0, gammam],
                  [gammai, betai, gammaj, betaj, gammam, betam]]) / (2 * A)
    if p == 1:
        D = (E / (1 - NU * NU)) * \
            np.array([[1,NU,0],
             [NU,1,0],
             [0,0,(1-NU)/2]])
    elif p == 2:
        D = (E / (1 + NU) / (1 - 2 * NU)) * \
            np.array([[1 - NU,NU,0],
             [NU,1-NU,0],
             [0,0,(1-2*NU)/2]])
    return np.matmul(np.matmul(D,B),u) # return a 3*1 vector

def LinearTriangleElementPStresses(sigma):
    R = (sigma[0] + sigma[1]) / 2;
    Q = ((sigma[0] - sigma[1]) / 2)**2 + sigma[2] * sigma[2];
    M = 2 * sigma[2] / (sigma[0] - sigma[1])
    s1 = R + np.sqrt(Q);
    s2 = R - np.sqrt(Q);
    theta = (np.arctan(M) / 2) * 180 / np.pi
    return np.array([[s1],[s2],[theta]])


def PlaneFrameElementLength(x1,y1,x2,y2):
    """
    %PlaneFrameElementLength      This function returns the length of the
    #                             plane frame element whose first node has
    %                             coordinates (x1,y1) and second node has
    %                             coordinates (x2,y2).
    """
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def computeN_VEC(x0,y0,x1,y1,x2,y2):
    #A = LinearTriangleElementArea(x0,y0,x1,y1,x2,y2)
    #L0 = PlaneFrameElementLength(x0,y0,x1,y1)
    #L1 = PlaneFrameElementLength(x1,y1,x2,y2)
    #L2 = PlaneFrameElementLength(x2,y2,x0,y0)
    v = np.array([x2,y2]) - np.array([x0,y0])
    if v[0] == 0:
        return normalizeVector([0,-1/v[1]])
    if v[1] == 0:
        return normalizeVector([-1/v[0],0])
    return normalizeVector([1,(-1-v[0])/v[1]])

def normalizeVector(v):

    norm = np.linalg.norm(v)
    assert(norm != 0)
    return v/norm