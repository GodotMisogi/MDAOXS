import numpy as np
ASSEMBLE_ENTRIES = 16



def BeamElementLength(x1,y1,x2,y2):
    """
    %PlaneFrameElementLength      This function returns the length of the
    #                             plane frame element whose first node has
    %                             coordinates (x1,y1) and second node has
    %                             coordinates (x2,y2).
    """
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


def BeamElementStiffness(E,I,L):
    """
    #%BeamElementStiffness          This function returns the element
    #%                              stiffness matrix for a beam
    #%                              element with modulus of elasticity E,
    #%                              moment of inertia I, and length L.
    #%                              The size of the element stiffness
    #%                              matrix is 4 x 4.
    """
    return E*I/(L*L*L) * np.array([[12,     6*L,    -12,    6*L],
                                   [6*L,    4*L*L,  -6*L,   2*L*L],
                                   [-12,    -6*L,   12,     -6*L],
                                   [6*L,    2*L*L,  -6*L,   4*L*L]])


def BeamAssemble(K,k,i,j):
    """
    %BeamAssemble            This function assembles the element stiffness
    %                        matrix k of the beam element with nodes
    %                        i and j into the global stiffness matrix K.
    %                        This function returns the global stiffness
    %                        matrix K after the element stiffness matrix
    %                        k is assembled.
    # 16
    """
    K[2 * i - 1 - 1, 2 * i - 1 - 1] = K[2 * i - 1 - 1, 2 * i - 1 - 1] + k[1 - 1, 1 - 1]
    K[2 * i - 1 - 1, 2 * i - 1] = K[2 * i - 1 - 1, 2 * i - 1] + k[1 - 1, 2 - 1]
    K[2 * i - 1 - 1, 2 * j - 1 - 1] = K[2 * i - 1 - 1, 2 * j - 1 - 1] + k[1 - 1, 3 - 1]
    K[2 * i - 1 - 1, 2 * j - 1] = K[2 * i - 1 - 1, 2 * j - 1] + k[1 - 1, 4 - 1]
    K[2 * i - 1, 2 * i - 1 - 1] = K[2 * i - 1, 2 * i - 1 - 1] + k[2 - 1, 1 - 1]
    K[2 * i - 1, 2 * i - 1] = K[2 * i - 1, 2 * i - 1] + k[2 - 1, 2 - 1]
    K[2 * i - 1, 2 * j - 1 - 1] = K[2 * i - 1, 2 * j - 1 - 1] + k[2 - 1, 3 - 1]
    K[2 * i - 1, 2 * j - 1] = K[2 * i - 1, 2 * j - 1] + k[2 - 1, 4 - 1]
    K[2 * j - 1 - 1, 2 * i - 1 - 1] = K[2 * j - 1 - 1, 2 * i - 1 - 1] + k[3 - 1, 1 - 1]
    K[2 * j - 1 - 1, 2 * i - 1] = K[2 * j - 1 - 1, 2 * i - 1] + k[3 - 1, 2 - 1]
    K[2 * j - 1 - 1, 2 * j - 1 - 1] = K[2 * j - 1 - 1, 2 * j - 1 - 1] + k[3 - 1, 3 - 1]
    K[2 * j - 1 - 1, 2 * j - 1] = K[2 * j - 1 - 1, 2 * j - 1] + k[3 - 1, 4 - 1]
    K[2 * j - 1, 2 * i - 1 - 1] = K[2 * j - 1, 2 * i - 1 - 1] + k[4 - 1, 1 - 1]
    K[2 * j - 1, 2 * i - 1] = K[2 * j - 1, 2 * i - 1] + k[4 - 1, 2 - 1]
    K[2 * j - 1, 2 * j - 1 - 1] = K[2 * j - 1, 2 * j - 1 - 1] + k[4 - 1, 3 - 1]
    K[2 * j - 1, 2 * j - 1] = K[2 * j - 1, 2 * j - 1] + k[4 - 1, 4 - 1]

def BeamElementForces(k,u):
    """
    %BeamElementForces       This function returns the element nodal force
    %                        vector given the element stiffness matrix k
    %                        and the element nodal displacement vector u.
    """
    return k * u



def generate_template(temp_global, i, j, element_num):
    # update 36 col in temp_global
    xlist = [2*i-1-1,2*i-1-1,2*i-1-1,2*i-1-1,2*i-1,2*i-1,2*i-1,2*i-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1-1,2*j-1,2*j-1,2*j-1,2*j-1]
    ylist = [2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1,2*i-1-1,2*i-1,2*j-1-1,2*j-1]
    temp_global[:,ASSEMBLE_ENTRIES*element_num:ASSEMBLE_ENTRIES*(element_num+1)] = np.vstack((xlist,ylist))


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


def BeamElementCS(x1,y1,x2,y2,L):
    v = np.array([x2,y2]) - np.array([x1,y1])
    C = v[0]/L
    S = v[1]/L
    return C,S