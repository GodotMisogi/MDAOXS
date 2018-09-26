import numpy as np
ASSEMBLE_ENTRIES = 36


def PlaneFrameElementLength(x1,y1,x2,y2):
    """
    %PlaneFrameElementLength      This function returns the length of the
    #                             plane frame element whose first node has
    %                             coordinates (x1,y1) and second node has
    %                             coordinates (x2,y2).
    """
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def PlaneFrameElementCS(x1,y1,x2,y2,L):
    v = np.array([x2,y2]) - np.array([x1,y1])
    C = v[0]/L
    S = v[1]/L
    return C,S

def PlaneFrameElementStiffness(E,A,I,L,C,S):
    """
    %PlaneFrameElementStiffness   This function returns the element
    %                             stiffness matrix for a plane frame
    %                             element with modulus of elasticity E,
    %                             cross-sectional area A, moment of
    %                             inertia I, length L, and angle
    %                             theta (in degrees).
    %                             The size of the element stiffness
    %                             matrix is 6 x 6.
    """
    #x = theta*np.pi/180
    #C = np.cos(x)
    #S = np.sin(x)
    w1 = A*C*C + 12*I*S*S/(L*L)
    w2 = A*S*S + 12*I*C*C/(L*L)
    w3 = (A-12*I/(L*L))*C*S
    w4 = 6*I*S/L
    w5 = 6*I*C/L
    return E/L*np.array([[w1,w3,-w4,-w1,-w3,-w4],
                         [w3,w2,w5,-w3,-w2,w5],
                         [-w4,w5,4*I,w4,-w5,2*I],
                         [-w1,-w3,w4,w1,w3,w4],
                         [-w3,-w2,-w5,w3,w2,-w5],
                         [-w4,w5,2*I,w4,-w5,4*I]])


def PlaneFrameAssemble(K, k, i, j):
    """
    %PlaneFrameAssemble This function assembles the element stiffness
    % matrix k of the plane frame element with nodes
    % i and j into the global stiffness matrix K.
    % This function returns the global stiffness
    % matrix K after the element stiffness matrix
    % k is assembled.
    """
    K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 2 - 1] = K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 2 - 1] + k[1 - 1, 1 - 1]
    K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 1 - 1] = K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 1 - 1] + k[1 - 1, 2 - 1]
    K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 1] = K[3 * (i + 1) - 2 - 1, 3 * (i + 1) - 1] + k[1 - 1, 3 - 1]
    K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 2 - 1] = K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 2 - 1] + k[1 - 1, 4 - 1]
    K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 1 - 1] = K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 1 - 1] + k[1 - 1, 5 - 1]
    K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 1] = K[3 * (i + 1) - 2 - 1, 3 * (j + 1) - 1] + k[1 - 1, 6 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 2 - 1] = K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 2 - 1] + k[2 - 1, 1 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 1 - 1] = K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 1 - 1] + k[2 - 1, 2 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 1] = K[3 * (i + 1) - 1 - 1, 3 * (i + 1) - 1] + k[2 - 1, 3 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 2 - 1] = K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 2 - 1] + k[2 - 1, 4 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 1 - 1] = K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 1 - 1] + k[2 - 1, 5 - 1]
    K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 1] = K[3 * (i + 1) - 1 - 1, 3 * (j + 1) - 1] + k[2 - 1, 6 - 1]
    K[3 * (i + 1) - 1, 3 * (i + 1) - 2 - 1] = K[3 * (i + 1) - 1, 3 * (i + 1) - 2 - 1] + k[3 - 1, 1 - 1]
    K[3 * (i + 1) - 1, 3 * (i + 1) - 1 - 1] = K[3 * (i + 1) - 1, 3 * (i + 1) - 1 - 1] + k[3 - 1, 2 - 1]
    K[3 * (i + 1) - 1, 3 * (i + 1) - 1] = K[3 * (i + 1) - 1, 3 * (i + 1) - 1] + k[3 - 1, 3 - 1]
    K[3 * (i + 1) - 1, 3 * (j + 1) - 2 - 1] = K[3 * (i + 1) - 1, 3 * (j + 1) - 2 - 1] + k[3 - 1, 4 - 1]
    K[3 * (i + 1) - 1, 3 * (j + 1) - 1 - 1] = K[3 * (i + 1) - 1, 3 * (j + 1) - 1 - 1] + k[3 - 1, 5 - 1]
    K[3 * (i + 1) - 1, 3 * (j + 1) - 1] = K[3 * (i + 1) - 1, 3 * (j + 1) - 1] + k[3 - 1, 6 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 2 - 1] = K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 2 - 1] + k[4 - 1, 1 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 1 - 1] = K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 1 - 1] + k[4 - 1, 2 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 1] = K[3 * (j + 1) - 2 - 1, 3 * (i + 1) - 1] + k[4 - 1, 3 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 2 - 1] = K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 2 - 1] + k[4 - 1, 4 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 1 - 1] = K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 1 - 1] + k[4 - 1, 5 - 1]
    K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 1] = K[3 * (j + 1) - 2 - 1, 3 * (j + 1) - 1] + k[4 - 1, 6 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 2 - 1] = K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 2 - 1] + k[5 - 1, 1 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 1 - 1] = K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 1 - 1] + k[5 - 1, 2 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 1] = K[3 * (j + 1) - 1 - 1, 3 * (i + 1) - 1] + k[5 - 1, 3 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 2 - 1] = K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 2 - 1] + k[5 - 1, 4 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 1 - 1] = K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 1 - 1] + k[5 - 1, 5 - 1]
    K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 1] = K[3 * (j + 1) - 1 - 1, 3 * (j + 1) - 1] + k[5 - 1, 6 - 1]
    K[3 * (j + 1) - 1, 3 * (i + 1) - 2 - 1] = K[3 * (j + 1) - 1, 3 * (i + 1) - 2 - 1] + k[6 - 1, 1 - 1]
    K[3 * (j + 1) - 1, 3 * (i + 1) - 1 - 1] = K[3 * (j + 1) - 1, 3 * (i + 1) - 1 - 1] + k[6 - 1, 2 - 1]
    K[3 * (j + 1) - 1, 3 * (i + 1) - 1] = K[3 * (j + 1) - 1, 3 * (i + 1) - 1] + k[6 - 1, 3 - 1]
    K[3 * (j + 1) - 1, 3 * (j + 1) - 2 - 1] = K[3 * (j + 1) - 1, 3 * (j + 1) - 2 - 1] + k[6 - 1, 4 - 1]
    K[3 * (j + 1) - 1, 3 * (j + 1) - 1 - 1] = K[3 * (j + 1) - 1, 3 * (j + 1) - 1 - 1] + k[6 - 1, 5 - 1]
    K[3 * (j + 1) - 1, 3 * (j + 1) - 1] = K[3 * (j + 1) - 1, 3 * (j + 1) - 1] + k[6 - 1, 6 - 1]

def generate_template(temp_global, i, j, element_num):
    # update 36 col in temp_global
    xlist = [3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-2-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1-1,3*i-1,3*i-1,3*i-1,3*i-1,3*i-1,3*i-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-2-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1-1,3*j-1,3*j-1,3*j-1,3*j-1,3*j-1,3*j-1]
    ylist = [3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1,3*i-2-1,3*i-1-1,3*i-1,3*j-2-1,3*j-1-1,3*j-1]
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