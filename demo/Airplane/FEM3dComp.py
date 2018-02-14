from __future__ import division
import numpy as np
from numpy.linalg import det as det
from openmdao.api import ImplicitComponent
from scipy.sparse.linalg import gmres as gmres_solve

class FEM3dComp(ImplicitComponent):

    def initialize(self):
        """
        initialize function
        :return: void
        """

        """
        num_elements: total number of linear tetrahedral (solid) element
        shape = 1, dtype = int
        """
        self.metadata.declare('num_elements', types=int)

        """
        num_node: total number of nodes
        """
        self.metadata.declare('num_node',types=int)

        """
        mesh: FEM mesh information. For an unstructured mesh, it need some extra calculation (outside openmdao)
        shape = (element_ID, [x,y,z,nodeID], 4)
        """
        self.metadata.declare('mesh')

        """
        ! involved into mesh
        node_positions: store the x,y,z value of each node in a row and sorted by ID
        shape = (total_node, 3), dtype = np.array
        loaded from a txt file
        """
        #self.metadata.declare('node_positions') # shape = (total_node,3)

        """
        ! involved into mesh
        adjacent_T: store the ID of 4 nodes in the same elements in a row  
        shape = (num_elements, 4), dtype = np.array 
        """
        # self.metadata.declare('adjacent_T')

        """
        E: modulus of elasticity
        shape = 1, val = 1, dtype = int
        """
        self.metadata.declare('E')

        """
        NU: Poisson's ratio
        shape = 1, val = 1, dtype = int
        """
        self.metadata.declare('NU')

    def setup(self):
        num_elements = self.metadata['num_elements']
        E = self.metadata['E']
        NU = self.metadata['NU']
        mesh = self.metadata['mesh']
        num_node = self.metadata['num_node']


        """
        f: unassemembled force vector. Although we only have single-direction force, we assume it to be (fx,fy,fz) 
        shape = (3 * num_elements,1)
        """
        self.add_input('f', shape=(3 * num_elements,1))
        self.add_output('d', shape=(num_elements, 4, 4))

        """
        K = VB'DB
        """
        """
        D is irrelevant to any position information (15.9)
        shape=(6,6), dtype=float
        """
        D = np.diag(np.array([1-NU,1-NU,1-NU,(1-2*NU)/2,(1-2*NU)/2,(1-2*NU)/2]))
        D[0,(1,2)]=D[1,(0,2)]=D[2,(0,1)] = NU
        D = E/((1+NU)*(1-2*NU))*D

        """
        K: global stiffness matrix
        shape=(3*num_node,3*num_node)
        """
        K = np.zeros([3*num_node,3*num_node])

        for element_ID in range(num_elements):
            """
            B and V are dependent on the element_ID
            """
            V,B = self.__VandB(mesh,element_ID)
            """
            k: local stiffness matrix of element_on
            """
            k = V * np.transpose(B) * D * B
            K = self.__K(K, k, element_ID)


    def __VandB(self, mesh, element_ID):
        """
        compute the volume of the linear Tetrahedron Elements
        :param element_ID:
        :return:
        :resource: (15.2),(15.3)
        """
        E = self.metadata['E']
        NU = self.metadata['NU']
        x1, y1, z1 = mesh[element_ID, 0, :3]
        x2, y2, z2 = mesh[element_ID, 1, :3]
        x3, y3, z3 = mesh[element_ID, 2, :3]
        x4, y4, z4 = mesh[element_ID, 3, :3]

        xyz = np.transpose(mesh[element_ID,:3,:])
        xyz = np.hstack([np.ones([4,1]),xyz])
        V = det(xyz) / 6

        mbeta1 = np.array([[1, y2, z2],[1, y3, z3], [1, y4, z4]])
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

    def __K(self, K, k, mesh, element_id):
        i = mesh[element_id, 0, 3]
        j = mesh[element_id, 1, 3]
        m = mesh[element_id, 2, 3]
        n = mesh[element_id, 3, 3]

        K[3 * i-2, 3 * i-2] = K[3 * i-2, 3 * i-2] + k[1, 1]
        K[3 * i-2, 3 * i-1] = K[3 * i-2, 3 * i-1] + k[1, 2]
        K[3 * i-2, 3 * i] = K[3 * i-2, 3 * i] + k[1, 3]
        K[3 * i-2, 3 * j-2] = K[3 * i-2, 3 * j-2] + k[1, 4]
        K[3 * i-2, 3 * j-1] = K[3 * i-2, 3 * j-1] + k[1, 5]
        K[3 * i-2, 3 * j] = K[3 * i-2, 3 * j] + k[1, 6]
        K[3 * i-2, 3 * m-2] = K[3 * i-2, 3 * m-2] + k[1, 7]
        K[3 * i-2, 3 * m-1] = K[3 * i-2, 3 * m-1] + k[1, 8]
        K[3 * i-2, 3 * m] = K[3 * i-2, 3 * m] + k[1, 9]
        K[3 * i-2, 3 * n-2] = K[3 * i-2, 3 * n-2] + k[1, 10]
        K[3 * i-2, 3 * n-1] = K[3 * i-2, 3 * n-1] + k[1, 11]
        K[3 * i-2, 3 * n] = K[3 * i-2, 3 * n] + k[1, 12]
        K[3 * i-1, 3 * i-2] = K[3 * i-1, 3 * i-2] + k[2, 1]
        K[3 * i-1, 3 * i-1] = K[3 * i-1, 3 * i-1] + k[2, 2]
        K[3 * i-1, 3 * i] = K[3 * i-1, 3 * i] + k[2, 3]
        K[3 * i-1, 3 * j-2] = K[3 * i-1, 3 * j-2] + k[2, 4]
        K[3 * i-1, 3 * j-1] = K[3 * i-1, 3 * j-1] + k[2, 5]
        K[3 * i-1, 3 * j] = K[3 * i-1, 3 * j] + k[2, 6]
        K[3 * i-1, 3 * m-2] = K[3 * i-1, 3 * m-2] + k[2, 7]
        K[3 * i-1, 3 * m-1] = K[3 * i-1, 3 * m-1] + k[2, 8]
        K[3 * i-1, 3 * m] = K[3 * i-1, 3 * m] + k[2, 9]
        K[3 * i-1, 3 * n-2] = K[3 * i-1, 3 * n-2] + k[2, 10]
        K[3 * i-1, 3 * n-1] = K[3 * i-1, 3 * n-1] + k[2, 11]
        K[3 * i-1, 3 * n] = K[3 * i-1, 3 * n] + k[2, 12]
        K[3 * i, 3 * i-2] = K[3 * i, 3 * i-2] + k[3, 1]
        K[3 * i, 3 * i-1] = K[3 * i, 3 * i-1] + k[3, 2]
        K[3 * i, 3 * i] = K[3 * i, 3 * i] + k[3, 3]
        K[3 * i, 3 * j-2] = K[3 * i, 3 * j-2] + k[3, 4]
        K[3 * i, 3 * j-1] = K[3 * i, 3 * j-1] + k[3, 5]
        K[3 * i, 3 * j] = K[3 * i, 3 * j] + k[3, 6]
        K[3 * i, 3 * m - 2] = K[3 * i, 3 * m-2] + k[3, 7]
        K[3 * i, 3 * m-1] = K[3 * i, 3 * m-1] + k[3, 8]
        K[3 * i, 3 * m] = K[3 * i, 3 * m] + k[3, 9]
        K[3 * i, 3 * n-2] = K[3 * i, 3 * n-2] + k[3, 10]
        K[3 * i, 3 * n-1] = K[3 * i, 3 * n-1] + k[3, 11]
        K[3 * i, 3 * n] = K[3 * i, 3 * n] + k[3, 12]
        K[3 * j-2, 3 * i-2] = K[3 * j-2, 3 * i-2] + k[4, 1]
        K[3 * j-2, 3 * i-1] = K[3 * j-2, 3 * i-1] + k[4, 2]
        K[3 * j-2, 3 * i] = K[3 * j-2, 3 * i] + k[4, 3]
        K[3 * j-2, 3 * j-2] = K[3 * j-2, 3 * j-2] + k[4, 4]
        K[3 * j-2, 3 * j-1] = K[3 * j-2, 3 * j-1] + k[4, 5]
        K[3 * j-2, 3 * j] = K[3 * j-2, 3 * j] + k[4, 6]
        K[3 * j-2, 3 * m-2] = K[3 * j-2, 3 * m-2] + k[4, 7]
        K[3 * j-2, 3 * m-1] = K[3 * j-2, 3 * m-1] + k[4, 8]
        K[3 * j-2, 3 * m] = K[3 * j-2, 3 * m] + k[4, 9]
        K[3 * j-2, 3 * n-2] = K[3 * j-2, 3 * n-2] + k[4, 10]
        K[3 * j-2, 3 * n-1] = K[3 * j-2, 3 * n-1] + k[4, 11]
        K[3 * j-2, 3 * n] = K[3 * j-2, 3 * n] + k[4, 12]
        K[3 * j-1, 3 * i-2] = K[3 * j-1, 3 * i-2] + k[5, 1]
        K[3 * j-1, 3 * i-1] = K[3 * j-1, 3 * i-1] + k[5, 2]
        K[3 * j-1, 3 * i] = K[3 * j-1, 3 * i] + k[5, 3]
        K[3 * j-1, 3 * j-2] = K[3 * j-1, 3 * j-2] + k[5, 4]
        K[3 * j-1, 3 * j-1] = K[3 * j-1, 3 * j-1] + k[5, 5]
        K[3 * j-1, 3 * j] = K[3 * j-1, 3 * j] + k[5, 6]
        K[3 * j-1, 3 * m-2] = K[3 * j-1, 3 * m-2] + k[5, 7]
        K[3 * j-1, 3 * m-1] = K[3 * j-1, 3 * m-1] + k[5, 8]
        K[3 * j-1, 3 * m] = K[3 * j-1, 3 * m] + k[5, 9]
        K[3 * j-1, 3 * n-2] = K[3 * j-1, 3 * n-2] + k[5, 10]
        K[3 * j-1, 3 * n-1] = K[3 * j-1, 3 * n-1] + k[5, 11]
        K[3 * j-1, 3 * n] = K[3 * j-1, 3 * n] + k[5, 12]
        K[3 * j, 3 * i-2] = K[3 * j, 3 * i-2] + k[6, 1]
        K[3 * j, 3 * i-1] = K[3 * j, 3 * i-1] + k[6, 2]
        K[3 * j, 3 * i] = K[3 * j, 3 * i] + k[6, 3]
        K[3 * j, 3 * j-2] = K[3 * j, 3 * j-2] + k[6, 4]
        K[3 * j, 3 * j-1] = K[3 * j, 3 * j-1] + k[6, 5]
        K[3 * j, 3 * j] = K[3 * j, 3 * j] + k[6, 6]
        K[3 * j, 3 * m-2] = K[3 * j, 3 * m-2] + k[6, 7]
        K[3 * j, 3 * m-1] = K[3 * j, 3 * m-1] + k[6, 8]
        K[3 * j, 3 * m] = K[3 * j, 3 * m] + k[6, 9]
        K[3 * j, 3 * n-2] = K[3 * j, 3 * n-2] + k[6, 10]
        K[3 * j, 3 * n-1] = K[3 * j, 3 * n-1] + k[6, 11]
        K[3 * j, 3 * n] = K[3 * j, 3 * n] + k[6, 12]
        K[3 * m-2, 3 * i-2] = K[3 * m-2, 3 * i-2] + k[7, 1]
        K[3 * m-2, 3 * i-1] = K[3 * m-2, 3 * i-1] + k[7, 2]
        K[3 * m-2, 3 * i] = K[3 * m-2, 3 * i] + k[7, 3]
        K[3 * m-2, 3 * j-2] = K[3 * m-2, 3 * j-2] + k[7, 4]
        K[3 * m-2, 3 * j-1] = K[3 * m-2, 3 * j-1] + k[7, 5]
        K[3 * m-2, 3 * j] = K[3 * m-2, 3 * j] + k[7, 6]
        K[3 * m-2, 3 * m-2] = K[3 * m-2, 3 * m-2] + k[7, 7]
        K[3 * m-2, 3 * m-1] = K[3 * m-2, 3 * m-1] + k[7, 8]
        K[3 * m-2, 3 * m] = K[3 * m-2, 3 * m] + k[7, 9]
        K[3 * m-2, 3 * n-2] = K[3 * m-2, 3 * n-2] + k[7, 10]
        K[3 * m-2, 3 * n-1] = K[3 * m-2, 3 * n-1] + k[7, 11]
        K[3 * m-2, 3 * n] = K[3 * m-2, 3 * n] + k[7, 12]
        K[3 * m-1, 3 * i-2] = K[3 * m-1, 3 * i-2] + k[8, 1]
        K[3 * m-1, 3 * i-1] = K[3 * m-1, 3 * i-1] + k[8, 2]
        K[3 * m-1, 3 * i] = K[3 * m-1, 3 * i] + k[8, 3]
        K[3 * m-1, 3 * j-2] = K[3 * m-1, 3 * j-2] + k[8, 4]
        K[3 * m-1, 3 * j-1] = K[3 * m-1, 3 * j-1] + k[8, 5]
        K[3 * m-1, 3 * j] = K[3 * m-1, 3 * j] + k[8, 6]
        K[3 * m-1, 3 * m-2] = K[3 * m-1, 3 * m-2] + k[8, 7]
        K[3 * m-1, 3 * m-1] = K[3 * m-1, 3 * m-1] + k[8, 8]
        K[3 * m-1, 3 * m] = K[3 * m-1, 3 * m] + k[8, 9]
        K[3 * m-1, 3 * n-2] = K[3 * m-1, 3 * n-2] + k[8, 10]
        K[3 * m-1, 3 * n-1] = K[3 * m-1, 3 * n-1] + k[8, 11]
        K[3 * m-1, 3 * n] = K[3 * m-1, 3 * n] + k[8, 12]
        K[3 * m, 3 * i-2] = K[3 * m, 3 * i-2] + k[9, 1]
        K[3 * m, 3 * i-1] = K[3 * m, 3 * i-1] + k[9, 2]
        K[3 * m, 3 * i] = K[3 * m, 3 * i] + k[9, 3]
        K[3 * m, 3 * j-2] = K[3 * m, 3 * j-2] + k[9, 4]
        K[3 * m, 3 * j-1] = K[3 * m, 3 * j-1] + k[9, 5]
        K[3 * m, 3 * j] = K[3 * m, 3 * j] + k[9, 6]
        K[3 * m, 3 * m-2] = K[3 * m, 3 * m-2] + k[9, 7]
        K[3 * m, 3 * m-1] = K[3 * m, 3 * m-1] + k[9, 8]
        K[3 * m, 3 * m] = K[3 * m, 3 * m] + k[9, 9]
        K[3 * m, 3 * n-2] = K[3 * m, 3 * n-2] + k[9, 10]
        K[3 * m, 3 * n-1] = K[3 * m, 3 * n-1] + k[9, 11]
        K[3 * m, 3 * n] = K[3 * m, 3 * n] + k[9, 12]
        K[3 * n-2, 3 * i-2] = K[3 * n-2, 3 * i-2] + k[10, 1]
        K[3 * n-2, 3 * i-1] = K[3 * n-2, 3 * i-1] + k[10, 2]
        K[3 * n-2, 3 * i] = K[3 * n-2, 3 * i] + k[10, 3]
        K[3 * n-2, 3 * j-2] = K[3 * n-2, 3 * j-2] + k[10, 4]
        K[3 * n-2, 3 * j-1] = K[3 * n-2, 3 * j-1] + k[10, 5]
        K[3 * n-2, 3 * j] = K[3 * n-2, 3 * j] + k[10, 6]
        K[3 * n-2, 3 * m-2] = K[3 * n-2, 3 * m-2] + k[10, 7]
        K[3 * n-2, 3 * m-1] = K[3 * n-2, 3 * m-1] + k[10, 8]
        K[3 * n-2, 3 * m] = K[3 * n-2, 3 * m] + k[10, 9]
        K[3 * n-2, 3 * n-2] = K[3 * n-2, 3 * n-2] + k[10, 10]
        K[3 * n-2, 3 * n-1] = K[3 * n-2, 3 * n-1] + k[10, 11]
        K[3 * n-2, 3 * n] = K[3 * n-2, 3 * n] + k[10, 12]
        K[3 * n-1, 3 * i-2] = K[3 * n-1, 3 * i-2] + k[11, 1]
        K[3 * n-1, 3 * i-1] = K[3 * n-1, 3 * i-1] + k[11, 2]
        K[3 * n-1, 3 * i] = K[3 * n-1, 3 * i] + k[11, 3]
        K[3 * n-1, 3 * j-2] = K[3 * n-1, 3 * j-2] + k[11, 4]
        K[3 * n-1, 3 * j-1] = K[3 * n-1, 3 * j-1] + k[11, 5]
        K[3 * n-1, 3 * j] = K[3 * n-1, 3 * j] + k[11, 6]
        K[3 * n-1, 3 * m-2] = K[3 * n-1, 3 * m-2] + k[11, 7]
        K[3 * n-1, 3 * m-1] = K[3 * n-1, 3 * m-1] + k[11, 8]
        K[3 * n-1, 3 * m] = K[3 * n-1, 3 * m] + k[11, 9]
        K[3 * n-1, 3 * n-2] = K[3 * n-1, 3 * n-2] + k[11, 10]
        K[3 * n-1, 3 * n-1] = K[3 * n-1, 3 * n-1] + k[11, 11]
        K[3 * n-1, 3 * n] = K[3 * n-1, 3 * n] + k[11, 12]
        K[3 * n, 3 * i-2] = K[3 * n, 3 * i-2] + k[12, 1]
        K[3 * n, 3 * i-1] = K[3 * n, 3 * i-1] + k[12, 2]
        K[3 * n, 3 * i] = K[3 * n, 3 * i] + k[12, 3]
        K[3 * n, 3 * j-2] = K[3 * n, 3 * j-2] + k[12, 4]
        K[3 * n, 3 * j-1] = K[3 * n, 3 * j-1] + k[12, 5]
        K[3 * n, 3 * j] = K[3 * n, 3 * j] + k[12, 6]
        K[3 * n, 3 * m-2] = K[3 * n, 3 * m-2] + k[12, 7]
        K[3 * n, 3 * m-1] = K[3 * n, 3 * m-1] + k[12, 8]
        K[3 * n, 3 * m] = K[3 * n, 3 * m] + k[12, 9]
        K[3 * n, 3 * n-2] = K[3 * n, 3 * n-2] + k[12, 10]
        K[3 * n, 3 * n-1] = K[3 * n, 3 * n-1] + k[12, 11]
        K[3 * n, 3 * n] = K[3 * n, 3 * n] + k[12, 12]
        return K
