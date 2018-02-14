import numpy as np

def stlTxtToNpArray(path):
    path = r"/Users/gakki/Downloads/SU2_mesh_point_clouds/Optimale_orig_points.txt"
    mesh = np.empty([1,3])
    with open(path) as fp:
        for line in fp:
            if line.__len__()<5:
                continue
            line = line.strip().split(';')
            line = list(map(str.strip, line))
            if line[1].startswith('-'):
                if line[1].count('-') == 2:
                    line[1] = line[1].replace('-','e-')[1:]
            else:
                line[1] = line[1].replace('-','e-')
            mesh = np.vstack([mesh,np.array(line,dtype='float')])
    mesh = mesh[1:,:]
    return mesh



if __name__=='__main__':
    path = r"/Users/gakki/Downloads/SU2_mesh_point_clouds/Optimale_orig_points.txt"
    mesh = stlTxtToNpArray(path=path)
    np.savetxt('new_mesh.txt',mesh,delimiter=';')