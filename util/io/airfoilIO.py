import numpy as np

def loadAirfoil(file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv'):
    info = np.loadtxt(file, delimiter=',', skiprows=1)
    num_nodes = info.__len__()
    info[:, 0] = np.arange(num_nodes)

    turning_point_id = num_nodes // 2

    force_dict = {}
    force_dict['GLOBALIDX'] = np.array(info[:, 0], dtype=int)
    force_dict['X'] = info[:, 1]
    force_dict['Y'] = info[:, 2]
    force_dict['PRESS'] = info[:, 3]
    force_dict['PRESSCO'] = info[:, 4]
    force_dict['MACHNUM'] = info[:, 5]

    airfoil = {}
    airfoil['x_bot'] = force_dict['X'][:turning_point_id]
    airfoil['y_bot'] = force_dict['Y'][:turning_point_id]
    airfoil['p_bot'] = force_dict['PRESS'][:turning_point_id]
    airfoil['x_top']= force_dict['X'][turning_point_id:][::-1]
    airfoil['y_top'] = force_dict['Y'][turning_point_id:][::-1]
    airfoil['p_top'] = force_dict['PRESS'][turning_point_id:][::-1]
    airfoil['num_data'] = num_nodes
    airfoil['turning_point'] = num_nodes//2
    return airfoil