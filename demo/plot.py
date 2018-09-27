import numpy as np

FORCE_FILE = "/Users/gakki/Dropbox/thesis/surface_flow_sort.csv"

info = np.loadtxt(FORCE_FILE,delimiter=',',skiprows=1)
num_elements = info.__len__()
num_nodes = info.__len__()


force_dict = {}
force_dict['GLOBALIDX'] = np.array(info[:,0],dtype=int)
force_dict['X'] = info[:,1]
force_dict['Y'] = info[:,2]
force_dict['PRESS'] = info[:,3]
force_dict['PRESSCO'] = info[:,4]
force_dict['MACHNUM'] = info[:,5]



from util.plot import *
plt.figure(2)
X = force_dict['X']
Y = force_dict['Y']
plt = twoDPlot(X,Y,plotstyle='scatter',xlabel='x',ylabel='y',label='',s=1,c='b')
plt.axis('equal')
finalizePlot(plt,title='Shape of NACA64a203',savefig=True,fname='NACA64a203.eps')

