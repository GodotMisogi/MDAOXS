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


#print(d)
import matplotlib.pyplot as plt
"""
# tangent
x = force_dict['X'][-7:]
y = force_dict['Y'][-7:]
plt.scatter(x,y,c='rrbbbrr',s=[10,10,30,100,30,10,10])
plt.plot(x,y)
plt.xlim(min(x),max(x))
plt.savefig(fname='20180905',aspect='auto')


"""


"""
x1 = force_dict['X'][-20:]
y1 = force_dict['Y'][-20:]
x2 = force_dict['X'][:20]
y2 = force_dict['Y'][:20]
x = [x1[0],(x1[0]+x1[-1])/2,x1[-1]]
y = [0,0,0]
plt.scatter(x1,y1,label='top nodes')
plt.scatter(x2,y2,label='bot nodes')
plt.scatter(x,y,label='element nodes')
plt.plot(x1,np.zeros(len(x1)))
plt.vlines(x[1],ymin=y2[10],ymax=y1[10],linestyles='dashed',label='divide line')
plt.xlim(min(x1),max(x1))
plt.legend()
plt.title('Example of Elements')
plt.savefig(fname='exampleOfElements')
"""
x1 = force_dict['X'][-20:]
y1 = force_dict['Y'][-20:]
x2 = force_dict['X'][:20]
y2 = force_dict['Y'][:20]
x = [x1[0],(x1[0]+x1[-1])/2,x1[-1]]
y = [(y1[0]+y2[-1])/2,(y1[10]+y2[10])/2,0]
plt.scatter(x1,y1,label='top nodes')
plt.scatter(x2,y2,label='bot nodes')
plt.scatter(x,y,label='element nodes')
plt.plot(x,y)
plt.vlines(x[1],ymin=y2[10],ymax=y1[10],linestyles='dashed',label='divide line')
plt.xlim(min(x1),max(x1))
plt.legend()
plt.title('Example of Plane Frame Elements')
plt.savefig('planeframeElement')