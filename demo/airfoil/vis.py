import numpy as np
from util.plot import *
force_file='/Users/gakki/Dropbox/thesis/surface_flow_sort.csv'

info = np.loadtxt(force_file, delimiter=',', skiprows=1)
num_nodes = info.__len__()
info[:, 0] = np.arange(num_nodes)

turning_point_id = num_nodes // 2

xBot = info[:, 1][:turning_point_id]
yBot = info[:, 2][:turning_point_id]
pBot = info[:, 3][:turning_point_id]
xTop = info[:, 1][turning_point_id:][::-1]
yTop = info[:, 2][turning_point_id:][::-1]
pTop = info[:, 3][turning_point_id:][::-1]


plt.figure()
X = xTop[0:5]
Y = yTop[0:5]
plt = twoDPlot(X,Y)
XXX = X[2]
YYY = Y[2]
XX = [X[1],X[3]]
YY = [Y[1],Y[3]]
plt = twoDPlot(X,Y,plotstyle='scatter',label='1st & 5th data points',xlabel='x',ylabel='y')
plt = twoDPlot(XX,YY,plotstyle='scatter', label='2nd & 4th data points',s=100)
plt = twoDPlot(XXX,YYY,plotstyle='scatter', label='3rd data point',c='green',s=300)
plt.plot([X[0],X[2],X[2]],[Y[2],Y[2],Y[0]],linestyle='-',c='green',lw=1,dashes=[2,2])
from demo.equalLengthBeamElement.functionality import computeN_VEC
n_vec = 0.0003*computeN_VEC(X[1],Y[1],X[3],Y[3])
plt.arrow(X[2]-n_vec[0],Y[2]-n_vec[1],n_vec[0],n_vec[1],width=0.00001,length_includes_head=True,label='Pressure')
plt.annotate(s='Pressure = %f' % pTop[2],xy=(X[2]-2*n_vec[0],Y[2]-2*n_vec[1]),rotation=38)
plt.annotate(s='Y = %f'%Y[2],xy=(X[2],(Y[2]+Y[0])/1.8),rotation=-90)
plt.annotate(s='X = %f'%X[2],xy=((X[0]+X[2])/3,Y[2]))

plt.xlim(X[0],X[-1])
plt.ylim(Y[0])

#plt.axis('equal')
finalizePlot(plt,title='An example of the third data point',savefig=True,fname='example3rdDataPoint.eps')