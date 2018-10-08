
P = [1,2,3,4,5,6,7,8,10,12,16]
t = [18.89,11.06,9.64,9.53,9.42,9.25,9.63,9.79,10.1,11.0,13.11]

fd = [0.03,0.25,0.6,2.07,7.16,25.56]
cs = [0.03,0.26,0.74,3.27,15.15,79.04]
direct = [0.04,0.13,0.28,1.66,11.07,77.23]
adjoint = [0.02,0.02,0.04,0.06,0.14,0.45]
N = [10,100,200,500,1000,2000]
from util.plot import *

plt.figure()
plt = twoDPlot(N,fd,label='FD',marker='o')
plt = twoDPlot(N,cs,label='CS',marker='o')
plt = twoDPlot(N,direct,label='direct',marker='o')
plt = twoDPlot(N,adjoint,label='adjoint',marker='o')
plt.xlabel('nx')
plt.ylabel('execution time (in seconds)')
plt.grid('on')
finalizePlot(plt,title='Execution time for computing derivatives when nx>>nf',savefig=True,fname='time_4.eps')

plt.figure()
plt = twoDPlot(P,t,marker='o',c='red')
plt.grid('on')
plt.xlabel('Processors')
plt.ylabel('execution time (in seconds)')
finalizePlot(plt,title='Execution time for paralleled linear solver',savefig=True,fname='time_ad_para.eps')
sp = t[0]/np.array(t)

plt.figure()
plt = twoDPlot(P,sp,marker='o',c='red')
plt.grid('on')
plt.xlabel('Processors')
plt.ylabel('speed up')
finalizePlot(plt,title='Speed up for paralleled linear solver',savefig=True,fname='sp_ad_para.eps')

