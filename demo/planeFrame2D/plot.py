from util.plot import *
from util.io.airfoilIO import *

airfoil = loadAirfoil()

N = 20

xTop = airfoil['x_top']
yTop = airfoil['y_top']
pTop = airfoil['p_top']
xBot = airfoil['x_bot']
yBot = airfoil['y_bot']
pBot = airfoil['p_bot']
turning_point_id = airfoil['turning_point']

divide_list = {}
divide_list['TOP'] = [0]
divide_list['BOT'] = [0]

NNode = N + 1
L = np.ones(shape=(N,)) / N
xi = np.cumsum(L) - L[0]
xi = np.hstack((xi, 1))

for element_ID in range(N):
    for local_node_ID in range(2):
        x = xi[element_ID + local_node_ID]
    count_bot = min(range(turning_point_id), key=lambda i: abs(xBot[i] - x))
    count_top = min(range(turning_point_id), key=lambda i: abs(xTop[i] - x))
    divide_list['TOP'].append(count_top)
    divide_list['BOT'].append(count_bot)

NNode = N + 1
L = np.ones(shape=(N,)) / N
xi = np.cumsum(L) - L[0]
xi = np.hstack((xi, 1))


Xt = xTop[0:divide_list['TOP'][2]]
Yt = yTop[0:divide_list['TOP'][2]]
Xb = xBot[0:divide_list['BOT'][2]]
Yb = yBot[0:divide_list['BOT'][2]]
print(np.shape(Yt),np.shape(Yb))
print([xTop[i] for i in divide_list['TOP']])
Ypf = (np.array([yTop[i] for i in divide_list['TOP']])+ np.array([yBot[i] for i in divide_list['BOT']]))/2

plt.figure()
plt = twoDPlot(Xt,Yt,plotstyle='scatter',label='top data points',xlabel='x',ylabel='y',s=0.5)
plt = twoDPlot(Xb,Yb,plotstyle='scatter',label='bottom data points',s=0.5)
plt = twoDPlot(xi[0:2],Ypf[0:2],plotstyle='plot',label='the 1st plane frame element',c='green')
plt = twoDPlot(xi[1:3],Ypf[1:3],plotstyle='plot',label='the 2nd plane frame element',c='red')
plt = twoDPlot(xi[0:3],Ypf[0:3],plotstyle='scatter',label='first three plane frame nodes',c='black',marker='x')
plt = twoDPlot([xi[1]]*2,[yBot[divide_list['BOT']][1],yTop[divide_list['TOP']][1]],c='grey',alpha=0.5,linestyle='dashed')
plt = twoDPlot(xi[0:3],np.zeros(3),plotstyle='plot',alpha=0.5,c='green',linestyle='dashed',marker='o')
plt.xlim([0,xi[2]])
plt.ylabel('y')
#finalizePlot(plt)
finalizePlot(plt,title='An example of two plane frame elements',savefig=True,fname='pf_example_element.eps')