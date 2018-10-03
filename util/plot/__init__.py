import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
def oneDPlot(x,plotstyle,span,xlabel='',ylabel='',label='',**linestyle):
    x_coord = [xl * span / (len(x) - 1) for xl in list(range(len(x)))]
    if plotstyle == 'scatter':
        plt.scatter(x_coord, x,label=label,**linestyle)
        pass
    elif plotstyle == 'plot':
        plt.plot(x_coord,x,label=label,**linestyle)
        pass
    elif plotstyle == 'bar':
        plt.bar(x_coord,x)
    if len(xlabel) != 0:
        plt.xlabel(xlabel)
    if len(ylabel) != 0:
        plt.ylabel(ylabel)
    if len(label) != 0:
        plt.legend()
    return plt

def twoDPlot(x,y,plotstyle='plot',xlabel='',ylabel='',label='',**linestyle):
    if plotstyle == 'scatter':
        plt.scatter(x,y,label=label,**linestyle)
        pass
    elif plotstyle == 'plot':
        plt.plot(x,y,label=label,**linestyle)
        pass
    if len(xlabel) != 0:
        plt.xlabel(xlabel)
    if len(ylabel) != 0:
        plt.ylabel(ylabel)
    if len(label) != 0:
        plt.legend()
    return plt

def interpolatePlot(x,y,plotstyple='plot',spline='bspline', N=100,**linestyle):
    if spline == 'bspline':
        t, c, k = scipy.interpolate.splrep(x, y,s=0,k=4)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = scipy.interpolate.BSpline(t, c, k, extrapolate=False)
        if plotstyple=='scatter':
            plt.scatter(xx,spline(xx),**linestyle)
        else:
            plt.plot(xx,spline(xx),**linestyle)
    return plt

def finalizePlot(plt,title='',savefig=False,**kwargs):
    plt.title(title)
    if savefig == False:
        plt.show()
        pass
    else:
        plt.savefig(**kwargs)



if __name__ == '__main__':
    x = np.array([1.,2,3,4,5,1.1])
    y = np.array([1.,4,2,3,6,1.1])
    #x = np.array([0., 1.2, 1.9, 3.2, 4., 6.5])
    #y = np.array([0., 2.3, 3., 4.3, 2.9, 3.1])
    plt.figure()
    plt = interpolatePlot(x,y,plotstyple='scatter',N=100)
    plt.show()