import matplotlib.pyplot as plt

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

def finalizePlot(plt,title='',savefig=False,**kwargs):
    plt.title(title)
    if savefig == False:
        plt.show()
        pass
    else:
        plt.savefig(**kwargs)


if __name__ == '__main__':

    plt = oneDPlot([1,2,3],plotstyle='scatter',xlabel='x',ylabel='y', span=1, label='l1', c='r', s=12)
    plt = oneDPlot([1, 2, 5], plotstyle='scatter', span=1, label='l2', c='b', s=122)
    finalizePlot(plt, title='Two dots')