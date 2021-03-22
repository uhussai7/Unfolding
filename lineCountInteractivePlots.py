import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib.widgets import Slider
from matplotlib import cm


#This file makes interactive plots with 3 sliders, the u_r(drt) value is shown with box size.

#load all sensitivities and specificities
def load_all_sens_spec(base):
    Nthres=31
    scale=50
    res=np.linspace(1.7,1.00,16)
    drt=np.linspace(0.1,0.3,16)
    ang_thr=np.linspace(20,90,16)
    w=np.linspace(0,0.99,4)
    thres=np.linspace(0,60,Nthres)

    tang_linecount_n=np.zeros([len(res),len(drt),len(ang_thr)])
    tang_linecount_u=np.zeros([len(res),len(drt),len(ang_thr)])

    rad_linecount_n = np.zeros([len(res), len(drt), len(ang_thr)])
    rad_linecount_u = np.zeros([len(res), len(drt), len(ang_thr)])

    for i_r in range(0,len(res)):
        for i_d in range(0, len(drt)):
            for i_th in range(0, len(ang_thr)):
                #for i_tt in range(0,len(thres)): #this is in the file


                base1='/home/uzair/PycharmProjects/Unfolding/data/radTangCountAlignmentLambdaStep-k-60_Fine/'
                filename = "tang_" + str(int(res[i_r] * 1000)) + "um_drt-" + str(int(drt[i_d] * 100)) + "_w-" + \
                           str(int(round(w[-1], 2) * 100)) + "_angthr-" + str(int(ang_thr[i_th])) + ".npy"
                temp = np.load(base1 + filename)
                tang_linecount_n[i_r, i_d, i_th]=temp[0]
                tang_linecount_u[i_r, i_d, i_th] = temp[1]

                filename = "rad_" + str(int(res[i_r] * 1000)) + "um_drt-" + str(int(drt[i_d] * 100)) + "_w-" + \
                           str(int(round(w[-1], 2) * 100)) + "_angthr-" + str(int(ang_thr[i_th])) + ".npy"
                temp = np.load(base1 + filename)
                rad_linecount_n[i_r, i_d, i_th] = temp[0]
                rad_linecount_u[i_r, i_d, i_th] = temp[1]

    return tang_linecount_n, tang_linecount_u, rad_linecount_n, rad_linecount_u


base='/home/uzair/PycharmProjects/Unfolding/data/rocsAlignmentLambdaStep-k-60_Fine/'
tang_linecount_n, tang_linecount_u, rad_linecount_n, rad_linecount_u = load_all_sens_spec(base)


##make some scatter plots
def plot_scatter(x,y,ax,i_th):
    c=2
    dc=20
    cc=dc*(16)+c
    boxsize=np.arange(c,cc,dc) #large boxes are low res
    marker="s"
    alpha=0.1

    #all u_r
    #for iu in range(0,15):
    ax.scatter(x[ :,i_th, :].flatten(), y[:, i_th, :].flatten(), boxsize,
                marker=marker,alpha=alpha)


#params
Nthres = 31
scale = 50
res = np.linspace(1.7, 1.00, 16)
drt = np.linspace(0.1, 0.3, 16)
ang_thr = np.linspace(20, 90, 16)
w = np.linspace(0, 0.99, 4)
thres = np.linspace(0, 60, Nthres)
beta = np.linspace(0, np.pi / 4, 5)


sliders=[]
#scatter plotter
def plot_scatter_interactive(x,y,xlim,ylim,fig,ax):
    # fig,ax=plt.subplots()
    # plt.subplots_adjust(left=0.25, bottom=0.25)


    axcolor = 'lightgoldenrodyellow'
    ax_ang_thr = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_thres = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_res = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    s_ang_thr=Slider(ax_ang_thr,'Angle Thres.',0,15,valinit=8,valstep=1)
    s_thres=Slider(ax_thres,'Streamline Thres.',0,len(thres),valinit=5,valstep=1)
    s_res=Slider(ax_res,'Resolution',0,15,valinit=7,valstep=1)

    sliders.append(s_ang_thr)
    sliders.append(s_thres)
    sliders.append(s_res)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plot_scatter(x, y, ax, 0, 2,0)
    ax.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]])



    def update(val):
        #i_th=np.where(ang_thr== s_ang_thr.val)
        #i_thres=np.where(thres==s_thres.val)
        #i_drt = np.where(drt == s_drt.val)

        i_th=int(s_ang_thr.val)
        i_thres=int(s_thres.val)
        i_drt = int(s_res.val)

        print(s_ang_thr.val,s_thres.val,s_res.val)
        title=ax.title._text
        xlabel=ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.cla()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plot_scatter(x, y, ax,i_th,i_thres,i_drt)
        ax.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]])

    s_ang_thr.on_changed(update)
    s_thres.on_changed(update)
    s_res.on_changed(update)

    plt.show()
