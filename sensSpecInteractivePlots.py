import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib.widgets import Slider
from matplotlib import cm


#This file makes interactive plots with 3 sliders, the u_r(drt) value is shown with box size.

#load all sensitivities and specificities
def load_all_sens_spec(base):
    Nthres = 31
    scale = 50
    res = np.linspace(0.4, 1.4, 16)
    drt = np.linspace(0.1, 0.3, 16)
    ang_thr = np.linspace(20, 90, 16)
    w = np.linspace(0, 0.99, 4)
    thres = np.linspace(0, 60, Nthres)
    beta = np.linspace(0, np.pi / 4, 5)

    n_sens_all=np.zeros([len(res),len(drt),len(ang_thr),len(thres)])
    u_sens_all=np.zeros([len(res),len(drt),len(ang_thr),len(thres)])

    n_spec_all=np.zeros([len(res),len(drt),len(ang_thr),len(thres)])
    u_spec_all=np.zeros([len(res),len(drt),len(ang_thr),len(thres)])

    tau_n = np.zeros([len(res), len(drt), len(ang_thr),len(thres)])
    tau_u = np.zeros([len(res), len(drt), len(ang_thr),len(thres)])

    auc_n = np.zeros([len(res),len(drt),len(ang_thr),len(thres)])
    auc_u = np.zeros([len(res),len(drt),len(ang_thr),len(thres)])

    tang_linecount_n=np.zeros([len(res),len(drt),len(ang_thr)])
    tang_linecount_u=np.zeros([len(res),len(drt),len(ang_thr)])

    rad_linecount_n = np.zeros([len(res), len(drt), len(ang_thr)])
    rad_linecount_u = np.zeros([len(res), len(drt), len(ang_thr)])

    for i_r in range(0,len(res)):
        for i_d in range(0, len(drt)):
            for i_th in range(0, len(ang_thr)):
                #for i_tt in range(0,len(thres)): #this is in the file
                filename="sens_"+str(int(res[i_r] * 1000)) + "um_drt-" + str(int(drt[i_d] * 100)) + "_w-" + \
                         str(int(round(w[-1], 2) * 100)) + "_angthr-" + str(int(ang_thr[i_th]))+".npy"
                temp=np.load(base+filename)
                n_sens_all[i_r,i_d,i_th,:]=temp[:,0,0]
                u_sens_all[i_r, i_d, i_th, :] = temp[:, 0, 1]

                filename="spec_"+str(int(res[i_r] * 1000)) + "um_drt-" + str(int(drt[i_d] * 100)) + "_w-" + \
                         str(int(round(w[-1], 2) * 100)) + "_angthr-" + str(int(ang_thr[i_th]))+".npy"
                temp=np.load(base+filename)
                n_spec_all[i_r, i_d, i_th, :] = temp[:, 0, 0]
                u_spec_all[i_r, i_d, i_th, :] = temp[:, 0, 1]

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

                tempJ=n_sens_all[i_r,i_d,i_th,:]+n_spec_all[i_r,i_d,i_th,:]-1
                tau_n[i_r, i_d, i_th,:] = thres[tempJ.argmax()]
                tempJ = u_sens_all[i_r, i_d, i_th, :] + u_spec_all[i_r, i_d, i_th, :] - 1
                tau_u[i_r, i_d, i_th,:] = thres[tempJ.argmax()]

                auc_n[i_r, i_d, i_th,:] = auc(1 - n_spec_all[i_r, i_d, i_th,:],n_sens_all[i_r, i_d, i_th,:])
                auc_u[i_r, i_d, i_th,:] = auc(1 - u_spec_all[i_r, i_d, i_th, :],u_sens_all[i_r, i_d, i_th, :])

    n_Y_all = n_sens_all + n_spec_all - 1
    u_Y_all = u_sens_all + u_spec_all - 1

    return n_sens_all, u_sens_all, n_spec_all, u_spec_all, n_Y_all, u_Y_all, tau_n,tau_u, auc_n, auc_u, \
           tang_linecount_n, tang_linecount_u, rad_linecount_n, rad_linecount_u

base='/home/uzair/PycharmProjects/Unfolding/data/rocsAlignmentLambdaStep-k-60_Fine/'
n_sens_all, u_sens_all, n_spec_all, u_spec_all, n_Y_all, u_Y_all,tau_n,tau_u, auc_n, auc_u, tang_linecount_n, \
tang_linecount_u, rad_linecount_n, rad_linecount_u = load_all_sens_spec(base)


##make some scatter plots
def plot_scatter(x,y,ax,i_th,i_thres,i_drt):
    c=2
    dc=20
    cc=dc*(16)+c
    boxsize=np.arange(c,cc,dc) #large boxes are low res
    marker="s"
    alpha=0.1

    #all u_r
    #for iu in range(0,15):
    ax.scatter(x[ i_drt, :, i_th, i_thres], y[i_drt,: , i_th, i_thres], boxsize,
                marker=marker,alpha=alpha)

    #low u_r
    # ax.scatter(x[:, 0, i_th,i_thres].flatten(), y[:, 0, i_th,i_thres].flatten(), boxsize, color='#d00000',marker=marker,
    #            alpha=alpha)
    # ax.scatter(x[:, 1, i_th,i_thres].flatten(), y[:, 1, i_th,i_thres].flatten(), boxsize, color='#e85d04',
    #            marker=marker,alpha=alpha)
    # #high u_r
    # ax.scatter(x[:, 2, i_th,i_thres].flatten(), y[:, 2, i_th,i_thres].flatten(), boxsize, color='#aacc00',
    #            marker=marker,alpha=alpha)
    # ax.scatter(x[:, 3, i_th,i_thres].flatten(), y[:, 3, i_th,i_thres].flatten(), boxsize, color='#008000',marker=marker,
    #            alpha=alpha)




#params
Nthres = 31
scale = 50
res = np.linspace(0.4, 1.4, 16)
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

def plot_all_youden(y,ax,thres,color):
    y=np.asarray( y.reshape([-1,len(thres)]))
    for i in range(0,y.shape[0]):
        #print(i)
        ax.plot(thres,y[i,:],color=color,alpha=0.2)


#sensitivity scatter
xlim=[-0.1,1.1]
fig,ax=plt.subplots()
ax.set_title('Sensitivity')
ax.set_xlabel('Cartesian')
ax.set_ylabel('Conformal')
plt.subplots_adjust(left=0.3, bottom=0.3)
plot_scatter_interactive(n_sens_all,u_sens_all,xlim,xlim,fig,ax)


#specitifivity scatter
xlim=[0.2,1.1]
fig,ax=plt.subplots()
ax.set_title('Specificity')
ax.set_xlabel('Cartesian')
ax.set_ylabel('Conformal')
plt.subplots_adjust(left=0.3, bottom=0.3)
plot_scatter_interactive(n_spec_all,u_spec_all,xlim,xlim,fig,ax)

#youden scatter
xlim=[-0.1,1.1]
fig,ax=plt.subplots()
ax.set_title('Youden')
ax.set_xlabel('Cartesian')
ax.set_ylabel('Conformal')
plt.subplots_adjust(left=0.3, bottom=0.3)
plot_scatter_interactive(n_Y_all,u_Y_all,xlim,xlim,fig,ax)

#youden all
fig,ax=plt.subplots()
plot_all_youden(n_Y_all,ax,thres,'blue')
plot_all_youden(u_Y_all,ax,thres,'orange')

#auc scatter
fig,ax=plt.subplots()
ax.set_title('AUC')
ax.set_xlabel('Cartesian')
ax.set_ylabel('Conformal')
xlim=[0.5,1.1]
plt.subplots_adjust(left=0.3, bottom=0.3)
plot_scatter_interactive(auc_n,auc_u,xlim,xlim,fig,ax)
#
# opacity=np.linspace(0,1,31)
# for i in range(0,31):
#     plt.plot(u_sens_all[:,7,7,:],alpha=opacity[i])


#make plots for line count
fig,ax =plt.subplots()
cols=plt.cm.viridis(ang_thr/ang_thr.max())
for i_ang in range(0,16):#len(ang_thr)):
    opacity=i_ang/len(ang_thr)/10
    ax.scatter(tang_linecount_n[7,i_ang,7].flatten(),tang_linecount_u[7,i_ang,7].flatten(), color=cols[i_ang],
               alpha=1,
               marker='.')
maxval=max(tang_linecount_n[:,:,:].flatten().max(),tang_linecount_u[:,:,:].flatten().max())
ax.plot([0,maxval],[0,maxval],color='orange')
ax.axis('equal')


#make plots for line count
fig,ax =plt.subplots()
cols=plt.cm.viridis(ang_thr/ang_thr.max())
for i_ang in range(0,len(ang_thr)):
    opacity=i_ang/len(ang_thr)/10
    ax.scatter(rad_linecount_n[7,i_ang,7].flatten(),rad_linecount_u[7,i_ang,7].flatten(), color=cols[i_ang],
               alpha=1,
               marker='.')
maxval=max(rad_linecount_n[7,:,7].flatten().max(),rad_linecount_u[7,:,7].flatten().max())
ax.plot([0,maxval],[0,maxval],color='orange')
ax.axis('equal')


#some other plots
plt.figure()
plt.plot(n_sens_all[7,7,:,1])
plt.plot(u_sens_all[7,7,:,1])

plt.figure()
plt.plot(n_spec_all[7,7,:,1])
plt.plot(u_spec_all[7,7,:,1])
