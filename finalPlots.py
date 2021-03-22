import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib.widgets import Slider
from matplotlib import cm

#load all sensitivities and specificities
def load_all_sens_spec(base):
    Nthres=31
    scale=50
    res=np.linspace(1.7,1.00,16)
    drt=np.linspace(0.1,0.3,16)
    ang_thr=np.linspace(20,90,16)
    w=np.linspace(0,0.99,4)
    thres=np.linspace(0,60,Nthres)
    beta=np.linspace(0,np.pi/4,5)

    n_sens_all=np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])
    u_sens_all=np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])

    n_spec_all=np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])
    u_spec_all=np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])

    tau_n = np.zeros([len(res), len(drt)-1, len(ang_thr),len(thres)])
    tau_u = np.zeros([len(res), len(drt) - 1, len(ang_thr),len(thres)])

    auc_n = np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])
    auc_u = np.zeros([len(res),len(drt)-1,len(ang_thr),len(thres)])

    for i_r in range(0,len(res)):
        for i_d in range(0, len(drt)-1):
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

                tempJ=n_sens_all[i_r,i_d,i_th,:]+n_spec_all[i_r,i_d,i_th,:]-1
                tau_n[i_r, i_d, i_th,:] = thres[tempJ.argmax()]
                tempJ = u_sens_all[i_r, i_d, i_th, :] + u_spec_all[i_r, i_d, i_th, :] - 1
                tau_u[i_r, i_d, i_th,:] = thres[tempJ.argmax()]

                auc_n[i_r, i_d, i_th,:] = auc(1-n_spec_all[i_r, i_d, i_th,:],n_sens_all[i_r, i_d, i_th,:])
                auc_u[i_r, i_d, i_th,:] = auc(1 - u_spec_all[i_r, i_d, i_th, :],u_sens_all[i_r, i_d, i_th, :] )

    n_Y_all = n_sens_all + n_spec_all - 1
    u_Y_all = u_sens_all + u_spec_all - 1

    return n_sens_all, u_sens_all, n_spec_all, u_spec_all, n_Y_all, u_Y_all, tau_n,tau_u, auc_n, auc_u

base='/home/uzair/PycharmProjects/Unfolding/data/rocsAlignmentLambdaStep-k-60_Fine/'
n_sens_all, u_sens_all, n_spec_all, u_spec_all, n_Y_all, u_Y_all,tau_n,tau_u, auc_n, auc_u = load_all_sens_spec(base)


##make some scatter plots
def plot_scatter(x,y,ax,i_th,i_thres):
    scale=35
    c=200/scale
    dc=500/scale
    cc=dc*16+c
    boxsize=np.arange(cc,c,-dc) #large boxes are low res
    marker="s"
    alpha=0.5

    colors=cm.get_cmap('Blues')
    #all u_r
    for iu in range(0,15):
        ax.scatter(x[:, iu, i_th, i_thres].flatten(), y[:, iu, i_th, i_thres].flatten(), boxsize, cmap=colors,
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


def plot_all_youden(y,ax,thres,color):
    y=np.asarray( y.reshape([-1,len(thres)]))
    for i in range(0,y.shape[0]):
        #print(i)
        ax.plot(thres,y[i,:],color=color,alpha=0.2)

#params
Nthres=31
scale=50
res=np.linspace(0.4,1.4,16)
drt=np.linspace(0.1,0.3,16)
ang_thr=np.linspace(20,90,16)
w=np.linspace(0,0.99,4)
thres=np.linspace(0,60,Nthres)
beta=np.linspace(0,np.pi/4,5)


sliders=[]
#scatter plotter
def plot_scatter_interactive(x,y,xlim,ylim,fig,ax):
    # fig,ax=plt.subplots()
    # plt.subplots_adjust(left=0.25, bottom=0.25)


    axcolor = 'lightgoldenrodyellow'
    ax_ang_thr = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_thres = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    s_ang_thr=Slider(ax_ang_thr,'Angle Thres.',ang_thr.min(),ang_thr.max(),valinit=ang_thr[4],valstep=ang_thr[1]-ang_thr[0])
    s_thres=Slider(ax_thres,'Streamline Thres.',thres.min(),thres.max(),valinit=thres[1],valstep=thres[1]-thres[0])

    sliders.append(s_ang_thr)
    sliders.append(s_thres)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plot_scatter(x, y, ax, 0, 2)
    ax.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]])



    def update(val):
        i_th=np.where(ang_thr== s_ang_thr.val)
        i_thres=np.where(thres==s_thres.val)
        print(i_th,i_thres)
        title=ax.title._text
        xlabel=ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.cla()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plot_scatter(x, y, ax,i_th,i_thres)
        ax.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]])

    s_ang_thr.on_changed(update)
    s_thres.on_changed(update)

    plt.show()



#sensitivity scatter
xlim=[-0.1,1.1]
fig,ax=plt.subplots()
ax.set_title('Sensitivity')
ax.set_xlabel('Cartesian')
ax.set_ylabel('Conformal')
plt.subplots_adjust(left=0.3, bottom=0.3)
plot_scatter_interactive(n_sens_all,u_sens_all,xlim,xlim,fig,ax)

#specitifivity scatter
xlim=[0.6,1.1]
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




# ## load the rocs and think of a plot for them
# import numpy as np
# import matplotlib.pyplot as plt
# from tslearn.clustering import TimeSeriesKMeans
#
# #---------------------threshold class plots with beta------------------------#
# Nthres=20
# scale=50
# res=np.linspace(1.7,1.00,8)
# drt=np.linspace(0.1,0.3,5)
# ang_thr=np.linspace(20,90,8)
# w=np.linspace(0,0.99,4)
# thres=np.linspace(0,150,Nthres)
# beta=np.linspace(0,np.pi/4,5)
#
#
# uu=0.3
# u=0
# vv=np.pi / 6
# v=-np.pi / 6
#
# i_io=0
# j_io=2
# l_io=-1
#
# #spec_roc=np.zeros([Nthres,8,2])
# #sens_roc=np.zeros([Nthres,8,2])
#
# opacity=np.linspace(0,0.5,len(ang_thr))
# opacity[:]=0.1
# nYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),Nthres-1,1])
# uYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),Nthres-1,1])
# indexing=np.zeros([len(res),len(drt)-1,len(ang_thr)])
# pp=0
# sens=np.zeros([len(res),len(drt),len(ang_thr),len(thres),2])
# spec=np.zeros([len(res),len(drt),len(ang_thr),len(thres),2])
# b=0
# for i_io in range(0,len(res)):
#     for j_io in range(0,len(drt)-1):
#         for k_io in range(0,len(ang_thr)):
#             base = "/home/uzair/PycharmProjects/Unfolding/data/rpcsAlignmentLambdaStep/"
#             sens_roc = np.load(base +'sens_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-"+ \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#             spec_roc = np.load(base +'spec_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#
#             ncolor = 'blue'
#             ucolor = 'orange'
#             #if j_io<2:
#             #    ncolor='black'
#             #    ucolor='red'
#             sens[i_io,j_io,k_io,:,:]= sens_roc[:,b,:]
#             spec[i_io, j_io, k_io, :, :] = spec_roc[:,b, :]
#
#             nJ=sens_roc[1:, b, 0]- 1 + spec_roc[1:,b,0]
#             uJ=sens_roc[1:, b, 1]- 1 + spec_roc[1:,b,1]
#             #nJ =spec_roc[1:, k_io, 0]
#             #uJ =spec_roc[1:, k_io, 1]
#             x=thres[1:]
#             nYouden[pp, :, 0]  =  nJ
#             uYouden[pp, :, 0]  =  uJ
#
#             plt.plot(1 - spec_roc[:,b,0], sens_roc[:, b, 0],linestyle='-', marker='o', color=ncolor,\
#                                                                                                    alpha=opacity[k_io])
#             plt.plot(1 - spec_roc[:, b, 1], sens_roc[:, b, 1],linestyle='-', marker='o',color=ucolor,
#                      alpha=opacity[k_io])
#             #plt.plot(x,nJ, linestyle='-', marker=None, color=ncolor, alpha=opacity[k_io])
#             #plt.plot(x,uJ, linestyle='-', marker=None, color=ucolor, alpha=opacity[k_io])
#             indexing[i_io,j_io,k_io]=pp
#             pp=pp+1
#
# Nlabels=3
# nmodel = TimeSeriesKMeans(n_clusters=Nlabels,metric='softdtw',max_iter=500)
# umodel = TimeSeriesKMeans(n_clusters=Nlabels,metric='softdtw',max_iter=500)
# nmodel.fit(nYouden)
# umodel.fit(uYouden)
#
#
# plt.figure()
# ncolor = ['red','green','blue']
# ucolor = ['blue','green','red']
# #ucolor = ['blue','red','green']
# for i in range(0,nYouden.shape[0]):
#     plt.plot(thres[1:],nYouden[i,:,0],alpha=0.1,color=ncolor[nmodel.labels_[i]])
#
# plt.figure()
# for i in range(0,uYouden.shape[0]):
#     plt.plot(thres[1:],uYouden[i,:,0],alpha=0.1,color=ucolor[umodel.labels_[i]])
#
# #class based roc curves
# #set plot formatting
# import matplotlib
# matplotlib.rc('xtick', labelsize=10)
# matplotlib.rc('ytick', labelsize=10)
# font = {'family' : 'sans-serif',
#         'weight' : 'normal',
#         'size'   : 'normal'}
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
#
# red_patch = mpatches.Patch(color='red', label='Class 1')
# green_patch = mpatches.Patch(color='green', label='Class 2')
# blue_patch = mpatches.Patch(color='blue', label='Class 3')
#
# sens_line = mlines.Line2D([], [],linestyle='--', color='black',label='Sensitivity')
# spec_line = mlines.Line2D([], [], color='black',label='Specificity')
# youden_line = mlines.Line2D([], [],linestyle= ':', color='black',label='Youden')
#
# sens_c=sens[:,:-1,:,1:,:]
# sens_c=sens_c.reshape(-1,19,2,order='F')
# spec_c=spec[:,:-1,:,1:,:]
# spec_c=spec_c.reshape(-1,19,2,order='F')
# fig,ax=plt.subplots(1,2)
# width=0.085
# height=2.5*width
# delta=0.1
# left=0.55-width-delta
# bottom=0.52
# ax2=fig.add_axes([left,bottom,width,height])
# ax3=fig.add_axes([left+0.52,bottom,width,height])
# nsensspec=np.zeros([2,3,3])
# usensspec=np.zeros([2,3,3])
# tau=2
# for ll in range(0,3):
#     labels=np.where(nmodel.labels_==ll)
#
#     nsens_m=  sens_c[labels,:,0].mean(axis=1)
#     nspec_m = spec_c[labels, :, 0].mean(axis=1)
#     nyouden_m= nYouden[labels,:,0].mean(axis=1)
#     nsensspec[0, ll, 0]= nsens_m[0][tau]
#     nsensspec[1, ll, 0] = nspec_m[0][tau]
#
#
#     nsens_s = sens_c[labels, :, 0].std(axis=1)
#     nspec_s = spec_c[labels, :, 0].std(axis=1)
#     nyouden_s = nYouden[labels, :, 0].std(axis=1)
#     nsensspec[0, ll, 1] = nsens_s[0][tau]
#     nsensspec[1, ll, 1] = nspec_s[0][tau]
#     nsensspec[0, ll, 2] = nsens_s[0][tau]/np.sqrt(sens_c[labels, :, 0].shape[1])
#     nsensspec[1, ll, 2] = nspec_s[0][tau]/np.sqrt(spec_c[labels, :, 0].shape[1])
#
#     labels = np.where(umodel.labels_ == ll)
#     usens_m = sens_c[labels, :, 1].mean(axis=1)
#     uspec_m = spec_c[labels, :, 1].mean(axis=1)
#     uyouden_m = uYouden[labels, :, 0].mean(axis=1)
#     usensspec[0, ll, 0] = usens_m[0][tau]
#     usensspec[1, ll, 0] = uspec_m[0][tau]
#
#     usens_s = sens_c[labels, :, 1].std(axis=1)
#     uspec_s = spec_c[labels, :, 1].std(axis=1)
#     uyouden_s = uYouden[labels, :, 0].std(axis=1)
#     usensspec[0, ll, 1] = usens_s[0][tau]
#     usensspec[1, ll, 1] = uspec_s[0][tau]
#     usensspec[0, ll, 2] = usens_s[0][tau] / np.sqrt(sens_c[labels, :, 1].shape[1])
#     usensspec[1, ll, 2] = uspec_s[0][tau] / np.sqrt(spec_c[labels, :, 1].shape[1])
#
#     #plt.subplot(121)
#     ax[0].set_ylim([0,1.05])
#     ax[0].set_xlabel(r'Streamline Threshold ($\tau$)')
#     ax[0].set_title('Cartesian Coordinates')
#     ax[0].grid(False)
#     ax[0].legend(handles=[red_patch,green_patch,blue_patch,spec_line,sens_line,youden_line],
#                loc='upper right', bbox_to_anchor=(0.95, 0.935),ncol=2)
#     ax[0].plot(thres[1:],nsens_m[0,:],'--',color=ncolor[ll])
#     ax[0].plot(thres[1:],nspec_m[0,:],color=ncolor[ll])
#     ax[0].plot(thres[1:], nyouden_m[0, :], ':', color=ncolor[ll])
#     ax2.axis('equal')
#     ax2.set_title('ROC')
#     ax2.set_xlim([-0.0,0.25])
#     ax2.set_ylim([0, 1])
#     ax2.tick_params(labelsize=8.5,direction='in')
#     #ax2.plot([0, 0.25], [0, 0.25] ,color='grey',alpha=0.3)
#     ax2.plot(1 - nspec_m[0, :], nsens_m[0, :],color=ncolor[ll])
#
#
#     #plt.subplot(122)
#     ax[1].set_ylim([0, 1.05])
#     ax[1].set_xlabel(r'Streamline Threshold ($\tau$)')
#     ax[1].set_title('Conformal Coordinates')
#     ax[1].grid(False)
#     ax[1].legend(handles=[red_patch,green_patch,blue_patch,spec_line,sens_line,youden_line],
#                loc='upper right', bbox_to_anchor=(0.95, 0.935),ncol=2)
#     ax[1].plot(thres[1:],usens_m[0,:],'--',color=ucolor[ll])
#     ax[1].plot(thres[1:],uspec_m[0,:],color=ucolor[ll])
#     ax[1].plot(thres[1:], uyouden_m[0, :], ':', color=ucolor[ll])
#     ax3.axis('equal')
#     ax3.set_title('ROC')
#     ax3.set_xlim([-0.0,0.25])
#     ax3.set_ylim([0, 1])
#     ax3.tick_params(labelsize=8.5,direction='in')
#     #ax3.plot([0, 0.25], [0, 0.25], color='grey',alpha=0.3)
#     ax3.plot(1 - uspec_m[0, :], usens_m[0, :],color=ucolor[ll])
#
#     # plt.figure()
#
# #get the inds from labels
# n_label_subs=[]
# u_label_subs=[]
# for l in range(0,Nlabels):
#     n_label_subs.append(np.unravel_index(np.where(nmodel.labels_==l)[0],indexing.shape))
#     u_label_subs.append(np.unravel_index(np.where(umodel.labels_ == l)[0],indexing.shape))
#
# label=2
# plt.figure()
# for ii in range(0,3):
#     plt.hist(n_label_subs[label][ii],8,alpha=0.1)
#
#
# #drt is parameter with lest variance get average values for all three classes for all three parameters
# nmeanstd=np.zeros([3,3,3])
# for ll in range(0,Nlabels):
#     nmeanstd[0,ll, 0] = res[n_label_subs[ll][0]].mean()
#     nmeanstd[0, ll, 1] = res[n_label_subs[ll][0]].std()
#     nmeanstd[0, ll, 2] = res[n_label_subs[ll][0]].std()/np.sqrt(len(res[n_label_subs[label][0]]))
#
#     nmeanstd[1, ll, 0] = drt[n_label_subs[ll][1]].mean()
#     nmeanstd[1, ll, 1] = drt[n_label_subs[ll][1]].std()
#     nmeanstd[1, ll, 2] = drt[n_label_subs[ll][1]].std() /np.sqrt(len(drt[n_label_subs[label][1]]))
#
#     nmeanstd[2,ll, 0] = ang_thr[n_label_subs[ll][2]].mean()
#     nmeanstd[2, ll, 1] = ang_thr[n_label_subs[ll][2]].std()
#     nmeanstd[2, ll, 2] = ang_thr[n_label_subs[ll][2]].std()/np.sqrt(len(drt[n_label_subs[label][1]]))
#
#
# for label in range(0,3):
#     plt.figure()
#     for ii in range(0,3):
#         plt.hist(u_label_subs[label][ii],8,alpha=0.1)
#
#
# umeanstd=np.zeros([3,3,3])
# for ll in range(0,Nlabels):
#     umeanstd[0,ll, 0] = res[u_label_subs[ll][0]].mean()
#     umeanstd[0, ll, 1] = res[u_label_subs[ll][0]].std()
#     umeanstd[0, ll, 2] = res[u_label_subs[ll][0]].std() / np.sqrt(len(res[u_label_subs[label][0]]))
#
#     umeanstd[1, ll, 0] = drt[u_label_subs[ll][1]].mean()
#     umeanstd[1, ll, 1] = drt[u_label_subs[ll][1]].std()
#     umeanstd[1, ll, 2] = drt[u_label_subs[ll][1]].std() / np.sqrt(len(drt[u_label_subs[label][1]]))
#
#     umeanstd[2,ll, 0] = ang_thr[u_label_subs[ll][2]].mean()
#     umeanstd[2, ll, 1] = ang_thr[u_label_subs[ll][2]].std()
#     umeanstd[2, ll, 2] = ang_thr[u_label_subs[ll][2]].std()/np.sqrt(len(ang_thr[u_label_subs[label][2]]))
#
#
#
#
# #----------------------------------no beta------------------------#
# Nthres=20
# scale=50
# res=np.linspace(1.7,1.00,8)
# drt=np.linspace(0.1,0.3,5)
# ang_thr=np.linspace(20,90,8)
# w=np.linspace(0,0.99,4)
# thres=np.linspace(0,150,Nthres)
#
# uu=0.3
# u=0
# vv=np.pi / 6
# v=-np.pi / 6
#
# i_io=0
# j_io=2
# l_io=-1
#
# #spec_roc=np.zeros([Nthres,8,2])
# #sens_roc=np.zeros([Nthres,8,2])
#
# opacity=np.linspace(0,0.5,len(ang_thr))
# opacity[:]=0.1
# nYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),Nthres-1,1])
# uYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),Nthres-1,1])
# indexing=np.zeros([len(res),len(drt)-1,len(ang_thr)])
# pp=0
# sens=np.zeros([len(res),len(drt),len(ang_thr),len(thres),2])
# spec=np.zeros([len(res),len(drt),len(ang_thr),len(thres),2])
# for i_io in range(0,len(res)):
#     for j_io in range(0,len(drt)-1):
#         for k_io in range(0,len(ang_thr)):
#             base = "/home/uzair/PycharmProjects/Unfolding/data/rocs/"
#             sens_roc = np.load(base +'sens_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-"+ \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#             spec_roc = np.load(base +'spec_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#
#             ncolor = 'blue'
#             ucolor = 'orange'
#             #if j_io<2:
#             #    ncolor='black'
#             #    ucolor='red'
#             sens[i_io,j_io,k_io,:,:]= sens_roc[:,k_io,:]
#             spec[i_io, j_io, k_io, :, :] = spec_roc[:, k_io, :]
#
#             nJ=sens_roc[1:, k_io, 0]- 1 + spec_roc[1:,k_io,0]
#             uJ=sens_roc[1:, k_io, 1]- 1 + spec_roc[1:,k_io,1]
#             #nJ =spec_roc[1:, k_io, 0]
#             #uJ =spec_roc[1:, k_io, 1]
#             x=thres[1:]
#             nYouden[pp, :, 0]  =  nJ
#             uYouden[pp, :, 0]  =  uJ
#
#             plt.plot(1 - spec_roc[:,k_io,0], sens_roc[:, k_io, 0],linestyle='-', marker='o', color=ncolor,\
#                                                                                                    alpha=opacity[k_io])
#             plt.plot(1 - spec_roc[:, k_io, 1], sens_roc[:, k_io, 1],linestyle='-', marker='o',color=ucolor,
#                      alpha=opacity[k_io])
#             #plt.plot(x,nJ, linestyle='-', marker=None, color=ncolor, alpha=opacity[k_io])
#             #plt.plot(x,uJ, linestyle='-', marker=None, color=ucolor, alpha=opacity[k_io])
#             indexing[i_io,j_io,k_io]=pp
#             pp=pp+1
#
# Nlabels=3
# nmodel = TimeSeriesKMeans(n_clusters=Nlabels,metric='softdtw',max_iter=500)
# umodel = TimeSeriesKMeans(n_clusters=Nlabels,metric='softdtw',max_iter=500)
# nmodel.fit(nYouden)
# umodel.fit(uYouden)
#
#
# plt.figure()
# ncolor = ['red','green','blue']
# ucolor = ['blue','green','red']
# #ucolor = ['blue','red','green']
# for i in range(0,nYouden.shape[0]):
#     plt.plot(thres[1:],nYouden[i,:,0],alpha=0.1,color=ncolor[nmodel.labels_[i]])
#
# plt.figure()
# for i in range(0,uYouden.shape[0]):
#     plt.plot(thres[1:],uYouden[i,:,0],alpha=0.1,color=ucolor[umodel.labels_[i]])
#
# #class based roc curves
# #set plot formatting
# import matplotlib
# matplotlib.rc('xtick', labelsize=10)
# matplotlib.rc('ytick', labelsize=10)
# font = {'family' : 'sans-serif',
#         'weight' : 'normal',
#         'size'   : 'normal'}
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
#
# red_patch = mpatches.Patch(color='red', label='Class 1')
# green_patch = mpatches.Patch(color='green', label='Class 2')
# blue_patch = mpatches.Patch(color='blue', label='Class 3')
#
# sens_line = mlines.Line2D([], [],linestyle='--', color='black',label='Sensitivity')
# spec_line = mlines.Line2D([], [], color='black',label='Specificity')
# youden_line = mlines.Line2D([], [],linestyle= ':', color='black',label='Youden')
#
# sens_c=sens[:,:-1,:,1:,:]
# sens_c=sens_c.reshape(-1,19,2,order='F')
# spec_c=spec[:,:-1,:,1:,:]
# spec_c=spec_c.reshape(-1,19,2,order='F')
# fig,ax=plt.subplots(1,2)
# width=0.085
# height=2.5*width
# delta=0.1
# left=0.55-width-delta
# bottom=0.52
# ax2=fig.add_axes([left,bottom,width,height])
# ax3=fig.add_axes([left+0.52,bottom,width,height])
# nsensspec=np.zeros([2,3,3])
# usensspec=np.zeros([2,3,3])
# tau=2
# for ll in range(0,3):
#     labels=np.where(nmodel.labels_==ll)
#
#     nsens_m=  sens_c[labels,:,0].mean(axis=1)
#     nspec_m = spec_c[labels, :, 0].mean(axis=1)
#     nyouden_m= nYouden[labels,:,0].mean(axis=1)
#     nsensspec[0, ll, 0]= nsens_m[0][tau]
#     nsensspec[1, ll, 0] = nspec_m[0][tau]
#
#
#     nsens_s = sens_c[labels, :, 0].std(axis=1)
#     nspec_s = spec_c[labels, :, 0].std(axis=1)
#     nyouden_s = nYouden[labels, :, 0].std(axis=1)
#     nsensspec[0, ll, 1] = nsens_s[0][tau]
#     nsensspec[1, ll, 1] = nspec_s[0][tau]
#     nsensspec[0, ll, 2] = nsens_s[0][tau]/np.sqrt(sens_c[labels, :, 0].shape[1])
#     nsensspec[1, ll, 2] = nspec_s[0][tau]/np.sqrt(spec_c[labels, :, 0].shape[1])
#
#     labels = np.where(umodel.labels_ == ll)
#     usens_m = sens_c[labels, :, 1].mean(axis=1)
#     uspec_m = spec_c[labels, :, 1].mean(axis=1)
#     uyouden_m = uYouden[labels, :, 0].mean(axis=1)
#     usensspec[0, ll, 0] = usens_m[0][tau]
#     usensspec[1, ll, 0] = uspec_m[0][tau]
#
#     usens_s = sens_c[labels, :, 1].std(axis=1)
#     uspec_s = spec_c[labels, :, 1].std(axis=1)
#     uyouden_s = uYouden[labels, :, 0].std(axis=1)
#     usensspec[0, ll, 1] = usens_s[0][tau]
#     usensspec[1, ll, 1] = uspec_s[0][tau]
#     usensspec[0, ll, 2] = usens_s[0][tau] / np.sqrt(sens_c[labels, :, 1].shape[1])
#     usensspec[1, ll, 2] = uspec_s[0][tau] / np.sqrt(spec_c[labels, :, 1].shape[1])
#
#     #plt.subplot(121)
#     ax[0].set_ylim([0,1.05])
#     ax[0].set_xlabel(r'Streamline Threshold ($\tau$)')
#     ax[0].set_title('Cartesian Coordinates')
#     ax[0].grid(False)
#     ax[0].legend(handles=[red_patch,green_patch,blue_patch,spec_line,sens_line,youden_line],
#                loc='upper right', bbox_to_anchor=(0.95, 0.935),ncol=2)
#     ax[0].plot(thres[1:],nsens_m[0,:],'--',color=ncolor[ll])
#     ax[0].plot(thres[1:],nspec_m[0,:],color=ncolor[ll])
#     ax[0].plot(thres[1:], nyouden_m[0, :], ':', color=ncolor[ll])
#     ax2.axis('equal')
#     ax2.set_title('ROC')
#     ax2.set_xlim([-0.0,0.25])
#     ax2.set_ylim([0, 1])
#     ax2.tick_params(labelsize=8.5,direction='in')
#     #ax2.plot([0, 0.25], [0, 0.25] ,color='grey',alpha=0.3)
#     ax2.plot(1 - nspec_m[0, :], nsens_m[0, :],color=ncolor[ll])
#
#
#     #plt.subplot(122)
#     ax[1].set_ylim([0, 1.05])
#     ax[1].set_xlabel(r'Streamline Threshold ($\tau$)')
#     ax[1].set_title('Conformal Coordinates')
#     ax[1].grid(False)
#     ax[1].legend(handles=[red_patch,green_patch,blue_patch,spec_line,sens_line,youden_line],
#                loc='upper right', bbox_to_anchor=(0.95, 0.935),ncol=2)
#     ax[1].plot(thres[1:],usens_m[0,:],'--',color=ucolor[ll])
#     ax[1].plot(thres[1:],uspec_m[0,:],color=ucolor[ll])
#     ax[1].plot(thres[1:], uyouden_m[0, :], ':', color=ucolor[ll])
#     ax3.axis('equal')
#     ax3.set_title('ROC')
#     ax3.set_xlim([-0.0,0.25])
#     ax3.set_ylim([0, 1])
#     ax3.tick_params(labelsize=8.5,direction='in')
#     #ax3.plot([0, 0.25], [0, 0.25], color='grey',alpha=0.3)
#     ax3.plot(1 - uspec_m[0, :], usens_m[0, :],color=ucolor[ll])
#
#
#     # plt.figure()
#
#
# #get the inds from labels
# n_label_subs=[]
# u_label_subs=[]
# for l in range(0,Nlabels):
#     n_label_subs.append(np.unravel_index(np.where(nmodel.labels_==l)[0],indexing.shape))
#     u_label_subs.append(np.unravel_index(np.where(umodel.labels_ == l)[0],indexing.shape))
#
# label=2
# plt.figure()
# for ii in range(0,3):
#     plt.hist(n_label_subs[label][ii],8,alpha=0.1)
#
# #drt is parameter with lest variance get average values for all three classes for all three parameters
# nmeanstd=np.zeros([3,3,3])
# for ll in range(0,Nlabels):
#     nmeanstd[0,ll, 0] = res[n_label_subs[ll][0]].mean()
#     nmeanstd[0, ll, 1] = res[n_label_subs[ll][0]].std()
#     nmeanstd[0, ll, 2] = res[n_label_subs[ll][0]].std()/np.sqrt(len(res[n_label_subs[label][0]]))
#
#     nmeanstd[1, ll, 0] = drt[n_label_subs[ll][1]].mean()
#     nmeanstd[1, ll, 1] = drt[n_label_subs[ll][1]].std()
#     nmeanstd[1, ll, 2] = drt[n_label_subs[ll][1]].std() /np.sqrt(len(drt[n_label_subs[label][1]]))
#
#     nmeanstd[2,ll, 0] = ang_thr[n_label_subs[ll][2]].mean()
#     nmeanstd[2, ll, 1] = ang_thr[n_label_subs[ll][2]].std()
#     nmeanstd[2, ll, 2] = ang_thr[n_label_subs[ll][2]].std()/np.sqrt(len(drt[n_label_subs[label][1]]))
#
#
# for label in range(0,3):
#     plt.figure()
#     for ii in range(0,3):
#         plt.hist(u_label_subs[label][ii],8,alpha=0.1)
#
#
# umeanstd=np.zeros([3,3,3])
# for ll in range(0,Nlabels):
#     umeanstd[0,ll, 0] = res[u_label_subs[ll][0]].mean()
#     umeanstd[0, ll, 1] = res[u_label_subs[ll][0]].std()
#     umeanstd[0, ll, 2] = res[u_label_subs[ll][0]].std() / np.sqrt(len(res[u_label_subs[label][0]]))
#
#     umeanstd[1, ll, 0] = drt[u_label_subs[ll][1]].mean()
#     umeanstd[1, ll, 1] = drt[u_label_subs[ll][1]].std()
#     umeanstd[1, ll, 2] = drt[u_label_subs[ll][1]].std() / np.sqrt(len(drt[u_label_subs[label][1]]))
#
#     umeanstd[2,ll, 0] = ang_thr[u_label_subs[ll][2]].mean()
#     umeanstd[2, ll, 1] = ang_thr[u_label_subs[ll][2]].std()
#     umeanstd[2, ll, 2] = ang_thr[u_label_subs[ll][2]].std()/np.sqrt(len(ang_thr[u_label_subs[label][2]]))
#
#
#
# ####----------------------------This analysis needs to be repeated for unaligned data ----------------------###
#
# Nthres=20
# scale=50
# res=np.linspace(1.7,1.00,8)
# drt=np.linspace(0.1,0.3,5)
# ang_thr=np.linspace(20,90,8)
# w=np.linspace(0,0.99,4)
# thres=np.linspace(0,150,Nthres)
#
# uu=0.3
# u=0
# vv=np.pi / 6
# v=-np.pi / 6
#
# i_io=-1
# j_io=3
# k_io=4
# l_io=-1
#
# #spec_roc=np.zeros([Nthres,8,2])
# #sens_roc=np.zeros([Nthres,8,2])
#
# opacity=np.linspace(0,0.5,len(ang_thr))
# opacity[:]=0.1
# indexing=np.zeros([len(res),len(drt)-1,len(ang_thr)])
# beta=np.linspace(0,np.pi/4,5)
# nYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),len(beta),1])
# uYouden=np.zeros([len(res)*(len(drt)-1)*len(ang_thr),len(beta),1])
#
# nallsens=[]
# nallspec=[]
# uallsens=[]
# uallspec=[]
#
#
# sens=np.zeros([len(res),len(drt),len(ang_thr),len(beta),2])
# spec=np.zeros([len(res),len(drt),len(ang_thr),len(beta),2])
# pp=0
# for i_io in range(0,len(res)):
#     for j_io in range(0,len(drt)-1):
#         for k_io in range(0,len(ang_thr)):
#             base = "/home/uzair/PycharmProjects/Unfolding/data/rocsAlignment/"
#             sens_roc = np.load(base +'sens_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-"+ \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#             spec_roc = np.load(base +'spec_'+ str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + ".npy")
#
#             ncolor = 'blue'
#             ucolor = 'orange'
#             #if j_io<2:
#             #    ncolor='black'
#             #    ucolor='red'
#
#             sens[i_io, j_io, k_io, :, :] = sens_roc[:, k_io, :]
#             spec[i_io, j_io, k_io, :, :] = spec_roc[:, k_io, :]
#
#             nallsens.append(sens_roc[:, k_io, 0])
#             nallspec.append(sens_roc[:, k_io, 0])
#             uallsens.append(spec_roc[:, k_io, 1])
#             uallspec.append(spec_roc[:, k_io, 1])
#
#             nJ=sens_roc[:, k_io, 0]#- 1 + spec_roc[:,k_io,0]
#             uJ=sens_roc[:, k_io, 1]#- 1 + spec_roc[:,k_io,1]
#             #nJ =  spec_roc[:,k_io,0]
#             #uJ =  spec_roc[:,k_io,1]
#             x=beta[:]
#             nYouden[pp, :, 0]  =  nJ
#             uYouden[pp, :, 0]  =  uJ
#
#             #plt.plot(1 - spec_roc[:,k_io,0], sens_roc[:, k_io, 0],linestyle='-', marker='o', color=ncolor,\
#             #                                                                                       alpha=opacity[k_io])
#             #plt.plot(1 - spec_roc[:, k_io, 1], sens_roc[:, k_io, 1],linestyle='-', marker='o',color=ucolor,
#             #         alpha=opacity[k_io])
#             plt.plot(x,nJ, linestyle='-', marker=None, color=ncolor, alpha=opacity[k_io])
#             plt.plot(x,uJ, linestyle='-', marker=None, color=ucolor, alpha=opacity[k_io])
#             indexing[i_io,j_io,k_io]=pp
#             pp=pp+1
#
#
# nallsens=np.asarray(nallsens)
# nallspec=np.asarray(nallspec)
# uallsens=np.asarray(uallsens)
# uallspec=np.asarray(uallspec)
#
# plt.figure()
# #plt.ylim([0,1])
# for b in range(0,len(beta)):
#     plt.scatter(b,nYouden[:,b,0].mean(),color='blue')
#     plt.scatter(b,uYouden[:, b, 0].mean(), color='orange')
#
#
#
# p=1
# fig, ax = plt.subplots(8,4)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
# for i in range(0,len(res)):
#     for j in range(0,len(drt)-1):
#         plotname=str((round(res[i],1))) + "mm " +   r"$u_r$= " + str(round(drt[j] * 10,1))
#         ax[i,j].set_title(plotname)
#         if( i==0 and j==0):
#             #ax[i,j].set_ylabel('Sensitivity')
#             ax[i, j].set_ylabel('Specificity')
#             ax[i, j].set_xlabel('Angle Thres.')
#
#         ax[i,j].set_xlim([20,95])
#         ax[i, j].set_ylim([0.5, 1.1])
#         y=spec[i,j,:,0,0]
#         x=ang_thr
#         ax[i,j].plot(x,y,color='blue')
#         y=spec[i,j,:,0,1]
#         ax[i,j].plot(x,y,color='orange')
#
#
# p=1
# fig, ax = plt.subplots(5,4)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
# alph=np.linspace(0.8,0.2,8)
# for kk in range(0,5):
#     for i in range(0,len(res)):
#         for j in range(0,len(drt)-1):
#             #plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
#             title=r'$u_r$='+str(round(drt[j],2))+ r' $\beta=$' + str(round(180*beta[kk]/np.pi,1)) +r'$^\circ$'
#             ax[kk,j].set_title(title)
#             if( i==0 and j==0 and kk==0):
#                 #ax[kk,i].set_ylabel('Sensitivity')
#                 ax[kk,i].set_ylabel('Specitivity')
#                 ax[kk,i].set_xlabel('Angle Thres.')
#
#             ax[kk,j].set_xlim([20,95])
#             ax[kk,j].set_ylim([0.5, 1.1])
#             y=spec[i,j,:,kk,0]
#             x=ang_thr
#             print(kk,i,j)
#             ax[kk,j].plot(x,y,color='blue',alpha=alph[i])
#             y=spec[i,j,:,kk,1]
#             ax[kk,j].plot(x,y,color='orange',alpha=alph[i])
#
#
# #plt.savefig(base+plotname)
# #plt.savefig(base+'alldata.png')
# #plt.close()
#
# # seeds=[]
# # testlines=[]
# # for line in ulines:
# #     seeds.append(line[0])
# #     temp, templine=tt.connectedInds(radtang_nii,2,line[0],const_coord='u')
# #     testlines.append(templine)
# #
# # for ss in range(0,len(seeds)):
# #     seed=seeds[ss]
# #     plt.scatter(seed[0],seed[1])
# #     if(len(testlines[ss])>0):
# #         thisline=np.asarray(testlines[ss])
# #         plt.plot(thisline[:,0],thisline[:,1])
# # plt.axis('equal')
#
# sens_flat=sens.reshape([-1,len(ang_thr),len(beta),2],order='F')
# spec_flat=spec.reshape([-1,len(ang_thr),len(beta),2],order='F')
#
# #plt.figure()
# #plt.ylim([0,1])
# fig,ax=plt.subplots(2,1)
# for b in range(0,len(beta)):
#     ax[0].scatter(np.rad2deg(beta)[b], sens_flat[:, 4, b, 0].mean(), color='blue')
#     ax[0].errorbar(np.rad2deg(beta)[b],sens_flat[:,4,b,0].mean(),
#                  yerr=sens_flat[:,4,b,0].std()/np.sqrt(len(sens_flat[:,4,b,0])),color='blue',capsize=5)
#
#     ax[0].scatter(np.rad2deg(beta)[b], sens_flat[:, 4, b, 1].mean(), color='orange')
#     ax[0].errorbar(np.rad2deg(beta)[b],sens_flat[:,4,b,1].mean(),
#                  yerr=sens_flat[:,4,b,1].std()/np.sqrt(len(sens_flat[:,4,b,1])),color='orange',capsize=5)
#     ax[0].set_title('Sensitivity Vs.'+r'$\beta$')
#     ax[0].set_xlabel(r'$\beta$')
#     ax[0].set_ylabel('Sensitivity')
#     ax[0].text(-4,0.75,'a)')
#
# for b in range(0,len(beta)):
#     ax[1].scatter(np.rad2deg(beta)[b], spec_flat[:, 4, b, 0].mean(), color='blue')
#     ax[1].errorbar(np.rad2deg(beta)[b],spec_flat[:,4,b,0].mean(),
#                  yerr=spec_flat[:,4,b,0].std()/np.sqrt(len(spec_flat[:,4,b,0])),color='blue',capsize=5)
#
#     ax[1].scatter(np.rad2deg(beta)[b], spec_flat[:, 4, b, 1].mean(), color='orange')
#     ax[1].errorbar(np.rad2deg(beta)[b],spec_flat[:,4,b,1].mean(),
#                  yerr=spec_flat[:,4,b,1].std()/np.sqrt(len(spec_flat[:,4,b,1])),color='orange',capsize=5)
#     ax[1].set_title('Specificity Vs.' + r'$\beta$')
#     ax[1].set_xlabel(r'$\beta$')
#     ax[1].set_ylabel('Specificity')
#     ax[1].text(-4,0.88,'b)')
#
#
#
