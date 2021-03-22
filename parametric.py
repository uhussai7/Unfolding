import numpy as np
import seaborn as ss
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
import matplotlib as mpl
import pandas as pd

Nthres=1
scale=1
res=np.linspace(0.02,0.12,16)
drt=[0.5*(1/1.75 - 0.07)]#np.linspace(0.1,0.3,16)
ang_thr=np.linspace(20,90,16)
w=np.linspace(1.0,1.99,16)
beta=np.linspace(0,np.pi/4,5)
#thres=np.linspace(0,10,Nthres)
thres=[1]




sens = np.zeros([16,16,16,Nthres,2])
spec = np.zeros([16,16,16,Nthres,2])
rad = np.zeros([16,16,16,2])
tang = np.zeros([16,16,16,2])
rad_para = np.zeros([16,16,16,2])
tang_para = np.zeros([16,16,16,2])


base='data/simulations_sens_spec_curv_tangOnly/'
for i in range(0,16):
    for j in range(0,16):
        for k in range(0,16):
            rs=str(int(res[i]*1000))
            drs='25'
            ws=str(int(round(w[j],2)*1000))
            angs=str(int(ang_thr[k]))
            #print('rad_'+rs+'um_drt-'+drs+'_w-'+ws+'_angthr-'+angs+'.npy')
            rad[i, j, k] = np.load(base+'rad_'+rs+'um_drt-'+drs+'_w-'+ws+'_angthr-'+angs+'.npy')
            tang[i, j, k] = np.load(base+'tang_' + rs + 'um_drt-' + drs + '_w-'+ws+'_angthr-' + angs + '.npy')
            rad_para[i, j, k] = np.load(base + 'radPara_' + rs + 'um_drt-' + drs + '_w-'+ws+'_angthr-' + angs + '.npy')
            tang_para[i, j, k] = np.load(base + 'tangPara_' + rs + 'um_drt-' + drs + '_w-'+ws+'_angthr-' + angs +
                                         '.npy')
            #try:
            sens[i,j,k,:,:]= np.load(base + 'sens_' + rs + 'um_drt-' + drs + '_w-'+ws+'_angthr-' + angs + '.npy')
            spec[i, j, k, :, :] = np.load(base + 'spec_' + rs + 'um_drt-' + drs + '_w-'+ws+'_angthr-' + angs + '.npy')
            #except:
            #    pass
#plt.scatter(spec[:,-2,:,1,0].flatten(),spec[:,-2,:,1,1].flatten())


total=rad_para+tang_para
rad_paraa=rad_para/total
tang_paraa=tang_para/total

# n=[]
# u=[]
nsens=[]
usens=[]
res_=[]
ang_=[]
w_=[]
for i in range(0,16,1):
    for j in range(0,16,1):
        print(j)
        for k in range(8, 9,1):
            nsens.append(spec[i,j,k,0,0])#+spec[i,j,k,0,0]-1)
            usens.append(spec[i,j,k,0,1])#+spec[i,j,k,0,1]-1)
            res_.append(i)
            ang_.append(ang_thr[k])
            w_.append(j)#w[j])

nsens=np.asarray(nsens)
usens=np.asarray( usens)
res_=np.asarray( res_)
ang_=np.asarray( ang_)

df=pd.DataFrame({
        'Cartesian': nsens,
        'Conformal': usens,
        'Resolution': res_,
        'Angle Threshold': ang_,
        'Curvature': w_,
    })

colors='flare'
norm = plt.Normalize(df['Angle Threshold'].min(), df['Angle Threshold'].max())
sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
sm.set_array([])

#ss.set_theme()
ss.set(rc={'figure.figsize':(15.5,12)})
ss.set_theme(font_scale=2.8)
ax = ss.scatterplot(x='Cartesian', y='Conformal', data=df, marker='o', palette=colors, hue='Curvature',
                    size='Resolution',alpha=0.5,sizes=(30, 300),label='big')
                    #alpha=0.6,sizes=(40, 400),label='big')
# ss.scatterplot(x='longitude', y='latitude', data=dg, marker='^',
#                 legend='brief', color='k', s=100, ax=ax)

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)
ax.axis('equal')
minn=min(nsens.min(),usens.min())
maxx=max(nsens.max(),usens.max())
ax.plot([minn,maxx],[minn,maxx],color='black')
#ax.set_title('Specificity')
#ax.set_title('Sensitivity')
ax.set_title('Youden')
plt.show()



# #scaar to color
# fig ,ax = plt.subplots()
# cmap=cm.get_cmap('magma')
# norm=mpl.colors.Normalize(vmin=ang_thr[0],vmax=ang_thr[-1])
# #cb1=mpl.colorbar.ColorbarBase(ax=ax,cmap=cmap)
# res_bins=[4,8,12,16]
# colors=['red','blue','green','orange']
# for i in range(0,16):
#     start=0
#     for j in range(0,16):
#         #x=tang_paraa[-1,i,start:res_bins[j],0].flatten()
#         #y=tang_paraa[-1,i,start:res_bins[j],1].flatten()
#
#         #x=tang_paraa[:,i,j, 0].flatten()
#         #y=tang_paraa[:,i,j, 1].flatten()
#
#         x = sens[i,-1,j,0, 0].flatten()
#         y = sens[i,-1,j,0, 1].flatten()
#
#         #x = spec[i,-1,j,0, 0].flatten()
#         #y = spec[i,-1,j,0, 1].flatten()
#
#         #x = sens[:, i, j, 0, 0].flatten() + spec[:,i, j, 0, 0].flatten() -1
#         #y = sens[:, i, j, 0, 1].flatten() + spec[:,i, j, 0, 1].flatten() -1
#
#         ax.scatter(x,y,i*50+150,color=cmap(j/16),alpha=0.3,marker='o')
#         #ax.scatter(x, y, i * 50 + 150, cmap=cmap,norm=norm, alpha=0.3, marker='o')
#         ax.axis('equal')
#         ax.set_title('Sensitivity')
#         ax.set_xlabel('Cartesian')
#         ax.set_ylabel('Conformal')
#         ax.plot([0,1],[0,1],color='black')
#         #start=res_bins[j]-1
# #cb=plt.colorbar()

for i in range(0,16):
    plt.plot(spec[-1,-1,:,0,0].flatten(),color='blue',alpha=i/16)
    plt.plot(spec[-1,-1,:,0,1].flatten(),color='orange',alpha=i/16)