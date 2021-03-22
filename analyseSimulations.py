import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

scale=100
res=np.linspace(1.7,1.00,8)
drt=np.linspace(0.1,0.3,5)
ang_thr=np.linspace(20,90,8)
w=np.linspace(0,0.99,4)


allsens=np.zeros([8,5,8,2,2])



l=-1
for i in range(0,len(res)):
	for j in range(0,len(drt)):
		for k in range(0,len(ang_thr)):
			base = "/home/u2hussai/scratch/unfoldingSimulations/diffusionSimulations_res-"
			npath = base + str(int(res[i] * 1000)) + "um_drt-" + str(int(drt[j] * 100)) + "_w-" +str(int(round(w[l],
																											   2) * 100)) + "_angthr-" + str(int(ang_thr[k])) + "/"

			ntang_sens=np.load(npath+'ntang_sens_alt.npy')
			utang_sens=np.load(npath+'utang_sens_alt.npy')

			#ntang_sens=ntang_sens[ntang_sens>0.2]
			#utang_sens=utang_sens[ntang_sens>0.2]


			allsens[i,j,k,0,0]=ntang_sens.mean()
			allsens[i,j,k,1,0]=utang_sens.mean()
			allsens[i,j,k,0,1]=ntang_sens.std()
			allsens[i,j,k,1,1]=utang_sens.std()


			hist = Image.open(npath+'tang_sens_hist_alt.png')
			base = "/home/u2hussai/Unfolding/data/diffusionSimulations_res-"
			npath=base + str(int(res[i] * 1000)) + "um_drt-" + str(int(drt[j] * 100)) + "_angthr-" + str(int(ang_thr[k])) + "/"
			#
			#
			plotname=str(int(res[i] * 1000)) + "um_drt-" + str(int(drt[j] * 100)) + "_angthr-" + str(int(ang_thr[k]))
			hist.save("/home/u2hussai/scratch/" + plotname + "_hist.png")
			#
			#
			#utang_lines=np.load(npath+'utang_lines.npy',allow_pickle=True)
			#ntang_lines = np.load(npath + 'ntang_lines.npy', allow_pickle=True)
			#
			#plt.figure(p)
			#fig, ax = plt.subplots(2, 1)
			#ax[0].axis('equal')
			#ax[1].axis('equal')
			#for line in ntang_lines:
			# 	line=np.asarray(line)
			# 	ax[0].plot(line[:,0],line[:,1],alpha=0.2,color='blue')
			#for line in utang_lines:
			# 	line=np.asarray(line)
			# 	ax[1].plot(line[:,0],line[:,1],alpha=0.2,color='orange')
			#plt.savefig("/home/u2hussai/scratch/" + plotname + "_tang_lines.png")
			#plt.close()
			#p=p+1


left  = 0.2  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.5   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
			
p=1
fig, ax = plt.subplots(8,5)
fig.set_figheight(16)
fig.set_figwidth(10)
fig.subplots_adjust(wspace=1)
fig.subplots_adjust(hspace=1)
for i in range(0,len(res)-1):
	for j in range(0,len(drt)):
		base = "/home/u2hussai/scratch/"
		plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
		#plt.figure(p)
		ax[i,j].set_title(plotname)
		if( i==0 and j==0):
			ax[i,j].set_ylabel('Sensitivity')
			ax[i, j].set_xlabel('Angle Thres.')

		ax[i,j].set_xlim([20,95])
		ax[i, j].set_ylim([0.3, 1])
		y=allsens[i,j,:,0,0]
		x=ang_thr
		ax[i,j].plot(x,y,color='blue')
		y=allsens[i,j,:,1,0]
		x=ang_thr
		ax[i,j].plot(x,y,color='orange')
#plt.savefig(base+plotname)
plt.savefig(base+'alldata.png')
plt.close()

