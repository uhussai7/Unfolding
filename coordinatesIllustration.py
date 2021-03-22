import numpy as np
import matplotlib.pyplot as plt
import cmath

def phi(X,Y,Z,w=None,scale=None,beta=None):
    if scale is None: scale=5
    if w is None: w=1
    C=X+Y*1j
    #A=C/scale + w+1
    #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
    #A=C/scale
    #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
    Cout=np.power(C,1./w)
    return np.real(Cout), np.imag(Cout), Z

def phiInv(U,V,W,w=None,scale=None,beta=None):
    if scale is None: scale = 5
    if w is None: w=1
    C = U + V * 1j
    #Cout = np.exp(C)
    #result = scale*(Cout-1 + w*(1/Cout-1))
    #result = scale*(Cout+ w*(1/Cout))
    result = np.power(C,w)
    return np.real(result), np.imag(result), W

def dphi(X,Y,Z,w=None, scale=None, beta=None):
    if scale is None: scale = 5
    if w is None: w=1
    if beta is None: beta=0
    #rot=cmath.rect(1,beta)
    C=X+Y*1j
    #C=C*rot
    # A = C / scale + w +1
    # dCout= 1/np.sqrt(-4 * w + A*A)*(1/scale)
    dCout = (1/w)*np.power(C,1/w-1)
    norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
    v2 = [v1[1], -v1[0], zeros]
    v3 = [zeros, zeros, ones]
    return v1, v2, v3


#set plot formatting
import matplotlib
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 25}

#matplotlib.rc('font', **font)
#matplotlib.rc('text', usetex=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 24,
})



#plot the grid
uu = 1/1.75
u = 0.02
vv = np.pi / 4
v = -np.pi / 4
N = 300

U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)

plt.figure()
UU,VV=np.meshgrid(U,V)
ws=np.linspace(1,1.99,4)
for idx,w in enumerate(ws):
    plt.subplot(1,4,idx+1)
    plt.axis('equal')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title("$w=$%.2f" % (w))
    plt.grid(True)
    XX,YY,ZZ=phiInv(UU,VV,VV,w=w)
    for i in range(0,N,30):
        plt.plot(XX[:,i],YY[:,i],color='b',alpha=1)
    plt.plot(XX[:,-1],YY[:,-1],color='b',alpha=1)
    for i in range(0, N,15):
        plt.plot(XX[i, :], YY[i, :], color='r', alpha=1)
    plt.plot(XX[-1, :], YY[-1, :], color='r', alpha=1)
    #plt.tight_layout()
#regions and tracts


#tangential tracts
uu = 1/1.75
u = 0.02
vv = np.pi / 4
v = -np.pi / 4
N = 300

x_h = 0.5 * (phiInv(uu, 0, 0, w=1.99)[0] + phiInv(u, 0, 0, w=1.99)[0])
drt = phi(x_h, 0, 0,w=1.99)[0]

uu=drt

U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)

UU,VV=np.meshgrid(U,V)
XX,YY,ZZ=phiInv(UU,VV,VV,w=1.99)

plt.figure(figsize=( 5.5,7))
plt.subplot(111)
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True)
for i in range(0,int(N/1),35):
    plt.plot(XX[:,i],YY[:,i],color='b',alpha=1)
    #plt.plot(XX[:,-1],YY[:,-1],color='b',alpha=0.6)

#radial tracts
uu = 1/1.75
u = 0.02
vv = np.pi / 4
v = -np.pi / 4
N = 300

x_h = 0.5 * (phiInv(uu, 0, 0, w=1.99)[0] + phiInv(u, 0, 0, w=1.99)[0])
drt = 0.88*phi(x_h, 0, 0,w=1.99)[0]

u=drt

U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)

UU,VV=np.meshgrid(U,V)
XX,YY,ZZ=phiInv(UU,VV,VV,w=1.99)

for i in range(0, N,15):
    plt.plot(XX[i, :], YY[i, :], color='r', alpha=1.0)
    plt.plot(XX[-1, :], YY[-1, :], color='r', alpha=1.0)
plt.tight_layout()

plt.figure(figsize=( 5.5,7))
plt.subplot(111)
plt.xlabel('$x$')
plt.ylabel('$y$')

#seed region
#regions and tracts
uu = 1/1.75
u = 0.02
vv = np.pi / 4
v = -np.pi / 4
N = 300

x_h = 0.5 * (phiInv(uu, 0, 0, w=1.99)[0] + phiInv(u, 0, 0, w=1.99)[0])
drt = phi(x_h, 0, 0,w=1.99)[0]

uu=drt

U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)

N_angles=19
delta_angles=(vv-v)/(N_angles-1)
delta_angles=v+delta_angles
N_line=50
uline=np.linspace(u,uu,N_line)
vline=v*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X[:],Y[:],color='orange')

N_line=50
uline=np.linspace(u,uu,N_line)
vline=delta_angles*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X[:],Y[:],color='orange')

N_line=50
uline=u*np.ones(N_line)
vline=np.linspace(v,delta_angles,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='orange')

N_line=50
uline=uu*np.ones(N_line)
vline=np.linspace(v,delta_angles,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='orange')

#radial region
x_h =phiInv(u, 0, 0, w=1.99)[0]+ 0.75 * (phiInv(1/1.75, 0, 0, w=1.99)[0] - phiInv(u, 0, 0, w=1.99)[0])
drt = phi(x_h, 0, 0,w=1.99)[0]

N_line=50
uline=np.linspace(drt,1/1.75,N_line)
vline=delta_angles*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='red')

N_line=50
uline=np.linspace(drt,1/1.75,N_line)
vline=vv*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='red')

N_line=50
uline=drt*np.ones(N_line)
vline=np.linspace(delta_angles,vv,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='red')

N_line=50
uline=(1/1.75)*np.ones(N_line)
vline=np.linspace(delta_angles,vv,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='red')


#tangential region
N_line=50
uline=u*np.ones(N_line)
vline=np.linspace(v,vv,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='blue')

N_line=50
uline=uu*np.ones(N_line)
vline=np.linspace(v,vv,N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='blue')

N_line=50
uline=np.linspace(u,uu,N_line)
vline=v*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='blue')

N_line=50
uline=np.linspace(u,uu,N_line)
vline=vv*np.ones(N_line)
X,Y,buffer=phiInv(uline,vline,0,w=1.99)
plt.plot(X,Y,color='blue')
plt.axis('equal')

plt.annotate('$U_\Sigma$',xy=(-0.5,-0.35),xycoords='data',xytext=(-0.5,-0.75),arrowprops=dict(facecolor='black',
            shrink=0.05),horizontalalignment='center', verticalalignment='top')
plt.annotate('$U_T$',xy=(-0.445,0.634),xycoords='data',xytext=(-0.626,0.839),arrowprops=dict(facecolor='black',
            shrink=0.05),horizontalalignment='center', verticalalignment='top')
plt.annotate('$U_R$',xy=(0.037,0.624),xycoords='data',xytext=(0.268,0.865),arrowprops=dict(facecolor='black',
            shrink=0.05),horizontalalignment='center', verticalalignment='top')
plt.tight_layout()