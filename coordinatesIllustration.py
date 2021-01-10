import numpy as np
import matplotlib.pyplot as plt
import cmath

def phi(X, Y, Z, w=None, scale=None, beta=None):
    if scale is None: scale = 5
    if w is None: w = 1
    rot = cmath.rect(1, beta)
    C = X + Y * 1j
    C=C*rot
    A = C / scale + w + 1
    Cout = np.log(0.5 * (A + np.sqrt(A * A - 4 * w)))
    return np.real(Cout), np.imag(Cout), Z

def dphi(X, Y, Z, w=None, scale=None, beta=None):
    if scale is None: scale = 5
    if w is None: w = 1
    if beta is None: beta = 0
    rot = cmath.rect(1, beta)
    C = X + Y * 1j
    C = C * rot
    A = C / scale + w + 1
    dCout = 1 / np.sqrt(-4 * w + A * A) * (1 / scale)
    norm = np.sqrt(np.real(dCout) * np.real(dCout) + np.imag(dCout) * np.imag(dCout))
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
    v2 = [v1[1], -v1[0], zeros]
    v3 = [zeros, zeros, ones]
    return v1, v2, v3

def phiInv(U, V, W, w=None, scale=None, beta=None):
    if scale is None: scale = 5
    if w is None: w = 1
    if beta is None: beta = 0
    rot = cmath.rect(1, beta)
    C = U + V * 1j
    #C = C*rot
    Cout = np.exp(C)
    result = scale * (Cout - 1 + w * (1 / Cout - 1))
    result=result*rot
    return np.real(result), np.imag(result), W


#set plot formatting
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)
#matplotlib.rc('text', usetex=True)


#plot the grid
uu = 0.3
u = 0
vv = np.pi / 6
v = -np.pi / 6
N = 300

U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)

UU,VV=np.meshgrid(U,V)

XX,YY,ZZ=phiInv(UU,VV,VV,w=0.95)

tx=-2.3
ty=1.5
plt.subplot(131)
plt.text(tx,ty,'a)')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
for i in range(0,N,30):
    plt.plot(XX[:,i],YY[:,i],color='b',alpha=0.6)
plt.plot(XX[:,-1],YY[:,-1],color='b',alpha=0.6)
for i in range(0, N,15):
    plt.plot(XX[i, :], YY[i, :], color='r', alpha=0.6)
plt.plot(XX[-1, :], YY[-1, :], color='r', alpha=0.6)

#plot example tracts
plt.subplot(132)
plt.text(tx,ty,'b)')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

uu = 0.2
u = 0.0
vv = np.pi / 6
v = -np.pi / 6
N = 300
U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)
UU,VV=np.meshgrid(U,V)
XX,YY,ZZ=phiInv(UU,VV,VV,w=0.95)
plt.axis('equal')
for i in range(0,N,30):
    plt.plot(XX[:,i],YY[:,i],color='b',alpha=0.6)
plt.plot(XX[:,-1],YY[:,-1],color='b',alpha=0.6)
uu = 0.3
u = 0.2
vv = np.pi / 6
v = -np.pi / 6
N = 300
U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)
UU,VV=np.meshgrid(U,V)
XX,YY,ZZ=phiInv(UU,VV,VV,w=0.95)
for i in range(0, N,15):
    plt.plot(XX[i, :], YY[i, :], color='r', alpha=0.6)
plt.plot(XX[-1, :], YY[-1, :], color='r', alpha=0.6)
#plt.text(-1.1,1.1,"u=u_r",horizontalalignment='right')

#make seed box
N=20
ones=np.ones(20)
start=0
end=0.2
lw=3
#1
u=np.linspace(start,end,N)
v=-ones*np.pi/6
x,y,z=phiInv(u,v,v,w=0.95)
plt.plot(x,y,'--',color='orange',linewidth=lw)
#2
v=np.linspace(-np.pi/6,-np.pi/9,N)
u=ones*end
x,y,z=phiInv(u,v,v,w=0.95)
plt.plot(x,y,'--',color='orange',linewidth=lw)
#3
u=np.linspace(start,end,N)
v=-ones*np.pi/9
x,y,z=phiInv(u,v,v,w=0.95)
plt.plot(x,y,'--',color='orange',linewidth=lw)
#4
v=np.linspace(-np.pi/6,-np.pi/9,N)
u=ones*start
x,y,z=phiInv(u,v,v,w=0.95)
plt.plot(x,y,'--',color='orange',linewidth=lw)


#plot example tracts notaligned
plt.subplot(133)
plt.text(tx,ty,'c)')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

uu = 0.3
u = -0.01
vv = np.pi / 2
v = -np.pi / 2
N = 450
U=np.linspace(u,uu,N)
V=np.linspace(v,vv,N)
UU,VV=np.meshgrid(U,V)
XX,YY,ZZ=phiInv(UU,VV,VV,w=0.95,beta=-np.pi/6)

plt.axis('equal')
for i in range(0,N,30):
    x=XX[:,i]
    y=YY[:, i]
    uc,vc,wc=phi(x,y,y,w=0.95,beta=0)
    line=[]
    for l in range(0,len(uc)):
        if((uc[l]<=0.2) & (uc[l]>0.0) & (vc[l]<=np.pi/6) & (vc[l]>=-np.pi/6)):
            line.append([x[l],y[l]])
        else:
            line.append([np.NaN,np.NaN])
    line=np.asarray(line)
    if line.shape[0]>1:
        plt.plot(line[:,0],line[:,1],color='b',alpha=0.6)

uu = 1.6
u = -0.05
vv = np.pi / 5
v = -np.pi / 5.5
N = 400
U = np.linspace(u, uu, N)
V = np.linspace(v, vv, N)
UU, VV = np.meshgrid(U, V)
XX, YY, ZZ = phiInv(UU, VV, VV, w=0.95, beta=-np.pi / 6)

plt.axis('equal')
for i in range(0, N, 15):
    x = XX[i, :]
    y = YY[i, :]
    uc, vc, wc = phi(x, y, y, w=0.95, beta=0)
    line = []
    for l in range(0, len(uc)):
        if ((uc[l] >= 0.2) & (uc[l] <= 0.3) & (vc[l] <= np.pi / 6) & (vc[l] >= -np.pi / 6)):
            line.append([x[l], y[l]])
        else:
            line.append([np.NaN, np.NaN])
    line = np.asarray(line)
    if line.shape[0] > 1:
        plt.plot(line[:, 0], line[:, 1], color='r', alpha=0.6)
