# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:31:08 2019

@author: dimo1
"""

from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import signal
from scipy import spatial

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# prepare some coordinates
x, y, z = np.indices((65, 65, 65))/64.0
xc = midpoints(x)
yc = midpoints(y)
zc = midpoints(z)

# draw cuboids in the top left and bottom right corners, and a link between them

#hollow shell
# sphere = (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.5)**2 ) < 0.5**2 ) & (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.5)**2 ) > 0.4**2)

# twinned dimer
# sphere = (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.25)**2 ) < 0.25**2 ) | (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.75)**2 ) < 0.25**2)
#sphere = (( abs(xc-0.5) + abs(yc-0.5) + abs(zc-0.5) ) <= 0.5 )

# trimer
# sphere = (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-1/6)**2 ) < (1/6)**2 ) | (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.5)**2 ) < (1/6)**2)| (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-5/6)**2 ) < (1/6)**2)

# # donuts
# sphere = (( (((xc-0.5)**2 + (yc-0.5)**2)**0.5 - 0.26)**2+ (zc-0.5)**2 ) < 0.22**2 ) 

 # solid sphere
sphere = (( (xc-0.5)**2 + (yc-0.5)**2 + (zc-0.5)**2 ) < 0.5**2 )

# # round plate
# sphere = (( (xc-0.5)**2 + (yc-0.5)**2 ) < 0.25 ) & (abs(zc-0.5) < 0.1 ) 

# # stick
#sphere = (( (xc-0.5)**2 + (yc-0.5)**2 ) < 0.20**2 ) & (abs(zc-0.5) <= 1 ) 

# combine the objects into a single boolean array
voxels = sphere
density = sphere.astype(int)

# set the colors of each object
colors = np.empty(voxels.shape+ (4,))
colors[..., 0] = 1
colors[..., 1] = 1-abs(density)
colors[..., 2] = 1-abs(density)
colors[..., 3] = 0.2
# and plot everything
fig1 = plt.figure(figsize=(10,10))
ax = Axes3D(fig1)
ax.voxels(x,y,z,voxels, 
          facecolors=colors,
          edgecolor='k',
          linewidth=0.2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.auto_scale_xyz
ax.set_aspect('auto')
ax.set_box_aspect(aspect = (1,1,1))
plt.draw()

#%%

autocorr=signal.correlate(density,density,mode='full')
autocorr=autocorr/autocorr.max()
autocorr_volume=autocorr.astype(bool)
dxc,dyc,dzc=np.indices((127, 127, 127))
coor=np.vstack((dxc.ravel(),dyc.ravel(),dzc.ravel())).T
center = np.array([63,63,63])
distance=spatial.distance.cdist(coor,center.reshape(1,-1)).ravel()
distance=np.reshape(distance,autocorr.shape)
distance=np.reshape(distance,autocorr.shape)/63

colors2 = np.empty(autocorr.shape + (4,))
colors2[...,0]=1
colors2[...,1]=1-autocorr
colors2[...,2]=1-autocorr
colors2[...,3]=1

fig2 = plt.figure(figsize=(10,10))
ax = Axes3D(fig2)
ax.voxels(autocorr_volume[0:64], 
          facecolors=colors2[0:64],
          edgecolor='k',
          linewidth=0.1)
ax.set_aspect('auto')
ax.set_box_aspect(aspect = (1,1,1))
ax.view_init(elev=10., azim=0)
plt.draw()

#%%
plt.figure()
R_size=41
R_space=np.empty(R_size+1)
pdf=np.empty(R_size+1)

for i in range(0,R_size+1):
    R_space[i]=i/R_size*1.732
    shell = (i/R_size<=distance/distance.max()) & (distance/distance.max()<(i+1)/R_size)
    pdf[i]=sum(autocorr[shell])
   
plt.plot(R_space,pdf)
#%%

with open('stick2.pickle', 'wb') as f:
    pickle.dump(pdf, f)
