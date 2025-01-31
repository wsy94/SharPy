# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:31:08 2019

Modified on Feb 1 2024

@author: wsy94
"""

#%%

from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import signal
from scipy import spatial
from scipy import ndimage

# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

def autocorr_map(density):
    autocorr=signal.correlate(density,density,mode='full')
    autocorr=autocorr/autocorr.max()
    # autocorr_volume=autocorr.astype(bool)
    return(autocorr)


def distance_map(autocorr, i):
    dxc,dyc,dzc=np.indices((2*i-1, 2*i-1, 2*i-1))
    coor=np.vstack((dxc.ravel(),dyc.ravel(),dzc.ravel())).T
    center = np.array([i-1,i-1,i-1])
    distance=spatial.distance.cdist(coor,center.reshape(1,-1)).ravel()
    distance=np.reshape(distance,autocorr.shape)
    distance=np.reshape(distance,autocorr.shape)/(i-1)
    
    return(distance)

def generate_voxel_structure(structure, xc, yc, zc, **kwargs):
    """
    Generates a 3D voxel structure based on the selected type.

    Parameters:
        structure (str): Type of structure ('Trimer', 'Donut', 'Sphere', 'RoundPlate', 'Stick', 'Octahedron', 'Tetrahedron', 'HollowShell')
        xc, yc, zc (ndarray): Voxel coordinates (0-1)

    Returns:
        ndarray: Boolean array defining the voxel structure
    """

    params = {
        "xo": 0.5,            # Center of the structure at X
        "yo": 0.5,            # Center of the structure at Y
        "zo": 0.5,            # Center of the structure at Z

        "sphere_radius": 0.45,  # Sphere radius

        "plate_radius": 0.25,       # RoundPlate radius
        "plate_thickness": 0.1,      # Plate thickness

        "stick_radius": 0.20,  # Stick radius
        "stick_length": 1.0,   # Stick length

        "octahedron_size": 0.5,  # Octahedron size

        "tetra_r": 0.90,       # Tetrahedron size factor
        "tetra_c": 0.05,       # Tetrahedron offset

        "hollow_outer_radius": 0.5,  # Outer radius for HollowShell
        "hollow_inner_radius": 0.4,  # Inner radius for HollowShell

        "hollow_outer_radius": 0.5,  # Outer radius for HollowShell
        "hollow_inner_radius": 0.4,  # Inner radius for HollowShell

        "torus_radius": 0.26,  # Torus (Donut) main radius
        "tube_radius": 0.22,   # Torus (Donut) tube radius

        "r": 1/6,          # Sphere radius for Dimer/Trimer
        "c": 0.5,            # Center offset for Dimer/Trimer
        "d": 1/3,            # distance between spheres in Dimer/Trimer

        "helix_fiber_radius": 0.03,    #radius of the fiber
        "helix_radius": 0.4,     #radius of the helix
        "helix_zf": 2,     #cycles of z-axis
        "helix_phase": 1,   #phase of helix, 0 or 1 for inverse direction
    }

    params.update(kwargs)
    sphere = (xc+yc+zc) < -1 # initializer

    if structure == "Sphere":
        f_name = "Sphere"
        sphere = ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - params["zo"]) ** 2) < params["sphere_radius"] ** 2

    elif structure == "RoundPlate":
        f_name = "RoundPlate"
        sphere = ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2) < params["plate_radius"] and (abs(zc - params["zo"]) < params["plate_thickness"])

    elif structure == "Stick":
        f_name = "Stick"
        sphere = ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2) < params["stick_radius"] ** 2 and (abs(zc - params["zo"]) <= params["stick_length"])

    elif structure == "Octahedron":
        f_name = "Octahedron"
        sphere = (abs(xc - params["xo"]) + abs(yc - params["yo"]) + abs(zc - params["zo"])) <= params["octahedron_size"]

    elif structure == "Tetrahedron":
        f_name = "Tetrahedron"
        sphere = (
            (((xc - params["xo"]) / 1) + ((yc - params["yo"]) / 1) + ((zc - params["zo"]) / 1)) >= params["tetra_r"]
        ) & (
            ((-((xc - params["xo"]) / 1) + ((yc - params["yo"]) / 1) + ((zc - params["zo"]) / 1))) <= params["tetra_r"]
        ) & (
            (((xc - params["xo"]) / 1) - ((yc - params["yo"]) / 1) + ((zc - params["zo"]) / 1)) <= params["tetra_r"]
        ) & (
            (((xc - params["xo"]) / 1) + ((yc - params["yo"]) / 1) - ((zc - params["zo"]) / 1)) <= params["tetra_r"]
        )

    elif structure == "HollowShell":
        f_name = "HollowShell"
        sphere = (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - params["zo"]) ** 2) < params["hollow_outer_radius"] ** 2
        ) & (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - params["zo"]) ** 2) > params["hollow_inner_radius"] ** 2
        )

    elif structure == "Donut":
        f_name = "Donut"
        sphere = (
            (((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2) ** 0.5 - params["torus_radius"]) ** 2
            + (zc - params["zo"]) ** 2
        ) < params["tube_radius"] ** 2

        # Dimer
    elif structure == "Dimer":
        f_name = "Dimer"
        radius = params["r"]
        c = params["zo"]
        distance = params["d"]  # Default separation (tunable)

        z1 = c - distance / 2
        z2 = c + distance / 2

        sphere = (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - z1) ** 2) < radius ** 2
        ) | (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - z2) ** 2) < radius ** 2
        )
    
    elif structure == "Trimer":
        f_name = "Trimer"
        radius = params["r"]
        c = params["zo"]
        distance = params["d"]  # Default separation (tunable)

        z1 = c - distance
        z2 = c
        z3 = c + distance
        sphere = (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - z1) ** 2) < radius ** 2
        ) | (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - z2) ** 2) < radius ** 2
        ) | (
            ((xc - params["xo"]) ** 2 + (yc - params["yo"]) ** 2 + (zc - z3) ** 2) < radius ** 2
        )

    elif structure == 'Helix':
        f_name = "Helix"
        r = params["helix_fiber_radius"]     # radius of the fiber
        R = params["helix_radius"]           # radius of the helix
        z_f = params["helix_zf"]             # cycles of z-axis
        phase = params["helix_phase"]             # 0 or 1 to reverse direction

        # helix will span along z-axis
        sphere = (xc+yc+zc) < -1            # initialize the voxel matrix
        for ii in np.arange(0,len(zc)):     # z-direction only, iterate from z-layers
            sphere_new= (((xc-params["xo"]-R*phase*np.cos(np.pi*(ii/len(zc))*z_f))**2 +
                        (yc-params["yo"]-R*phase*np.sin(np.pi*(ii/len(zc))*z_f))**2) < (r)**2 ) & (abs(zc-ii/len(zc)) < 1/len(zc))
            sphere += sphere_new

    else:
        raise ValueError(f"Unknown structure type: {structure}")

    return sphere, f_name

def generate_lattice_structure_3D(grid, xc, yc, zc, element="Sphere", **kwargs):
    """
    Generates a 3D periodic lattice by repeating a structure (e.g., Sphere, Donut, etc.).

    Parameters:
        grid (np.ndarray): A 3D boolean array (N x N x N) defining occupied sites (True = occupied).
        element (str): Name of the structure type (e.g., 'Sphere', 'Trimer', etc.).
        xc, yc, zc (ndarray): voxel space coordinates
        **kwargs: Additional parameters passed to generate_voxel_structure().

    Returns:
        np.ndarray: A combined voxel lattice structure.
    """
    N = grid.shape[0]  # Lattice size (NxNxN)
    
    # Define tick marks (center positions of elements)
    # tick = np.linspace(0, 1, N, endpoint=False) + 1/(2*N) # not including edge
    tick = np.linspace(0, 1, N, endpoint=True)  # on edge of box
    CoS_x, CoS_y, CoS_z = np.meshgrid(tick, tick, tick)  # Generate center coordinates

    # Initialize an empty 3D volume for storing the voxels
    lattice = (xc+yc+zc) < -1

    # Iterate over the lattice grid
    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                if grid[ii, jj, kk]:  # Only generate structures at occupied sites
                    xo, yo, zo = CoS_x[ii, jj, kk], CoS_y[ii, jj, kk], CoS_z[ii, jj, kk], 

                    # Extract site-specific parameters (if provided)
                    site_kwargs = {key: val[ii, jj, kk] if isinstance(val, np.ndarray) else val for key, val in kwargs.items()}

                    # Generate the voxel structure at this site
                    structure, _ = generate_voxel_structure(element, xc, yc, zc, xo=xo, yo=yo, zo=zo, **site_kwargs)

                    # Merge into the lattice (OR operation to combine shapes)
                    lattice |= structure

    return lattice


#%%
# define voxel space size
i=64
# periodic boundary conditions switch
Is_periodic = 0

# prepare some coordinates
x, y, z = np.indices((i+1, i+1, i+1))/float(i)
xc = midpoints(x)
yc = midpoints(y)
zc = midpoints(z)

#%%
# Example use of generating a sphere
structure = "Helix"
# voxel, f_name = generate_voxel_structure(structure, xc, yc, zc, sphere_radius=0.4)
voxel, f_name = generate_voxel_structure(structure, xc, yc, zc)
density = voxel.astype(float)


#%%
# You can also generate lattice structure here


# Example (generating 2D lattice of assembled helix structure):
f_name='helix_assembles'
Is_periodic = 1
Grid = np.zeros((5,5,5))
Grid[:,:,2] = np.array([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1]])

Phase  = np.zeros((5,5,5))
Phase[:,:,2] = np.array([[1, 0, 1, 0, 1],
                [0, -1, 0, -1, 0],
                [1, 0, 1, 0, 1],
                [0, -1, 0, -1, 0],
                [1, 0, 1, 0, 1]])

Radius  = np.zeros((5,5,5))
Radius[:,:,2] = 0.15

voxel = generate_lattice_structure_3D(Grid, xc, yc, zc, element="Helix", helix_phase=Phase, helix_radius = Radius)
density = voxel.astype(float)

#%%
# Experimental code for generating triple helix structure
# R*Phase[jj,kk]*np.cos(np.pi*(ii/i)*z_f+2*np.pi/3) where 2*np.pi/3 stand for internal phase shift

# for ii in np.arange(0,i): #z-direction
#     for jj in np.arange(0,N):
#         for kk in np.arange(0,N):
#             if Grid[jj,kk]:
#                 sphere_new = (((xc-CoS[0][jj,kk]-R*Phase[jj,kk]*np.cos(np.pi*(ii/i)*z_f))**2 + \
#                                 (yc-CoS[1][jj,kk]-R*Phase[jj,kk]*np.sin(np.pi*(ii/i)*z_f))**2) < (r)**2 ) & (abs(zc-ii/i) < 1/i)
#                 sphere = sphere + Grid[jj,kk]*sphere_new
                
#                 sphere_new = (((xc-CoS[0][jj,kk]-R*Phase[jj,kk]*np.cos(np.pi*(ii/i)*z_f+2*np.pi/3))**2 + \
#                                 (yc-CoS[1][jj,kk]-R*Phase[jj,kk]*np.sin(np.pi*(ii/i)*z_f+2*np.pi/3))**2) < (r)**2 ) & (abs(zc-ii/i) < 1/i)
#                 sphere = sphere + Grid[jj,kk]*sphere_new

#                 sphere_new = (((xc-CoS[0][jj,kk]-R*Phase[jj,kk]*np.cos(np.pi*(ii/i)*z_f-2*np.pi/3))**2 + \
#                                 (yc-CoS[1][jj,kk]-R*Phase[jj,kk]*np.sin(np.pi*(ii/i)*z_f-2*np.pi/3))**2) < (r)**2 ) & (abs(zc-ii/i) < 1/i)
#                 sphere = sphere + Grid[jj,kk]*sphere_new


# Example (generating 3D lattice of assembled sphere particles):
# 7x7 Grid
Grid_gammabrass=np.array([[[0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0]],

                [[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]],

                [[1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1]],

                [[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]],
               
                [[1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1]],

                [[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]],
               
                [[0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0]]]
               ) #gamma brass

# 5x5 Grid
Grid_fcc=np.array([[[1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1]],

               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1]]]) #fcc grid

Grid_fcc=np.array([[[1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1]],

               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1]]]) #bcc grid

Grid_A15=np.array([[[1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1]],

               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0]],
               
               [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
               
               [[1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1]]]) #A15 grid

# 3x3 Grid (Archived)
# # Grid=np.array([[[1, 0, 1],
# #                 [0, 1, 0],
# #                 [1, 0, 1]],

# #                 [[0, 1, 0],
# #                 [1, 0, 1],
# #                 [0, 1, 0]],

# #                 [[1, 0, 1],
# #                 [0, 1, 0],
# #                 [1, 0, 1]]]) #fcc grid

# # Grid=np.array([[[1, 0, 1],
# #                 [0, 0, 0],
# #                 [1, 0, 1]],

# #                [[0, 0, 0],
# #                 [0, 1, 0],
# #                 [0, 0, 0]],

# #                [[1, 0, 1],
# #                 [0, 0, 0],
# #                 [1, 0, 1]]]) #bcc grid

f_name='GridNxN'
Grid=Grid_fcc
N = Grid.shape[0]
r = np.ones(Grid.shape)*(1/N*1/2*0.5); # Define the radius of each sphere
voxel = generate_lattice_structure_3D(Grid, xc, yc, zc, element="Sphere", sphere_radius=r)

density = voxel.astype(float)


#%%
# # Archived: Plot 3D density/autocorrelation voxel graph
# Reason: Axes3D() is too slow

# fig1 = plt.figure(figsize=(10,10))
# ax = Axes3D(fig1)
# ax.voxels(x,y,z,voxels, 
#           facecolors=colors,
#           edgecolor='k',
#           linewidth=0.2)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.auto_scale_xyz
# ax.set_aspect('auto')
# ax.set_box_aspect(aspect = (1,1,1))
# plt.draw()

#%%

# Experimental:  Optional Gaussian Filter for Smearing out Voxel Surfaces for reduced roughness
# density = ndimage.gaussian_filter(density, sigma=0.5)

#%%
if Is_periodic:
    density_tiled = np.tile(density, (3, 3, 3))
    autocorr_tiled = autocorr_map(density_tiled)
    autocorr = autocorr_tiled[2*i:4*i-1,2*i:4*i-1,2*i:4*i-1]
else:
    autocorr = autocorr_map(density)
    
distance = distance_map(autocorr, i)
distance = distance/distance.max()


# Experimental:  Storing density/autocorrelation/distancemap in a single pickle file

# class obj_3D:
#     density = density
#     autocorr = autocorr
#     distance = distance
    
# obj = obj_3D()

# with open(f_name+'_shape.pickle', 'wb') as f:
#     pickle.dump(obj, f)

#%%

# Experimental:  Storing density/autocorrelation/distancemap in a single pickle file

# f_name='Tetrahedron'

# with open(f_name+'shape.pickle', 'rb') as f:
#     obj=pickle.load(f)

# density = obj.density
# autocorr = obj.autocorr
# distance = obj.distance

# Archived: Plot 3D density/autocorrelation voxel graph
# Reason: Axes3D() is too slow

# colors2 = np.empty(autocorr.shape + (4,))
# colors2[...,0]=1
# colors2[...,1]=1-autocorr
# colors2[...,2]=1-autocorr
# colors2[...,3]=1

# fig2 = plt.figure(figsize=(10,10))
# ax = Axes3D(fig2)
# ax.voxels(autocorr_volume[0:i], 
#           facecolors=colors2[0:i],
#           edgecolor='k',
#           linewidth=0.1)
# ax.set_aspect('auto')
# ax.set_box_aspect(aspect = (1,1,1))
# ax.view_init(elev=10., azim=0)
# plt.draw()

# %%
# Experimental: binning the distance map to calculate pair distribution function (PDF)

# plt.figure()
# R_size=81
# R_space=np.empty(R_size+1)
# pdf=np.empty(R_size+1)

# for iii in range(0,R_size+1):
#     R_space[iii]=iii/R_size*1.732
#     shell = (iii/R_size<=distance/distance.max()) & (distance/distance.max()<(iii+1)/R_size)
#     pdf[iii]=sum(autocorr[shell])
   
# plt.plot(R_space,pdf)

# with open(f_name+'_pdf.pickle', 'wb') as f:
#     pickle.dump(pdf, f)

#%%
# Plot the density or autocorrelation function using an interactive slider tool
# Faster than Axel3D

# 0 = x-axis; 1 = y-axis; 2 = z-axis

# %matplotlib widget
import matplotlib
matplotlib.use('widget')
from matplotlib.widgets import Slider

# Create a figure and axis
fig, ax = plt.subplots(1,2,figsize=(12, 5))

# Create a slider
slider_ax_1 = plt.axes([0.1, 0.04, 0.35, 0.03])
slider_1 = Slider(slider_ax_1, 'Slice', 0, density.shape[2]-1, int(density.shape[2]/2), valstep=1)

# Define a function to update the plot when the slider value changes
def update_1(val):
    slice_index = int(slider_1.val)
    ax[0].clear()
    # ax.scatter(xc[slice_index], yc[slice_index], vmin=0)
    ax[0].imshow(density[:, :, slice_index], cmap='gray')
    fig.canvas.draw_idle()

# Register the update function with the slider
slider_1.on_changed(update_1)

# Initial plot
ax[0].imshow(density[:, :, int(density.shape[2]/2)], cmap='gray')


# Create another slider
slider_ax_2 = plt.axes([0.55, 0.04, 0.35, 0.03])
slider_2 = Slider(slider_ax_2, 'Slice', 0, autocorr.shape[2]-1, int(autocorr.shape[2]/2), valstep=1)

# Define a function to update the plot when the slider value changes
def update_2(val):
    slice_index = int(slider_2.val)
    ax[1].clear()
    # ax.scatter(xc[slice_index], yc[slice_index], vmin=0)
    ax[1].imshow(autocorr[:, :, slice_index], cmap='gray')
    fig.canvas.draw_idle()

# Register the update function with the slider
slider_2.on_changed(update_2)

# Initial plot
ax[1].imshow(autocorr[:, :, int(autocorr.shape[2]/2)], cmap='gray')

# Show the plot
plt.show()

fig2, ax2 = plt.subplots(1,3,figsize=(12, 4))
autocorr_stacked = autocorr.sum(axis=2)
ax2[0].imshow(autocorr_stacked)
ax2[0].set_title('Stacked autocorr X-Y plane')

autocorr_stacked = autocorr.sum(axis=1)
ax2[1].imshow(autocorr_stacked)
ax2[1].set_title('Stacked autocorr X-Z plane')

autocorr_stacked = autocorr.sum(axis=0)
ax2[2].imshow(autocorr_stacked)
ax2[2].set_title('Stacked autocorr Y-Z plane')

plt.show()

# %%
# Experimental: Convert PDF to 1D SAXS data

# import pdf2iq
# pdf2iq.pdf2iq(R_space, pdf, mean = 10)
# %%
# TBD: Using FFT to calculate the 3D scattering intensity distribution
