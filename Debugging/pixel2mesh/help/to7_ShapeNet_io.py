from __future__ import print_function
import numpy as np 
import os
import cPickle as pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.spatial import ConvexHull  


def dat_read(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        pkl_dict = pickle.load(pkl_file)
        img = pkl_dict[0].astype('float32')/255.0
        label = pkl_dict[1]
        gt_pt = label[:,:3]
        gt_nm = label[:,3:]
        return img, gt_pt, gt_nm

def plot_3D_mesh(gt_pt):
    # Initialize
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    X = gt_pt[:,[0]]
    Y = gt_pt[:,[1]]
    Z = gt_pt[:,[2]]

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    pass

def plot_3D_scatter(gt_pt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot_3D_ConvexHull(gt_pt):
    x = gt_pt[:,0]
    y = gt_pt[:,1]
    z = gt_pt[:,2]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hull=ConvexHull(gt_pt)

    edges= zip(*gt_pt)

    for i in hull.simplices:
        plt.plot(gt_pt[i,0], gt_pt[i,1], gt_pt[i,2], 'r-', linewidth=0.1)
    
    ax.plot(edges[0],edges[1],edges[2],'b.', linewidth=0.1) 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d(x.min(), x.max())
    ax.set_ylim3d(y.min(), y.max())
    ax.set_zlim3d(z.min(), z.max())


    plt.show()


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

if __name__ == "__main__":

    print(os.getcwd())
    # pkl_path = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/ShapeNetTrain/04256520_1a4a8592046253ab5ff61a3a2a0e2484_00.dat'
    pkl_path = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/ShapeNetTrain/02691156_787d2fbb247c04266818a2bd5aa39c80_08.dat'
    # Read the .dat file
    _, gt_pt, gt_nm = dat_read(pkl_path)

    gt_pt
    # Draw the 3D Plot
    plot_3D_ConvexHull(gt_pt)

    pass
