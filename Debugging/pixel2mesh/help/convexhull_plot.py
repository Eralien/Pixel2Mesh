import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

points= np.array([[0,0,0],
            [4,0,0],
            [4,4,0],
            [0,4,0],
            [0,0,4],
            [4,0,4],
            [4,4,4],
            [0,4,4]])

hull=ConvexHull(points)

edges= zip(*points)

for i in hull.simplices:
    plt.plot(points[i,0], points[i,1], points[i,2], 'r-')

ax.plot(edges[0],edges[1],edges[2],'bo') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim3d(-5,5)
ax.set_ylim3d(-5,5)
ax.set_zlim3d(-5,5)

plt.show()