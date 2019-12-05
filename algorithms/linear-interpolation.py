#!/usr/bin/env python3


#==========================================
# Linearly interpolate between 2 points.
#==========================================


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np




def do_1d_interpol():

    def linear_interpol_1D(x, xi, yi, xj, yj):
        """
        Linearly interpolate between (xi, yi) and (yi, xi) at the point x

        returns: y(x)
        """

        return yi + (yj - yi)/(xj - xi)*(x - xi)

    xi = 5
    yi = 7

    xj = 10
    yj = 12

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([xi], [yi], c='b')
    ax.scatter([xj], [yj], c='b')
    ax.plot([xi, xj], [yi, yj], c='b', ls='--')

    for x in np.linspace(xi+0.1, xj-0.1, 10):
        ax.scatter([x], [linear_interpol_1D(x, xi, yi, xj, yj)], c='r')

    ax.set_title("1D interpolation")
    plt.show()





def do_interpol_3d_scalar():


    def linear_interpol_3D_scalar(x, xi, xj, vi, vj):
        """
        Interpolate scalar v(x) in 3D at point x between then points
        xi with vi = v(xi) and xj with vj = v(xj)
        """

        # compute vector of the line
        dx = xj - xi
        norm = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)

        dxi = x - xi
        norm_i = np.sqrt(dxi[0]**2 + dxi[1]**2 + dxi[2]**2)

        # now just interpolate like 1D along the line
        return vi + (vj - vi)/norm * norm_i

    xi = np.array([1., 2., 4.])
    xj = np.array([1., 4., 2.])

    vi = 1.2
    vj = 10.2
    vmin=min(vi, vj)-0.1
    vmax=max(vi, vj)+0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xi[0], xi[1], xi[2], c=[vi], vmin=vmin, vmax=vmax, cmap='jet', s=40)
    sc = ax.scatter(xj[0], xj[1], xj[2], c=[vj], vmin=vmin, vmax=vmax, cmap='jet', s=40)
    ax.plot( [xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], c='k')
    fig.colorbar(sc)

    n = 50
    dx = (xj - xi) / n
    for i in range(n-1):
        xij = xi + (i+1)*dx
        
        ax.scatter(xij[0], xij[1], xij[2], c=[linear_interpol_3D_scalar(xij, xi, xj, vi, vj)], vmin=vmin, vmax=vmax, cmap='jet')

    ax.set_title("3D interpolation of a scalar quantity")
    plt.show()




def do_interpol_3d_vector():

    def linear_interpol_3D_vector(x, xi, xj, vi, vj):
        """
        Interpolate vector v(x) in 3D at point x between then points
        xi with vi = v(xi) and xj with vj = v(xj)
        """

        # compute vector of the line
        dx = xj - xi
        norm = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)

        dxi = x - xi
        norm_i = np.sqrt(dxi[0]**2 + dxi[1]**2 + dxi[2]**2)



        # now just interpolate like 1D along the line
        
        return vi + (vj - vi)/norm * norm_i


    xi = np.array([1., 2., 4.])
    xj = np.array([2., 4., 2.])

    vi = np.array([0.1, 0.4, 0.3])
    vj = np.array([0.3, -0.4, 0.3])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(xi[0], xi[1], xi[2], vi[0], vi[1], vi[2], normalize=True )
    ax.quiver(xj[0], xj[1], xj[2], vj[0], vj[1], vj[2], normalize=True )
    ax.plot( [xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], c='k')

    n = 10
    dx = (xj - xi) / n
    for i in range(n-1):
        xij = xi + (i+1)*dx
        vij = linear_interpol_3D_vector(xij, xi, xj, vi, vj)
        ax.quiver(xij[0], xij[1], xij[2], vij[0], vij[1], vij[2], normalize=True, color='orange' )
        

    ax.set_title("3D interpolation of a vector quantity")
    plt.show()





if __name__ == "__main__":
    do_1d_interpol()
    do_interpol_3d_scalar()
    do_interpol_3d_vector()
