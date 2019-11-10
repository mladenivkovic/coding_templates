#!/usr/bin/env python3

#=====================================================================
# Perform a neighbour search for particles given a maximal distance
# for a neighbour, where the distance is individual to each particle.
# Cross compare with a naive N^2 search, time it, and plot it.
# Do it for 2D only. Extension to 3D is trivial.
#=====================================================================


import numpy as np
import matplotlib.pyplot as plt
import time



#====================================
def main():
#====================================

    # invent some data
    npart = 1000
    L = 1.0
    periodic = False
    h_glob = 0.12

    np.random.seed(10)
    x = np.random.random(size=npart) * L
    y = np.random.random(size=npart) * L
    h = np.ones(npart, dtype=np.float)*h_glob

    # if you want to manually set a particle, set it here
    part_to_plot = 0
    x[0] = 0.7
    y[0] = 0.04


    # get neighbours, and time it
    start_n = time.time()
    neigh_n, nneigh_n = get_neighbours_naive(x, y, h, L=L, periodic=periodic)
    stop_n = time.time()

    start_s = time.time()
    neigh_s, nneigh_s = get_neighbours(x, y, h, L=L, periodic=periodic)
    stop_s = time.time()

    tn = stop_n - start_n
    ts = stop_s - start_s
    print('{0:18} {1:18} {2:18}'.format("time naive", "time better", "naive/better"))
    print('{0:18.4f} {1:18.4f} {2:18.4f}'.format(tn, ts, tn/ts))

    #  check that results are identical
    compare_results(x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s, L)

    # plot the neighbours of chosen particle
    plot_solution(x, y, h, part_to_plot, neigh_s, L=L, periodic=periodic)
    
    # run a speed comparison 
    compare_speeds()


    return







#========================================================================
def get_neighbours(x, y, h, fact=1.0, L=1.0, periodic=True):
#========================================================================
    """
    Gets all the neighbour data for all particles ready.
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns:
        neighbours:     list of lists of all neighbours for each particle
        nneigh:         numpy array of now many neighbours each particle has


    """


    #-----------------------------------------------
    class cell:
    #-----------------------------------------------
        """
        A cell object to store particles in.
        Stores particle indexes
        """

        def __init__(self):
           self.npart = 0
           self.size = 100
           self.parts = np.zeros(self.size, dtype=np.int)
           self.x = np.zeros(self.size, dtype=np.float)
           self.y = np.zeros(self.size, dtype=np.float)
           self.h = np.zeros(self.size, dtype=np.float)
           return

        def add_particle(self, ind, xp, yp, hp):
            """
            Add a particle, store the index, positions and h
            """
            if self.npart == self.size:
                self.parts = np.append(self.parts, np.zeros(self.size, dtype=np.int))
                self.x = np.append(self.x, np.zeros(self.size, dtype=np.float))
                self.y = np.append(self.y, np.zeros(self.size, dtype=np.float))
                self.h = np.append(self.h, np.zeros(self.size, dtype=np.float))
                self.size *= 2

            self.parts[self.npart] = ind
            self.x[self.npart] = xp
            self.y[self.npart] = yp
            self.h[self.npart] = hp
            self.npart += 1
            
            return


    
    #-------------------------------------------------------
    def find_neighbours_in_cell(i, j, p, xx, yy, hh):
    #-------------------------------------------------------
        """
        Find neighbours of a particle in the cell with indices i,j
        of the grid
        p:      global particle index to work with
        xx, yy: position of particle x
        hh:     compact support radius for p
        """
        n = 0
        neigh = [0 for i in range(1000)]
        ncell = grid[i][j] # neighbour cell we're checking for

        N = ncell.npart
        
        fhsq = hh*hh*fact*fact

        for c, cp in enumerate(ncell.parts[:N]):
            if cp == p:
                # skip yourself
                continue

            dx, dy = get_dx(xx, ncell.x[c], yy, ncell.y[c], L=L, periodic=periodic)

            dist = dx**2 + dy**2

            if dist <= fhsq:
                try:
                    neigh[n] = cp
                except ValueError:
                    nneigh+=[0 for i in range(1000)]
                    nneigh[n] = cp
                n += 1

        return neigh[:n]




    npart = x.shape[0]

    # first find cell size
    ncells = int(L/h.max()) + 1
    cell_size = L/ncells
    #  print("ncells is", ncells)
    #  print("cell size is", cell_size)

    # create grid
    grid = [[cell() for i in range(ncells)] for j in range(ncells)]

    # sort out particles
    for p in range(npart):
        i = int(x[p]/cell_size)
        j = int(y[p]/cell_size)
        grid[i][j].add_particle(p, x[p], y[p], h[p])


    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)
   

    if ncells < 4:
        # you'll always need to check all cells, so just do that
        i_search = np.zeros(ncells*ncells, dtype=np.int)
        j_search = np.zeros(ncells*ncells, dtype=np.int)
        ind = 0
        for i in range(ncells):
            for j in range(ncells):
                i_search[ind] = i
                j_search[ind] = j
                ind += 1

        # main loop: find and store all neighbours:
        for p in range(npart):
            nbors = []
            for i,j in zip(i_search, j_search):
                nbors += find_neighbours_in_cell(i, j, p, x[p], y[p], h[p])
        
            neighbours[p] = nbors
            nneigh[p] = len(nbors)

        return neighbours, nneigh





    # main loop: find and store all neighbours;
    # go cell by cell
    for row in range(ncells):
        for col in range(ncells):

            cell = grid[row][col]
            N = cell.npart
            parts = cell.parts
            if N == 0: continue

            hmax = cell.h[:N].max()

            maxdist = int(cell_size/hmax+0.5) + 1

            # loop over all neighbours
            # need to loop over entire square. You need to consider
            # the maximal distance from the edges/corners, not from
            # the center of the cell!
            for i in range(-maxdist, maxdist+1):
                for j in range(-maxdist, maxdist+1):
                    iind = row+i
                    jind = col+j

                    if periodic:
                        while iind<0: iind += ncells
                        while iind>=ncells: iind -= ncells
                        while jind<0: jind += ncells
                        while jind>=ncells: jind -= ncells
                    else:
                        if iind < 0 or iind >= ncells:
                            continue
                        if jind < 0 or jind >= ncells:
                            continue

                    # get closest corner of neighbour
                    ic = iind
                    if i < 0: ic += 1
                    jc = jind
                    if j < 0: jc += 1

                    xc = ic*cell_size
                    yc = jc*cell_size

                    # loop over all particles in cell
                    for pc, pg in enumerate(cell.parts[:N]):

                        xp = cell.x[pc]
                        yp = cell.y[pc]
                        hp = cell.h[pc]

                        # if i or j = 0, then compare to the edge, not to the corner
                        if i == 0:
                            xc = xp
                        if j == 0:
                            yc = yp

                        # check distance to corner of neighbour cell 
                        dx, dy = get_dx(xp, xc, yp, yc, L=L, periodic=periodic)
                        dsq = dx**2 + dy**2 
                        if dsq / hp**2 > 1:
                            continue

                        neighbours[pg] += find_neighbours_in_cell(iind, jind, pg, xp, yp, hp)


    # sort neighbours by index
    for p in range(npart):
        neighbours[p].sort()
        nneigh[p] = len(neighbours[p])

    return neighbours, nneigh








#===============================================================================
def get_neighbours_naive(x, y, h, fact=1.0, L=1.0, periodic=True):
#===============================================================================
    """
    Gets all the neighbour data for all particles ready.
    Naive way: Loop over all particles for each particle
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    
    returns:
        neighbours:     list of lists of all neighbours for each particle
        nneigh:         numpy array of now many neighbours each particle has

    """
     
    #--------------------------------------------------------------------------------
    def find_neighbours_for_i_naive(ind, x, y, h, fact=1.0, L=1.0, periodic=True):
    #--------------------------------------------------------------------------------
        """
        Find indices of all neighbours of a particle with index ind
        within fact*h (where kernel != 0)
        x, y, h:    arrays of positions/h of all particles
        fact:       kernel support radius factor: W = 0 for r > fact*h
        L:          boxsize
        periodic:   Whether you assume periodic boundary conditions

        returns list of neighbour indices in x,y,h array
        """


        # None for Gaussian
        if fact is not None:

            x0 = x[ind]
            y0 = y[ind]
            fhsq = h[ind]*h[ind]*fact*fact
            neigh = [None for i in x]

            j = 0
            for i in range(x.shape[0]):
                if i == ind:
                    continue

                dx, dy = get_dx(x0, x[i], y0, y[i], L=L, periodic=periodic)

                dist = dx**2 + dy**2

                if dist < fhsq:
                    neigh[j] = i
                    j += 1

            return neigh[:j]

        else:
            neigh = [i for i in range(x.shape[0])]
            neigh.remove(ind)
            return neigh


   
    npart = x.shape[0]

    # find and store all neighbours;
    neighbours = [[] for i in x]
    for i in range(npart):
        neighbours[i] = find_neighbours_for_i_naive(i, x, y, h, fact=fact, L=L, periodic=periodic)


    # get neighbour counts array
    nneigh = np.zeros((npart), dtype=np.int)
    for i in range(npart):
        nneigh[i] = len(neighbours[i])


    return neighbours, nneigh





#=====================================================
def get_dx(x1, x2, y1, y2, L=1.0, periodic=True):
#=====================================================
    """
    Compute difference of vectors [x1 - x2, y1 - y2] while
    checking for periodicity if necessary
    L:          boxsize
    periodic:   whether to assume periodic boundaries
    """

    dx = x1 - x2
    dy = y1 - y2

    if periodic:

        Lhalf = 0.5*L

        if dx > Lhalf:
            dx -= L
        elif dx < -Lhalf:
            dx += L

        if dy > Lhalf:
            dy -= L
        elif dy < -Lhalf:
            dy += L


    return dx, dy






#==================================================================================
def plot_solution(x, y, h, part_to_plot, neigh, L=1.0, periodic=True):
#==================================================================================
    """
    Plot the solution: All particles, mark the chosen one and its neighbours

    x, y, h:    positions, compact support radius numpy arrays
    part_to_plot: particle INDEX to plot
    neigh:      list of lists of all neighbours for all particles
    L:          boxsize
    periodic:   periodic or not


    returns:    Nothing
    """


    # Set point parameters : Set circle centres
    xpart = x[part_to_plot]
    ypart = y[part_to_plot]
    rpart  = h[part_to_plot]
    xp = [xpart]
    yp = [ypart]
    if periodic:
        if xpart - rpart < 0:
            xp.append(xpart+L)
            yp.append(ypart)
        if xpart + rpart > L:
            xp.append(xpart-L)
            yp.append(ypart)
        if ypart - rpart < 0:
            xp.append(xpart)
            yp.append(ypart+L)
        if ypart + rpart > L:
            xp.append(xpart)
            yp.append(ypart-L)
        if xpart - rpart < 0 and ypart + rpart > L:
            xp.append(xpart+L)
            yp.append(ypart-L)
        if xpart - rpart < 0 and ypart - rpart < 0:
            xp.append(xpart+L)
            yp.append(ypart+L)
        if xpart + rpart > L and ypart + rpart > L:
            xp.append(xpart-L)
            yp.append(ypart-L)
        if xpart + rpart > L and ypart - rpart < 0:
            xp.append(xpart-L)
            yp.append(ypart+L)
        
    r = [rpart for i in xp] 


    plt.close('all') #safety measure
    fig = plt.figure(facecolor='white', figsize=(7,7))
    ax1 = fig.add_subplot(111, aspect='equal')


    # Plot Circle first

    #Plot the data without size; Markers will be resized later
    scat = ax1.scatter(xp,yp,s=0, alpha=0.5,clip_on=False, facecolor='grey')

    #Set axes edges
    ax1.set_xlim(0.00,L)
    ax1.set_ylim(0.00,L)  

    # Draw figure
    fig.canvas.draw()

    # Calculate radius in pixels :
    N=len(r)
    rr_pix = (ax1.transData.transform(np.vstack([r, r]).T) -
          ax1.transData.transform(np.vstack([np.zeros(N), np.zeros(N)]).T))
    rpix, _ = rr_pix.T
        
    
    # Calculate and update size in points:
    size_pt = (2*rpix/fig.dpi*72)**2
    scat.set_sizes(size_pt)



    # Plot all particles
    scat_p = ax1.scatter(x, y, s=8, facecolor='k')


    # Plot all neighbours
    scat_p = ax1.scatter(x[neigh[part_to_plot]], y[neigh[part_to_plot]], s=10, facecolor='green', alpha=0.6)
    
    # Plot chosen particle
    scat_p = ax1.scatter(x[part_to_plot], y[part_to_plot], s=10, facecolor='red', alpha=1.)



    # Plot cell boundaries
    ncells = int(L/h.max()) + 1
    cell_size = L/ncells

    for i in range(1, ncells+1):
        ax1.plot([i*cell_size, i*cell_size], [0, L], lw=0.5, c='darkgrey')
        ax1.plot([0, L], [i*cell_size, i*cell_size], lw=0.5, c='darkgrey')

    print("Warning: Manually resizing the figure changes the circle size.")
    plt.show()

    return
















#======================================================================
def compare_results(x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s, L):
#======================================================================
    """
    Compare the neighboursearch results element-by-element
    x, y, h:    np.arrays of positions, compact support lengths
    neigh_n, nneigh_n:  list of neighbours and numbers of neighbours for naive search
    neigh_s, nneigh_s:  list of neighbours and numbers of neighbours for smart search

    returns:
        nothing
    """

    print("Comparing naive and better results now.")

    npart = x.shape[0]

    ncells = int(L/h.max()) + 1
    cell_size = L/ncells

    found_difference = False

    for p in range(npart):
        nn = nneigh_n[p]
        ns = nneigh_s[p]
        if nn != ns:
            print("Got different number of neighbours for particle index", p, ":", nn, ns)
            print(neigh_n[p])
            print(neigh_s[p])

            if nn > ns:
                larger = neigh_n
                nl = nn
                smaller = neigh_s
                nsm = ns
                larger_is = 'naive'
            else:
                larger = neigh_s
                nl = ns
                smaller = neigh_n
                nsm = nn
                larger_is = 'smart'
                
            i = 0
            while i < nl:
                if larger[p][i] != smaller[p][i]:
                    problem = i
                    break
                i += 1

            xl = x[larger[p][problem]]
            yl = y[larger[p][problem]]
            Hl = h[larger[p][problem]]
            dxl, dyl = get_dx(xl, x[p], yl, y[p])
            rl = np.sqrt(dxl**2 + dyl**2)

            xS = x[smaller[p][problem]]
            yS = y[smaller[p][problem]]
            HS = h[smaller[p][problem]]
            dxS, dyS = get_dx(xS, x[p], yS, y[p])
            rS = np.sqrt(dxS**2 + dyS**2)
    
            ip = int(x[p]/cell_size)
            jp = int(y[p]/cell_size)
            iS = int(x[smaller[p][problem]]/cell_size)
            jS = int(y[smaller[p][problem]]/cell_size)
            il = int(x[larger[p][problem]]/cell_size)
            jl = int(y[larger[p][problem]]/cell_size)

            print("Larger is:", larger_is, " positions:")
            print("ind part:  {0:6d}  x: {1:14.7f}  y: {2:14.7f}  H: {3:14.7f}; i= {4:5d} j= {5:5d}".format(p, x[p], y[p], h[p], ip, jp))
            print("ind large: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}; i= {6:5d} j= {7:5d}".format(larger[p][problem], xl, yl, rl, Hl, rl/Hl, il, jl))
            print("ind small: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}; i= {6:5d} j= {7:5d}".format(smaller[p][problem], xS, yS, rS, HS, rS/HS, iS, jS))

            quit()

        for n in range(nneigh_n[p]):
            nn = neigh_n[p][n]
            ns = neigh_s[p][n]
            if nn != ns:
                print("Got different neighbour indexes:", nn, ns)
                print(neigh_n[p])
                print(neigh_s[p])
                found_difference = True

    if not found_difference:
        print("Found no difference.")







#=======================================
def compare_speeds():
#=======================================
    """
    Run the neighboursearch naively and with the better algorithm
    for multiple particle numbers, time the execution and plot it
    """

    nparts = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    print("Comparing speeds. This may take quite a while. (~ 10-15min)")

    np.random.seed(10)
    n = len(nparts)
    naives = np.zeros(n)
    better = np.zeros(n)

    L = 1.0
    periodic = True
    h_glob = 0.22

    for n, npart in enumerate(nparts):
        print("Running for npart=", npart)

        # first determine mean size of h so you roughly have
        # the same number of neighbours for every particle number

        h_glob = L / np.sqrt(np.pi * npart)

        x = np.random.random(size=npart)
        y = np.random.random(size=npart)
        h = np.ones(npart, dtype=np.float)*h_glob

        start_n = time.time()
        neigh_n, nneigh_n = get_neighbours_naive(x, y, h, L=L, periodic=periodic)
        stop_n = time.time()

        start_s = time.time()
        neigh_s, nneigh_s = get_neighbours(x, y, h, L=L, periodic=periodic)
        stop_s = time.time()

        tn = stop_n - start_n
        ts = stop_s - start_s
        print('{0:18} {1:18} {2:18}'.format("time naive", "time better", "naive/better"))
        print('{0:18.4f} {1:18.4f} {2:18.4f}'.format(tn, ts, tn/ts))

        naives[n] = tn
        better[n] = ts


    fig = plt.figure(figsize=(8,5))
 
    ax1 = fig.add_subplot(121)
    ax1.semilogx(nparts, naives, label="naive", c='C0')
    ax1.scatter(nparts, naives, c='C0', s=10)
    ax1.semilogx(nparts, better, label="better", c='C1')
    ax1.scatter(nparts, better, c='C1', s=10)
    ax1.set_ylabel('s')

    ax2 = fig.add_subplot(122)
    ax2.semilogx(nparts, naives/better,c='C0', label="naive/better")
    ax2.scatter(nparts, naives/better, c='C0', s=10)


    for ax in fig.axes:
        ax.grid()
        ax.legend()
        ax.set_xlabel('nparts')

    #  plt.show()
    plt.tight_layout()
    plt.savefig("speed_comparison_neighboursearch.png", dpi=200)





if __name__ == '__main__':
    main()
