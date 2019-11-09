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
    npart = 200
    L = 1.0
    periodic = True
    h_glob = 0.22
    part_to_plot = 0

    np.random.seed(10)
    x = np.random.random(size=npart) * L
    y = np.random.random(size=npart) * L
    h = np.ones(npart, dtype=np.float)*h_glob

    x[0] = 0.05
    y[0] = 0.6



    # get neighbours
    start_n = time.time()
    neigh_n, nneigh_n = get_neighbours_naive(x, y, h, L=L, periodic=periodic)
    stop_n = time.time()

    start_s = time.time()
    neigh_s, nneigh_s = get_neighbours(x, y, h, L=L, periodic=periodic)
    stop_s = time.time()


    # check that results are identical
    compare_results(x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s)

    print()
    print('{0:18} {1:18} {2:18}'.format("time naive", "time better", "|1 - naive/better|"))
    tn = stop_n - start_n
    ts = stop_s - start_s
    print('{0:18.4f} {1:18.4f} {2:18.4f}'.format(tn, ts, abs(1 - tn/ts)))




    plot_solution(x, y, h, part_to_plot, neigh_s, L=L, periodic=periodic)


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

    partcheck = 1


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
           return

        def add_particle(self, ind):
            """
            Add a particle, store the index
            """
            if self.npart == self.size:
                self.parts = np.append(self.parts, np.zeros(self.size, dtype=np.int))
                self.size *= 2

            self.parts[self.npart] = ind
            self.npart += 1
            
            return


    
    #-----------------------------------------------
    def find_neighbours_in_cell(i, j, p):
    #-----------------------------------------------
        """
        Find neighbours of particles with index p in the cell
        i, j of the grid
        """
        n = 0
        neigh = [0 for i in range(1000)]

        if p==partcheck:
            print("checking in function p=", p,"i,j=", i, j)

        np = grid[i][j].npart
        parts = grid[i][j].parts
        
        xp = x[p]
        yp = y[p]
        hp = h[p]
        fhsq = hp*hp*fact*fact

        for cp in parts[:np]:
            if cp == p:
                continue

            dx, dy = get_dx(xp, x[cp], yp, y[cp], L=L, periodic=periodic)

            dist = dx**2 + dy**2

            if dist <= fhsq:
                neigh[n] = cp
                n += 1

        return neigh[:n]







    npart = x.shape[0]

    # first find cell size
    cell_size = 2*h.max()
    ncells = int(L/cell_size) + 1

    # store cell boundaries
    cell_lower_boundaries = [i*cell_size for i in range(ncells)]
    cell_upper_boundaries = [(i+1)*cell_size for i in range(ncells)]
    cell_upper_boundaries[-1] = L
    print("Upper boundaries:", cell_upper_boundaries)
    print("Lower boundaries:", cell_lower_boundaries)

    # create grid
    grid = [[cell() for i in range(ncells)] for j in range(ncells)]


    # sort out particles
    for p in range(npart):
        i = int(x[p]/cell_size)
        j = int(y[p]/cell_size)
        if p == 1:
            print("p=1 is in", i, j)
        if p==52:
            print("p=52 is in", i, j)
        grid[i][j].add_particle(p)



    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)

    # main loop: find and store all neighbours;
    for p in range(npart):
    #  for p in [9]:
 
        self_i = int(x[p]/cell_size)
        self_j = int(y[p]/cell_size)

        H = h[p]
   
   
        # find what work needs to be done, and find neighbour indices
        # also store whether you went over the boundary
        do_upper = cell_upper_boundaries[self_i] - x[p] < H
        do_right = cell_upper_boundaries[self_j] - y[p] < H
        if ncells > 2: 
            # only do this if you have at least 3 cells.
            # otherwise, left = right, and you're doing work twice.
            do_lower = x[p] - cell_lower_boundaries[self_i] < H
            do_left  = y[p] - cell_lower_boundaries[self_j] < H
        else:
            do_lower = False
            do_left = False

        over_upper_boundary = False
        over_left_boundary = False
        over_lower_boundary = False
        over_right_boundary = False

        if p == partcheck:
            print("----------------------------------------------------------")
            print(" working for p=", p, "i=", x[p]//cell_size, "j=", y[p]//cell_size)
            print("upper", do_upper, "lower", do_lower)
            print("left:", do_left, "right:", do_right)

        # check whether you need to check another cell in the direction
        # you're checking. The last cell has size < cell_size so that
        # the others can have 2*H_max consistently.
        # There are 2 cases to consider:
        # If you are inside the grid, and the neighbour is the last cell
        # (upper or right neighbour): Then you need to check the first
        # cell in the grid, provided you're doing a periodic space.
        # If you are on the grid border (left or lower), then your
        # neighbour is the last cell in the grid. Then you need to check
        # one more cell inwards, i.e. the second-to-last cell.

        if do_upper:
            upper_i = self_i + 1
            if upper_i == ncells:
                if periodic:
                    upper_i -= ncells
                else:
                    do_upper = False
            if upper_i == ncells - 1:
                if periodic:
                    over_upper_boundary = True

        if do_lower:
            lower_i = self_i - 1
            if lower_i == -1:
                if periodic:
                    lower_i += ncells
                else:
                    do_lower = False
            if lower_i == ncells-1:
                if periodic:
                    over_lower_boundary = True

        if do_right:
            right_j = self_j + 1
            if right_j == ncells:
                if periodic:
                    right_j -= ncells
                else:
                    do_right = False
            if right_j == ncells - 1:
                if periodic:
                    over_right_boundary = True

        if do_left:
            left_j = self_j - 1
            if left_j == -1:
                if periodic:
                    left_j += ncells
                else:
                    do_left = False
            if left_j == ncells -1:
                if periodic:
                    over_left_boundary = True

        if ncells < 3:
            # if it's less than 3 cells, you're re-doing work if you're
            # re-checking one more cell after the boundary
            over_upper_boundary = False
            over_left_boundary = False
            over_lower_boundary = False
            over_right_boundary = False


        if p==partcheck:
            print("over upper:", over_upper_boundary)
            print("over lower:", over_lower_boundary)
            print("over left:", over_left_boundary)
            print("over right:", over_right_boundary)



        # now get all neighbours

        nbors = find_neighbours_in_cell(self_i, self_j, p) 

        # upper and lower neighbours
        if do_upper:
            if p==partcheck: print("check1")
            nbors += find_neighbours_in_cell(upper_i, self_j, p)
            #  print("upper", nbors)

        if do_lower:
            if p==partcheck: print("check2")
            nbors += find_neighbours_in_cell(lower_i, self_j, p)
            #  print("lower", nbors)

        if do_right:
            if p==partcheck: print("check3")
            nbors += find_neighbours_in_cell(self_i, right_j, p)
            #  print("right", nbors)

        if do_left:
            if p==partcheck: print("check4")
            nbors += find_neighbours_in_cell(self_i, left_j, p)
            #  print("left", nbors)

        # diagonal neighbours
        if do_upper:
            if do_right:
                if p==partcheck: print("check5")
                nbors += find_neighbours_in_cell(upper_i, right_j, p)
            if do_left:
                if p==partcheck: print("check6")
                nbors += find_neighbours_in_cell(upper_i, left_j, p)
        if do_lower:
            if do_right:
                if p==partcheck: print("check7")
                nbors += find_neighbours_in_cell(lower_i, right_j, p)
            if do_left:
                if p==partcheck: print("check8")
                nbors += find_neighbours_in_cell(lower_i, left_j, p)



        if periodic:
            # last cell doesn't have equal witdth as the others.
            # check cells one further up if you went over the boundary

            if over_upper_boundary:
                if p==partcheck: print("check9")
                nbors += find_neighbours_in_cell(0, self_j, p)
                if do_right:
                    if p==partcheck: print("check10")
                    nbors += find_neighbours_in_cell(0, right_j, p)
                    if over_right_boundary:
                        if p==partcheck: print("check11")
                        nbors += find_neighbours_in_cell(0, 0, p)
                if do_left:
                    if p==partcheck: print("check12")
                    nbors += find_neighbours_in_cell(0, left_j, p)
                    if over_left_boundary:
                        if p==partcheck: print("check13")
                        nbors += find_neighbours_in_cell(0, ncells-2, p)

            if over_lower_boundary:
                if p==partcheck: print("check14")
                nbors += find_neighbours_in_cell(ncells-2, self_j, p)
                if do_right:
                    if p==partcheck: print("check15")
                    nbors += find_neighbours_in_cell(ncells-2, right_j, p)
                    if over_right_boundary:
                        if p==partcheck: print("check16")
                        nbors += find_neighbours_in_cell(ncells-2, 0, p)
                if do_left:
                    if p==partcheck: print("check17")
                    nbors += find_neighbours_in_cell(ncells-2, left_j, p)
                    if over_left_boundary:
                        if p==partcheck: print("check18")
                        nbors += find_neighbours_in_cell(ncells-2, ncells-1, p)

            if over_left_boundary:
                if p==partcheck: print("check19")
                nbors += find_neighbours_in_cell(self_i, ncells-2, p)
                if do_upper:
                    if p==partcheck: print("check20")
                    nbors += find_neighbours_in_cell(upper_i, ncells-2, p)
                if do_lower:
                    if p==partcheck: print("check21")
                    nbors += find_neighbours_in_cell(lower_i, ncells-2, p)
                # the over_upper/lower_boundaries cases have been checked above already

            if over_right_boundary:
                if p==partcheck: print("check22")
                nbors += find_neighbours_in_cell(self_i, 0, p)
                if do_upper:
                    if p==partcheck: print("check23")
                    nbors += find_neighbours_in_cell(upper_i, 0, p)
                if do_lower:
                    if p==partcheck: print("check24")
                    nbors += find_neighbours_in_cell(lower_i, 0, p)
                # the over_upper/lower_boundaries cases have been checked above already

        neighbours[p] = sorted(nbors)
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
    cell_size = 2*h.max()
    ncells = int(L/cell_size) + 1

    for i in range(1, ncells):
        ax1.plot([i*cell_size, i*cell_size], [0, L], lw=0.5, c='darkgrey')
        ax1.plot([0, L], [i*cell_size, i*cell_size], lw=0.5, c='darkgrey')

    print("Warning: Manually resizing the figure changes the circle size.")
    plt.show()

    return
















#======================================================================
def compare_results(x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s):
#======================================================================
    """
    Compare the neighboursearch results element-by-element
    x, y, h:    np.arrays of positions, compact support lengths
    neigh_n, nneigh_n:  list of neighbours and numbers of neighbours for naive search
    neigh_s, nneigh_s:  list of neighbours and numbers of neighbours for smart search

    returns:
        nothing
    """

    npart = x.shape[0]

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
    

            print("Larger is:", larger_is, " positions:")
            print("ind part:  {0:6d}  x: {1:14.7f}  y: {2:14.7f}  H: {3:14.7f}".format(p, x[p], y[p], h[p]))
            print("ind large: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}".format(larger[p][problem], xl, yl, rl, Hl, rl/Hl))
            print("ind small: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}".format(smaller[p][problem], xS, yS, rS, HS, rS/HS))
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







if __name__ == '__main__':
    main()
