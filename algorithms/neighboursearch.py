#!/usr/bin/env python3

# =====================================================================
# Perform a neighbour search for particles given a maximal distance
# for a neighbour, where the distance is individual to each particle.
# Cross compare with a naive N^2 search, time it, and plot it.
# Do it for 2D only. Extension to 3D is trivial.
# =====================================================================


import numpy as np
import matplotlib.pyplot as plt
import time


# ====================================
def main():
    # ====================================

    # invent some data
    npart = 1000
    L = [1.0, 2.0]
    periodic = True
    h_glob = 0.10

    np.random.seed(10)
    x = np.random.random(size=npart) * L[0]
    y = np.random.random(size=npart) * L[1]
    h = np.ones(npart, dtype=np.float) * h_glob

    # if you want to manually set a particle, set it here
    part_to_plot = 0
    x[0] = 0.99
    y[0] = 0.04

    # get neighbours, and time it
    start_n = time.time()
    neigh_n, nneigh_n = get_neighbours_naive(x, y, h, L=L, periodic=periodic)
    stop_n = time.time()

    start_s = time.time()
    #  neigh_s, nneigh_s = get_neighbours_square(x, y, h, L=L, periodic=periodic)
    neigh_s, nneigh_s = get_neighbours_rect(x, y, h, L=L, periodic=periodic)
    stop_s = time.time()

    tn = stop_n - start_n
    ts = stop_s - start_s
    print("{0:18} {1:18} {2:18}".format("time naive", "time better", "naive/better"))
    print("{0:18.4f} {1:18.4f} {2:18.4f}".format(tn, ts, tn / ts))

    #  check that results are identical
    compare_results(x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s, L)

    # plot the neighbours of chosen particle
    plot_solution(x, y, h, part_to_plot, neigh_s, L=L, periodic=periodic)

    # run a speed comparison
    #  compare_speeds()

    return


# ========================================================================
def get_neighbours_rect(x, y, h, fact=1.0, L=[1.0, 1.0], periodic=True):
    # ========================================================================
    """
    Gets all the neighbour data for all particles ready.
    Assumes domain is a rectangle with boxsize L[0], L[1].
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize. List/Array or scalar.
    periodic:   Whether you assume periodic boundary conditions

    returns:
        neighbours:     list of lists of all neighbours for each particle
        nneigh:         numpy array of now many neighbours each particle has


    """

    # if it isn't in a list already, create one
    # do this before function/class definition
    if not hasattr(L, "__len__"):
        L = [L, L]

    # -----------------------------------------------
    class cell:
        # -----------------------------------------------
        """
        A cell object to store particles in.
        Stores particle indexes, positions, compact support radii
        """

        def __init__(self):
            self.npart = 0
            self.size = 100
            self.parts = np.zeros(self.size, dtype=np.int)
            self.x = np.zeros(self.size, dtype=np.float)
            self.y = np.zeros(self.size, dtype=np.float)
            self.h = np.zeros(self.size, dtype=np.float)
            self.xmin = 1e300
            self.xmax = -1e300
            self.ymin = 1e300
            self.ymax = -1e300
            self.hmax = -1e300
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

            if self.xmax < xp:
                self.xmax = xp
            if self.xmin > xp:
                self.xmin = xp
            if self.ymax < yp:
                self.ymax = yp
            if self.ymin > yp:
                self.ymin = yp
            if self.hmax < hp:
                self.hmax = hp

            return

        def is_within_h(self, xp, yp, hp):
            """
            Check whether any particle of this cell is within
            compact support of particle with x, y, h = xp, yp, hp
            """
            dx1, dy1 = get_dx(xp, self.xmax, yp, self.ymax, L=L, periodic=periodic)
            dx2, dy2 = get_dx(xp, self.xmin, yp, self.ymin, L=L, periodic=periodic)
            dxsq = min(dx1 * dx1, dx2 * dx2)
            dysq = min(dy1 * dy1, dy2 * dy2)
            if dxsq / hp ** 2 <= 1 or dysq / hp ** 2 <= 1:
                return True
            else:
                return False

    # ------------------------------------------------------------
    def find_neighbours_in_cell(i, j, p, xx, yy, hh, is_self):
        # ------------------------------------------------------------
        """
        Find neighbours of a particle in the cell with indices i,j
        of the grid
        p:      global particle index to work with
        xx, yy: position of particle x
        hh:     compact support radius for p
        is_self: whether this is the cell where p is in anyway
        """
        n = 0
        neigh = [0 for i in range(1000)]
        ncell = grid[i][j]  # neighbour cell we're checking for

        if not is_self:
            if not ncell.is_within_h(xx, yy, hh):
                return []

        N = ncell.npart

        fhsq = hh * hh * fact * fact

        for c, cp in enumerate(ncell.parts[:N]):
            if cp == p:
                # skip yourself
                continue

            dx, dy = get_dx(xx, ncell.x[c], yy, ncell.y[c], L=L, periodic=periodic)

            dist = dx ** 2 + dy ** 2

            if dist <= fhsq:
                try:
                    neigh[n] = cp
                except IndexError:
                    nneigh += [0 for i in range(1000)]
                    nneigh[n] = cp
                n += 1

        return neigh[:n]

    npart = x.shape[0]

    # first find cell size
    ncells_x = int(L[0] / h.max()) + 1
    ncells_y = int(L[1] / h.max()) + 1
    cell_size_x = L[0] / ncells_x
    cell_size_y = L[1] / ncells_y

    # create grid
    grid = [[cell() for j in range(ncells_y)] for i in range(ncells_x)]

    # sort out particles
    for p in range(npart):
        i = int(x[p] / cell_size_x)
        j = int(y[p] / cell_size_y)
        grid[i][j].add_particle(p, x[p], y[p], h[p])

    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)

    # main loop: find and store all neighbours;
    # go cell by cell
    for row in range(ncells_y):
        for col in range(ncells_x):

            cell = grid[col][row]
            N = cell.npart
            parts = cell.parts
            if N == 0:
                continue

            hmax = cell.h[:N].max()

            # find over how many cells to loop in every direction
            maxdistx = int(cell_size_x / hmax + 0.5) + 1
            maxdisty = int(cell_size_y / hmax + 0.5) + 1

            xstart = -maxdistx
            xstop = maxdistx + 1
            ystart = -maxdisty
            ystop = maxdisty + 1

            # exception handling: if ncells < 4, just loop over
            # all of them so that you don't add neighbours multiple
            # times
            if ncells_x < 4:
                xstart = 0
                xstop = ncells_x
            if ncells_y < 4:
                ystart = 0
                ystop = ncells_y

            checked_cells = [
                (None, None) for i in range((2 * maxdistx + 1) * (2 * maxdisty + 1))
            ]
            it = 0

            # loop over all neighbours
            # need to loop over entire square. You need to consider
            # the maximal distance from the edges/corners, not from
            # the center of the cell!
            for i in range(xstart, xstop):
                for j in range(ystart, ystop):

                    if ncells_x < 4:
                        iind = i
                    else:
                        iind = col + i

                    if ncells_y < 4:
                        jind = j
                    else:
                        jind = row + j

                    if periodic:
                        while iind < 0:
                            iind += ncells_x
                        while iind >= ncells_x:
                            iind -= ncells_x
                        while jind < 0:
                            jind += ncells_y
                        while jind >= ncells_y:
                            jind -= ncells_y
                    else:
                        if iind < 0 or iind >= ncells_x:
                            continue
                        if jind < 0 or jind >= ncells_y:
                            continue

                    it += 1
                    if (iind, jind) in checked_cells[: it - 1]:
                        continue
                    else:
                        checked_cells[it - 1] = (iind, jind)

                    # loop over all particles in THIS cell
                    for pc, pg in enumerate(cell.parts[:N]):

                        xp = cell.x[pc]
                        yp = cell.y[pc]
                        hp = cell.h[pc]

                        neighbours[pg] += find_neighbours_in_cell(
                            iind, jind, pg, xp, yp, hp, (iind == col and jind == row)
                        )

    # sort neighbours by index
    for p in range(npart):
        neighbours[p].sort()
        nneigh[p] = len(neighbours[p])

    return neighbours, nneigh


# ========================================================================
def get_neighbours_square(x, y, h, fact=1.0, L=1.0, periodic=True):
    # ========================================================================
    """
    Gets all the neighbour data for all particles ready.
    Assumes domain is a square with boxsize L.
    x, y, h:    arrays of positions/h of all particles
    fact:       kernel support radius factor: W = 0 for r > fact*h
    L:          boxsize
    periodic:   Whether you assume periodic boundary conditions

    returns:
        neighbours:     list of lists of all neighbours for each particle
        nneigh:         numpy array of now many neighbours each particle has


    """

    if hasattr(L, "__len__"):
        if L[0] == L[1]:
            L = L[0]
        else:
            print("Have different box dimensions:", L[0], "and", L[1])
            print("Can't work like this")
            quit()

    # -----------------------------------------------
    class cell:
        # -----------------------------------------------
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

    # -------------------------------------------------------
    def find_neighbours_in_cell(i, j, p, xx, yy, hh):
        # -------------------------------------------------------
        """
        Find neighbours of a particle in the cell with indices i,j
        of the grid
        p:      global particle index to work with
        xx, yy: position of particle x
        hh:     compact support radius for p
        """
        n = 0
        neigh = [0 for i in range(1000)]
        ncell = grid[i][j]  # neighbour cell we're checking for

        N = ncell.npart

        fhsq = hh * hh * fact * fact

        for c, cp in enumerate(ncell.parts[:N]):
            if cp == p:
                # skip yourself
                continue

            dx, dy = get_dx(xx, ncell.x[c], yy, ncell.y[c], L=L, periodic=periodic)

            dist = dx ** 2 + dy ** 2

            if dist <= fhsq:
                try:
                    neigh[n] = cp
                except ValueError:
                    nneigh += [0 for i in range(1000)]
                    nneigh[n] = cp
                n += 1

        return neigh[:n]

    npart = x.shape[0]

    # first find cell size
    ncells = int(L / h.max()) + 1
    cell_size = L / ncells
    #  print("ncells is", ncells)
    #  print("cell size is", cell_size)

    # create grid
    grid = [[cell() for i in range(ncells)] for j in range(ncells)]

    # sort out particles
    for p in range(npart):
        i = int(x[p] / cell_size)
        j = int(y[p] / cell_size)
        grid[i][j].add_particle(p, x[p], y[p], h[p])

    neighbours = [[] for i in x]
    nneigh = np.zeros(npart, dtype=np.int)

    if ncells < 4:
        # you'll always need to check all cells, so just do that
        i_search = np.zeros(ncells * ncells, dtype=np.int)
        j_search = np.zeros(ncells * ncells, dtype=np.int)
        ind = 0
        for i in range(ncells):
            for j in range(ncells):
                i_search[ind] = i
                j_search[ind] = j
                ind += 1

        # main loop: find and store all neighbours:
        for p in range(npart):
            nbors = []
            for i, j in zip(i_search, j_search):
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
            if N == 0:
                continue

            hmax = cell.h[:N].max()

            maxdist = int(cell_size / hmax + 0.5) + 1

            # loop over all neighbours
            # need to loop over entire square. You need to consider
            # the maximal distance from the edges/corners, not from
            # the center of the cell!
            for i in range(-maxdist, maxdist + 1):
                for j in range(-maxdist, maxdist + 1):
                    iind = row + i
                    jind = col + j

                    if periodic:
                        while iind < 0:
                            iind += ncells
                        while iind >= ncells:
                            iind -= ncells
                        while jind < 0:
                            jind += ncells
                        while jind >= ncells:
                            jind -= ncells
                    else:
                        if iind < 0 or iind >= ncells:
                            continue
                        if jind < 0 or jind >= ncells:
                            continue

                    # get closest corner of neighbour
                    ic = iind
                    if i < 0:
                        ic += 1
                    jc = jind
                    if j < 0:
                        jc += 1

                    xc = ic * cell_size
                    yc = jc * cell_size

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
                        dsq = dx ** 2 + dy ** 2
                        if dsq / hp ** 2 > 1:
                            continue

                        neighbours[pg] += find_neighbours_in_cell(
                            iind, jind, pg, xp, yp, hp
                        )

    # sort neighbours by index
    for p in range(npart):
        neighbours[p].sort()
        nneigh[p] = len(neighbours[p])

    return neighbours, nneigh


# ===============================================================================
def get_neighbours_naive(x, y, h, fact=1.0, L=1.0, periodic=True):
    # ===============================================================================
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

    # --------------------------------------------------------------------------------
    def find_neighbours_for_i_naive(ind, x, y, h, fact=1.0, L=1.0, periodic=True):
        # --------------------------------------------------------------------------------
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
            fhsq = h[ind] * h[ind] * fact * fact
            neigh = [None for i in x]

            j = 0
            for i in range(x.shape[0]):
                if i == ind:
                    continue

                dx, dy = get_dx(x0, x[i], y0, y[i], L=L, periodic=periodic)

                dist = dx ** 2 + dy ** 2

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
        neighbours[i] = find_neighbours_for_i_naive(
            i, x, y, h, fact=fact, L=L, periodic=periodic
        )

    # get neighbour counts array
    nneigh = np.zeros((npart), dtype=np.int)
    for i in range(npart):
        nneigh[i] = len(neighbours[i])

    return neighbours, nneigh


# =====================================================
def get_dx(x1, x2, y1, y2, L=1.0, periodic=True):
    # =====================================================
    """
    Compute difference of vectors [x1 - x2, y1 - y2] while
    checking for periodicity if necessary
    L:          boxsize. Scalar or array_like
    periodic:   whether to assume periodic boundaries
    """

    dx = x1 - x2
    dy = y1 - y2

    if periodic:

        if hasattr(L, "__len__"):
            Lxhalf = L[0] / 2.0
            Lyhalf = L[1] / 2.0
        else:
            Lxhalf = L / 2.0
            Lyhalf = L / 2.0
            L = [L, L]

        if dx > Lxhalf:
            dx -= L[0]
        elif dx < -Lxhalf:
            dx += L[0]

        if dy > Lyhalf:
            dy -= L[1]
        elif dy < -Lyhalf:
            dy += L[1]

    return dx, dy


# ==================================================================================
def plot_solution(x, y, h, part_to_plot, neigh, L=1.0, periodic=True):
    # ==================================================================================
    """
    Plot the solution: All particles, mark the chosen one and its neighbours

    x, y, h:    positions, compact support radius numpy arrays
    part_to_plot: particle INDEX to plot
    neigh:      list of lists of all neighbours for all particles
    L:          boxsize
    periodic:   periodic or not


    returns:    Nothing
    """

    if not hasattr(L, "__len__"):
        L = [L, L]

    # Set point parameters : Set circle centres
    xpart = x[part_to_plot]
    ypart = y[part_to_plot]
    rpart = h[part_to_plot]
    xp = [xpart]
    yp = [ypart]
    if periodic:
        if xpart - rpart < 0:
            xp.append(xpart + L[0])
            yp.append(ypart)
        if xpart + rpart > L[0]:
            xp.append(xpart - L[0])
            yp.append(ypart)
        if ypart - rpart < 0:
            xp.append(xpart)
            yp.append(ypart + L[1])
        if ypart + rpart > L[1]:
            xp.append(xpart)
            yp.append(ypart - L[1])
        if xpart - rpart < 0 and ypart + rpart > L[1]:
            xp.append(xpart + L[0])
            yp.append(ypart - L[1])
        if xpart - rpart < 0 and ypart - rpart < 0:
            xp.append(xpart + L[0])
            yp.append(ypart + L[1])
        if xpart + rpart > L[0] and ypart + rpart > L[1]:
            xp.append(xpart - L[0])
            yp.append(ypart - L[1])
        if xpart + rpart > L[0] and ypart - rpart < 0:
            xp.append(xpart - L[0])
            yp.append(ypart + L[1])

    r = [rpart for i in xp]

    plt.close("all")  # safety measure
    fig = plt.figure(facecolor="white", figsize=(7, 7))
    ax1 = fig.add_subplot(111, aspect="equal")

    # Plot Circle first

    # Plot the data without size; Markers will be resized later
    scat = ax1.scatter(xp, yp, s=0, alpha=0.5, clip_on=False, facecolor="grey")

    # Set axes edges
    ax1.set_xlim(0.00, L[0])
    ax1.set_ylim(0.00, L[1])

    # Draw figure
    fig.canvas.draw()

    # Calculate radius in pixels :
    N = len(r)
    rr_pix = ax1.transData.transform(np.vstack([r, r]).T) - ax1.transData.transform(
        np.vstack([np.zeros(N), np.zeros(N)]).T
    )
    rpix, _ = rr_pix.T

    # Calculate and update size in points:
    size_pt = (2 * rpix / fig.dpi * 72) ** 2
    scat.set_sizes(size_pt)

    # Plot all particles
    scat_p = ax1.scatter(x, y, s=8, facecolor="k")

    # Plot all neighbours
    scat_p = ax1.scatter(
        x[neigh[part_to_plot]],
        y[neigh[part_to_plot]],
        s=10,
        facecolor="green",
        alpha=0.6,
    )

    # Plot chosen particle
    scat_p = ax1.scatter(
        x[part_to_plot], y[part_to_plot], s=10, facecolor="red", alpha=1.0
    )

    # Plot cell boundaries
    ncells_x = int(L[0] / h.max()) + 1
    cell_size_x = L[0] / ncells_x
    ncells_y = int(L[1] / h.max()) + 1
    cell_size_y = L[1] / ncells_y

    for i in range(1, ncells_x + 1):
        ax1.plot([i * cell_size_x, i * cell_size_x], [0, L[1]], lw=0.5, c="darkgrey")
    for j in range(1, ncells_y + 1):
        ax1.plot([0, L[0]], [j * cell_size_y, j * cell_size_y], lw=0.5, c="darkgrey")

    print("Warning: Manually resizing the figure changes the circle size.")
    plt.show()

    return


# ===============================================================================================
def compare_results(
    x, y, h, neigh_n, nneigh_n, neigh_s, nneigh_s, L=1.0, periodic=True
):
    # ===============================================================================================
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

    if not hasattr(L, "__len__"):
        L = [L, L]

    ncells_x = int(L[0] / h.max()) + 1
    ncells_y = int(L[1] / h.max()) + 1
    cell_size_x = L[0] / ncells_x
    cell_size_y = L[1] / ncells_y

    found_difference = False

    for p in range(npart):
        nn = nneigh_n[p]
        ns = nneigh_s[p]
        if nn != ns:
            print(
                "Got different number of neighbours for particle index", p, ":", nn, ns
            )
            print(neigh_n[p])
            print(neigh_s[p])

            if nn > ns:
                larger = neigh_n
                nl = nn
                smaller = neigh_s
                nsm = ns
                larger_is = "naive"
            else:
                larger = neigh_s
                nl = ns
                smaller = neigh_n
                nsm = nn
                larger_is = "smart"

            i = 0
            while i < nl:
                if larger[p][i] != smaller[p][i]:
                    problem = i
                    break
                i += 1

            xl = x[larger[p][problem]]
            yl = y[larger[p][problem]]
            Hl = h[larger[p][problem]]
            dxl, dyl = get_dx(xl, x[p], yl, y[p], L=L, periodic=periodic)
            rl = np.sqrt(dxl ** 2 + dyl ** 2)

            xS = x[smaller[p][problem]]
            yS = y[smaller[p][problem]]
            HS = h[smaller[p][problem]]
            dxS, dyS = get_dx(xS, x[p], yS, y[p], L=L, periodic=periodic)
            rS = np.sqrt(dxS ** 2 + dyS ** 2)

            ip = int(x[p] / cell_size_x)
            jp = int(y[p] / cell_size_y)
            iS = int(x[smaller[p][problem]] / cell_size_x)
            jS = int(y[smaller[p][problem]] / cell_size_y)
            il = int(x[larger[p][problem]] / cell_size_x)
            jl = int(y[larger[p][problem]] / cell_size_y)

            print("Larger is:", larger_is, " positions:")
            print(
                "ind part:  {0:6d}  x: {1:14.7f}  y: {2:14.7f}  H: {3:14.7f}; i= {4:5d} j= {5:5d}".format(
                    p, x[p], y[p], h[p], ip, jp
                )
            )
            print(
                "ind large: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}; i= {6:5d} j= {7:5d}".format(
                    larger[p][problem], xl, yl, rl, Hl, rl / Hl, il, jl
                )
            )
            print(
                "ind small: {0:6d}  x: {1:14.7f}  y: {2:14.7f}  r: {3:14.7f}  H: {4:14.7f} r/H: {5:14.7f}; i= {6:5d} j= {7:5d}".format(
                    smaller[p][problem], xS, yS, rS, HS, rS / HS, iS, jS
                )
            )

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

    return


# =======================================
def compare_speeds():
    # =======================================
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
        h = np.ones(npart, dtype=np.float) * h_glob

        start_n = time.time()
        neigh_n, nneigh_n = get_neighbours_naive(x, y, h, L=L, periodic=periodic)
        stop_n = time.time()

        start_s = time.time()
        neigh_s, nneigh_s = get_neighbours(x, y, h, L=L, periodic=periodic)
        stop_s = time.time()

        tn = stop_n - start_n
        ts = stop_s - start_s
        print(
            "{0:18} {1:18} {2:18}".format("time naive", "time better", "naive/better")
        )
        print("{0:18.4f} {1:18.4f} {2:18.4f}".format(tn, ts, tn / ts))

        naives[n] = tn
        better[n] = ts

    fig = plt.figure(figsize=(8, 5))

    ax1 = fig.add_subplot(121)
    ax1.semilogx(nparts, naives, label="naive", c="C0")
    ax1.scatter(nparts, naives, c="C0", s=10)
    ax1.semilogx(nparts, better, label="better", c="C1")
    ax1.scatter(nparts, better, c="C1", s=10)
    ax1.set_ylabel("s")

    ax2 = fig.add_subplot(122)
    ax2.semilogx(nparts, naives / better, c="C0", label="naive/better")
    ax2.scatter(nparts, naives / better, c="C0", s=10)

    for ax in fig.axes:
        ax.grid()
        ax.legend()
        ax.set_xlabel("nparts")

    #  plt.show()
    plt.tight_layout()
    plt.savefig("speed_comparison_neighboursearch.png", dpi=200)


if __name__ == "__main__":
    main()
