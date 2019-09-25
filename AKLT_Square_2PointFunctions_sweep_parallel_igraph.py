# Import necessary packages
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import igraph as ig
from scipy import stats
from scipy import linalg as LA
from scipy.optimize import curve_fit
from scipy.special import gammaln, hyp2f1
from itertools import combinations, product
import multiprocessing as mp
import os
import time



def PlotGraph(G):
    # Plots graph G with appropriate positions of vertices

    layout = zip(G.vs["x"], -1*np.array(G.vs["y"]))
    pl = ig.plot(G, layout=layout)
    pl.show()

def ConstructSquareLattices(N_x, N_y, w, h):
    # Constructs cylindrical square lattices

    # Square lattice
    Square_Lattice = ig.Graph()     # VERTICES  ARE NOW INDEXED FROM 0
    x_squares = np.arange(0.0, w * N_x, w)
    y_squares = 0 * h * np.ones(N_x)
    y_squares[-1] = y_squares[-1] + 0.35 * h
    Square_Lattice.add_vertices(N_x)
    Square_Lattice.vs["x"] = x_squares[0:N_x]
    Square_Lattice.vs["y"] = y_squares[0:N_x]
    Square_Lattice.add_edges(zip(range(0, N_x-1), range(1, N_x)))
    Square_Lattice.add_edge(0, N_x-1)
    Square_Lattice.es["weight"] = np.ones(N_x)
    for rung in range(2, N_y + 1 + 1):
        x_squares = np.append(x_squares, np.arange(0.0, w*N_x, w))
        y_squares = np.append(y_squares, (rung - 1)*h*np.ones(N_x))
        y_squares[-1] = y_squares[-1] + 0.35*h
        Square_Lattice.add_vertices(N_x)
        Square_Lattice.vs.select(range((rung - 1)*N_x, rung*N_x))["x"] = x_squares[(rung - 1)*N_x:rung*N_x]
        Square_Lattice.vs.select(range((rung - 1)*N_x, rung*N_x))["y"] = y_squares[(rung - 1)*N_x:rung*N_x]
        Square_Lattice.add_edges(zip(range((rung - 1)*N_x, rung*N_x-1), range((rung - 1)*N_x + 1, rung*N_x)))
        Square_Lattice.add_edge((rung - 1)*N_x, rung*N_x - 1)
        Square_Lattice.es.select(range((rung - 1)*N_x, rung*N_x))["weight"] = np.ones(N_x)
        for ladder in range(0, N_x):
            Square_Lattice.add_edge(((rung - 1)*N_x) + ladder, ((rung - 2)*N_x) + ladder, weight=1)
    x_squares = np.append(x_squares, np.arange(0.0, w*N_x, w))
    y_squares = np.append(y_squares, -1*h*np.ones(N_x))
    y_squares[-1] = y_squares[-1] + 0.35*h
    Square_Lattice.add_vertices(N_x)
    Square_Lattice.vs.select(range((N_y + 1)*N_x, (N_y + 2)*N_x))["x"] = x_squares[(N_y + 1)*N_x: (N_y + 2)*N_x]
    Square_Lattice.vs.select(range((N_y + 1)*N_x, (N_y + 2)*N_x))["y"] = y_squares[(N_y + 1)*N_x: (N_y + 2)*N_x]
    Square_Lattice.add_edges(zip(range(0, N_x), range(N_x*(N_y + 1), N_x*(N_y + 2))))
    Square_Lattice.es.select(range((N_y + 1)*N_x, (N_y + 2)*N_x))["weight"] = np.ones(N_x)

    # Square lattice with no edges
    Square_Lattice_Bare = Square_Lattice.copy()
    Square_Lattice_Bare.delete_edges(range(Square_Lattice_Bare.ecount()))

    return Square_Lattice, Square_Lattice_Bare

def InitializeSquareLattice(nodes, Square_Lattice_Bare, Square_Lattice, N_x, N_y):
    # Constructs initial square lattice that connects the vertices in 'nodes'

    # Re-orders nodes according to shortest length between the two vertices
    nodes = nodes + N_x * (N_y + 1)

    # Constructs initial lattice connecting nodes
    Lattice_Initial = Square_Lattice_Bare.copy()
    Lattice_temp = Square_Lattice.copy()
    for i in range(0, np.shape(nodes)[0]):
        paths = Lattice_temp.get_shortest_paths(nodes[i][0], to=nodes[i][1])
        path = paths[np.random.randint(np.array(paths).shape[0])]
        for j in range(0, len(path) - 1):
            Lattice_Initial.add_edge(path[j], path[j + 1],
                weight=Square_Lattice.es.select(Square_Lattice.get_eid(path[j], path[j + 1]))['weight'])

    return nodes, Lattice_Initial

def AddInitialStrings_SquareLattice(nodes, Flipped_Square_Lattice, Square_Lattice, N_x, N_y):
    # Adds to Flipped_Square_Lattice the initial strings that connects the vertices in 'nodes'

    nodes = nodes + N_x*(N_y+1)

    # Constructs initial lattice connecting nodes
    New_Lattice = Flipped_Square_Lattice.copy()
    Lattice_temp = Square_Lattice.copy()
    for i in range(0, np.shape(nodes)[0]):
        paths = Lattice_temp.get_shortest_paths(nodes[i][0], to=nodes[i][1])
        path = paths[np.random.randint(np.array(paths).shape[0])]
        for j in range(0, len(path) - 1):
            if New_Lattice.are_connected(path[j], path[j + 1]):
                New_Lattice.delete_edges((path[j], path[j + 1]))
            else:
                New_Lattice.add_edge(path[j], path[j + 1], weight=1)

    return New_Lattice

def ComputeRandomUniqueCombinations(N_faces, n, samples):
    # Determines random faces to flip

    np.random.seed()
    combs = np.zeros([samples, n])
    i = 0
    while i < samples:
        combs[i, :] = np.sort(np.random.permutation(range(1, N_faces + 1))[0:n])
        i = i + 1
        if i == samples:
            combs = np.unique(combs, axis=0)
            i = np.shape(combs)[0]
            combs = np.pad(combs, ((0, samples - i), (0, 0)), 'constant')
    return combs

def FlipSquareLatticeFaces(Lattice, coords, N_x):
    # Flips the faces in coords

    for x, y in coords:
        v = N_x * (y - 1) + x - 1   # Lower left vertex of face to be flipped

        # Edges to be flipped
        if np.mod(v+1, N_x) == 0:
            to_flip = np.array(
                [[v, v + N_x], [v, N_x * (y - 1)], [v + N_x, N_x * y], [N_x * (y - 1), N_x * y]]).astype(int)
        else:
            to_flip = np.array([[v, v + 1], [v, v + N_x], [v + N_x, v + 1 + N_x], [v + 1, v + 1 + N_x]]).astype(int)

        # Flips edges
        for v1, v2 in to_flip:
            if Lattice.are_connected(v1, v2):
                Lattice.delete_edges((v1, v2))
            else:
                Lattice.add_edge(v1, v2, weight=1)

    return Lattice

def AddNCLoop(Lattice, N_x, rung):
    # Adds to Lattice a noncontractible loop at rung

    for i in range(0, N_x-1):
        if Lattice.are_connected(int((rung - 1)*N_x + i), int((rung - 1)*N_x + i + 1)):
            Lattice.delete_edges( (int((rung - 1)*N_x + i), int((rung - 1)*N_x + i + 1)) )
        else:
            Lattice.add_edge(int((rung - 1) * N_x + i), int((rung - 1) * N_x + i + 1), weight=1)

    if Lattice.are_connected(int((rung - 1)*N_x), int(rung*N_x - 1)):
        Lattice.delete_edges( (int((rung - 1)*N_x), int(rung*N_x - 1)) )
    else:
        Lattice.add_edge(int((rung - 1)*N_x), int(rung*N_x - 1), weight=1)

    return Lattice

def ComputeAs_component_0_Square(N_x, Lattice_Initial, deg2_weight, loop_weight, rung):
    # Computes 0th component of A (configuration with no strings)

    As_component_0 = 0

    As_component_0 += 1  # Contractible configuration (no faces flipped)

    # Noncontractible configuration
    Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
    deg2 = Lattice_nc.degree().count(2)
    loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components())
    As_component_0 += (deg2_weight ** deg2) * (loop_weight ** loops)  # Sum of Boltzmann weights

    return As_component_0

def ComputeAs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma, loop_weight,
                                            combs, rung, sample):
    # Computes contribution to A (configurations with no strings) by flipping faces of combs

    As_component_contribution = 0

    # Finds coordinates of faces to be flipped in loop configuration
    n = np.shape(combs)[1]
    coords = np.zeros([n, 2])
    for j in range(0, n):
        coords[j, :] = [np.floor((combs[sample, j] - 1) / N_y) + 1, np.mod(combs[sample, j] - 1, N_y) + 1]

    # Flips faces, contractible configuration
    Lattice_c = FlipSquareLatticeFaces(Lattice_Initial.copy(), coords, N_x)
    # Flips faces, noncontractible configuration
    Lattice_nc = AddNCLoop(Lattice_c.copy(), N_x, rung)

    # Adds contribution from contractible lattice configuration
    deg2 = Lattice_c.degree().count(2)
    deg4 = Lattice_c.degree().count(4)
    loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components())
    As_component_contribution += (deg2_weight) ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

    # Adds contribution from noncontractible lattice configuration
    deg2 = Lattice_nc.degree().count(2)
    deg4 = Lattice_nc.degree().count(4)
    loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components())
    As_component_contribution += (deg2_weight) ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

    return As_component_contribution

def ComputeA_and_B_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma, loop_weight, strings, n_low,
                          n_high, samples, AllCombinations, iteration):
    # Computes components of A (configurations wih 0 strings) and B (configurations with 1 string)

    print('Iteration: ' + str(iteration + 1))
    rung = np.mod(iteration, N_y + 1) + 1
    As_component = np.zeros(n_high - n_low + 1)
    Bs_components_dict = {}
    for end_node in range(2, int(N_x/2)+1+1):
        Bs_components_dict[end_node] = np.zeros(n_high - n_low + 1)


    for n in range(n_low, n_high + 1):
        if np.mod(iteration+1, 16) == 0 or iteration == 0:
            print('Iteration: ' + str(iteration + 1) + ', n: ' + str(n))
        if n == 0:
            # Contractible and noncontractible configurations
            Lattice_c_NoStrings = Lattice_Initial.copy()
            Lattice_nc_NoStrings = AddNCLoop(Lattice_Initial.copy(), N_x, rung)

            # Contribution from contractible configuration
            As_component[0] += 1

            # Contribution form noncontractible configuration
            deg2 = Lattice_nc_NoStrings.degree().count(2)
            deg4 = Lattice_nc_NoStrings.degree().count(4)
            loops = np.array((Lattice_nc_NoStrings.subgraph(
                [nod for (nod, deg) in enumerate(Lattice_nc_NoStrings.degree()) if deg >= 1])).components()).shape[0]
            As_component[0] += deg2_weight ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

            # Finds B for different end_nodes
            for end_node in range(2, int(N_x / 2) + 1 + 1):
                # Computes B fo each different end_node
                nodes = np.array([[1, end_node]])
                parity = np.mod(end_node - 1, 2)

                # Contractible and noncontractible configurations
                Lattice_c = AddInitialStrings_SquareLattice(nodes, Lattice_c_NoStrings.copy(), Square_Lattice, N_x, N_y)
                Lattice_nc = AddInitialStrings_SquareLattice(nodes, Lattice_nc_NoStrings.copy(),
                                                             Square_Lattice, N_x, N_y)

                # Contribution from contractible configuration
                deg2 = Lattice_c.degree().count(2)
                deg4 = Lattice_c.degree().count(4)
                loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                      if deg >= 1])).components()) - strings
                Bs_components_dict[end_node][0] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

                # Contribution from noncontractible configuration
                deg2 = Lattice_nc.degree().count(2)
                deg4 = Lattice_nc.degree().count(4)
                loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                      if deg >= 1])).components()) - strings
                Bs_components_dict[end_node][0] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)
        else:
            # Constructs list of combinations (loop configurations) to analyze
            combs = AllCombinations[iteration, n]

            # Contributions to A and B
            As_component_contributions = np.zeros(np.shape(combs)[0])
            Bs_component_contributions_dict = {}
            for end_node in range(2, int(N_x / 2) + 1 + 1):
                Bs_component_contributions_dict[end_node] = np.zeros(np.shape(combs)[0])

            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(0, np.shape(combs)[0]):
                # Finds coordinates of faces to be flipped in loop configuration
                coords = np.zeros([n, 2])
                for j in range(0, n):
                    coords[j, :] = [np.floor((combs[i, j] - 1)/N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]

                # Flips faces, contractible config
                Lattice_c_NoStrings = FlipSquareLatticeFaces(Lattice_Initial.copy(), coords, N_x)
                # Flips faces, noncontractible config
                Lattice_nc_NoStrings = AddNCLoop(Lattice_c_NoStrings.copy(), N_x, rung)

                # Contribution from contractible configuration
                deg2 = Lattice_c_NoStrings.degree().count(2)
                deg4 = Lattice_c_NoStrings.degree().count(4)
                loops = len((Lattice_c_NoStrings.subgraph([nod for (nod, deg)
                    in enumerate(Lattice_c_NoStrings.degree()) if deg >= 1])).components())
                As_component_contributions[i] += deg2_weight ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

                # Contribution from noncontractible configuration
                deg2 = Lattice_nc_NoStrings.degree().count(2)
                deg4 = Lattice_nc_NoStrings.degree().count(4)
                loops = len((Lattice_nc_NoStrings.subgraph([nod for (nod, deg)
                    in enumerate(Lattice_nc_NoStrings.degree()) if deg >= 1])).components())
                As_component_contributions[i] += deg2_weight ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

                for end_node in range(2, int(N_x/2)+1+1):
                    # Computes B fo each different end_node
                    nodes = np.array([[1, end_node]])
                    parity = np.mod(end_node - 1, 2)

                    # Contractible and noncontractible configurations
                    Lattice_c = AddInitialStrings_SquareLattice(nodes, Lattice_c_NoStrings.copy(),
                                                                Square_Lattice, N_x, N_y)
                    Lattice_nc = AddInitialStrings_SquareLattice(nodes, Lattice_nc_NoStrings.copy(),
                                                                 Square_Lattice, N_x, N_y)

                    # Contribution from contractible configuration
                    deg2 = Lattice_c.degree().count(2)
                    deg4 = Lattice_c.degree().count(4)
                    loops = len((Lattice_c.subgraph([nod for (nod, deg)
                        in enumerate(Lattice_c.degree()) if deg >= 1])).components()) - strings
                    Bs_component_contributions_dict[end_node][i] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*\
                                                                    loop_weight**(loops)

                    # Contribution from noncontractible configuration
                    deg2 = Lattice_nc.degree().count(2)
                    deg4 = Lattice_nc.degree().count(4)
                    loops = len((Lattice_nc.subgraph([nod for (nod, deg)
                        in enumerate(Lattice_nc.degree()) if deg >= 1])).components()) - strings
                    Bs_component_contributions_dict[end_node][i] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*\
                                                                    loop_weight**(loops)


            # Averages terms if necessary
            if (gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1)) > np.log(samples):
                # Computes log averages
                As_component[n-n_low] = np.exp(np.mean(np.log(As_component_contributions)))*\
                        np.exp(gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1))
                for end_node in range(2, int(N_x/2)+1+1):
                    parity = np.mod(end_node - 1, 2)
                    Bs_components_dict[end_node][n-n_low] = (-1)**parity*\
                        np.exp(np.mean(np.log(np.abs(Bs_component_contributions_dict[end_node]))))*\
                        np.exp(gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1))
                    # equivalent to Bs(n,iteration)/samples*nchoosek(N_faces,n)
            else:
                As_component[n - n_low] = sum(As_component_contributions)
                for end_node in range(2, int(N_x/2)+1+1):
                    Bs_components_dict[end_node][n-n_low] = sum(Bs_component_contributions_dict[end_node])

    As_and_Bs_component = Bs_components_dict
    As_and_Bs_component[0] = As_component
    return As_and_Bs_component

def exp_correlation_function(x, A, xi):
    # Correlation function with exponential decay
    return A*np.exp(-x/xi)



if __name__ == '__main__':
    # Parameter specification
    t = time.time()
    N_x = 4                             # Number of squares in x direction; assumed to be even
    N_y = 4                             # Number of squares in y direction
    N_faces = N_x*N_y                   # Total number of squares being considered
    h = 1                               # Height of squares
    w = 1                               # Width of squares

    deg2_weights = np.array([1/3])      # Weights of degree 2 vertex
    CR_weight = 1/15                    # Weight of crossing
    CP_weight = 1/15                    # Weight of a corner pass
    loop_weight = 3                     # Weight of a closed loop
    gamma = (loop_weight+1)*CP_weight + CR_weight  # Total contribution fom a degree 4 vertex

    epsilon = 0.01                      # Maximum admissible error in coefficients
    samples = 100                       # Maximum number of samples (loop configurations) evaluated
    range_samples = 100                 # Number of samples used in determining n_range
    iterations = 32                     # Number of iterations over which coefficients are averaged

    # Initializes lattices and two point functions
    Square_Lattice, Square_Lattice_Bare = ConstructSquareLattices(N_x, N_y, w, h)
    TwoPointFunctions = np.zeros([int(N_x/2), len(deg2_weights)])
    correlation_lengths_log = np.zeros(len(deg2_weights))
    correlation_lengths_exp = np.zeros(len(deg2_weights))

    for deg2_index in range(len(deg2_weights)):
        deg2_weight = deg2_weights[deg2_index]  # Weight of a degree 2 vertex
        print('\n \n \n deg2_weight = ' + str(deg2_weight) + '\n')

        print('Determining Combinations: ')
        AllCombinations = {}
        for iteration in range(iterations):
            print('Iteration: ' + str(iteration+1))
            for n in range(1, N_faces+1):
                if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                        np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                    AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                else:
                    avg = 0
                    AllCombinations[iteration, n] = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))

        # Determines n_range - the lowest and highest number of 'on' faces to be considered
        print('Determining n_range')
        As_component = np.zeros(N_faces + 1)
        iteration = np.random.randint(0, iterations)
        rung = np.random.randint(1, N_y + 1 + 1)

        # 0th component
        Lattice_Initial = Square_Lattice_Bare.copy()
        As_component[0] = ComputeAs_component_0_Square(N_x, Lattice_Initial, deg2_weight, loop_weight, rung)

        max = As_component[0]
        max_n_index = 0
        # Highest and lowest number of 'on' faces to be considered
        high = N_faces
        low = 0
        for n in range(1, N_faces + 1):
            if np.mod(n, 1) == 0:
                print('n: ' + str(n))

            combs = AllCombinations[iteration, n]
            # Computes exp(-energy) for each loop config to be analyzed
            def ComputeAs_component_contribution_Square_parallel(sample):
                return ComputeAs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma,
                                                               loop_weight, combs, rung, sample)

            pool = mp.Pool()
            As_component_contributions = \
                np.transpose(pool.map(ComputeAs_component_contribution_Square_parallel, range(np.shape(combs)[0])))
            pool.close()
            pool.join()


            # Averages As terms if necessary
            if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > np.log(range_samples):
                As_component[n] = np.exp(np.mean(np.log(As_component_contributions))) * \
                    np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1))
                # equivalent to As(n,iter)/(range_samples)*nchoosek(N_faces,n)
            else:
                As_component[n] = np.sum(As_component_contributions[:])

            if np.abs(As_component[n]) > max:
                max = abs(As_component[n])
                max_n_index = n
            elif np.abs(As_component[n]) * (N_faces - n) / max < epsilon:
                high = n
                break
        for n in range(max_n_index, 0, -1):
            if np.abs(As_component[n]) * (n + 1) / max < epsilon:
                low = n
                break
        approx_error = (np.abs(As_component[high] * (N_faces - high)) + np.abs(As_component[low - 1]) * (low)) / \
                       np.abs(np.sum(As_component[low:high + 1]))
        As_determine_n = As_component[np.nonzero(As_component)]
        n_low = low
        n_high = high

        print('\n As for determining n_range: ')
        print(As_determine_n)
        print('\n n_high: ' + str(n_high))
        print('n_low: ' + str(n_low))
        print('error: <= ' + str(approx_error))


        # Computes A and B
        Lattice_Initial = Square_Lattice_Bare.copy()
        strings = 1

        def ComputeA_and_B_Square_parallel(iteration):
            return ComputeA_and_B_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma, loop_weight, strings,
                                         n_low, n_high, samples, AllCombinations, iteration)

        pool = mp.Pool()
        # A and B coefficients for each number of flips
        As_and_Bs = pool.map(ComputeA_and_B_Square_parallel, range(iterations))
        pool.close()
        pool.join()

        # Organizes A and B's
        As = np.zeros([n_high-n_low+1, iterations])
        for iteration in range(iterations):
            As[:, iteration] = As_and_Bs[iteration][0]

        Bs_dict = {}
        for end_node in range(2, int(N_x/2)+1+1):
            Bs_dict[end_node] = np.zeros([n_high-n_low+1, iterations])
            for iteration in range(iterations):
                Bs_dict[end_node][:, iteration] = As_and_Bs[iteration][end_node]

        # Averages As over iterations
        As_avg = np.zeros(n_high - n_low + 1)
        for n in range(0, n_high - n_low + 1):
            As_avg[n] = np.mean(As[n, :])

        # Averages Bs over iterations
        Bs_avg_dict = {}
        for end_node in range(2, int(N_x / 2) + 1 + 1):
            Bs_avg_dict[end_node] = np.zeros(n_high - n_low + 1)
            for n in range(0, n_high - n_low + 1):
                Bs_avg_dict[end_node][n] = np.mean(Bs_dict[end_node][n, :])

        # Computes A
        A = np.real(np.sum(As_avg))

        # Computes B's
        B = np.zeros(int(N_x/2))
        for end_node in range(2, int(N_x/2)+1+1):
            B[end_node-2] = np.real(np.sum(Bs_avg_dict[end_node]))

        # Prints results
        print('\n A_avg:')
        print(As_avg)
        print('\n A: ' + str(A) + '\n')

        print('\n Bs_avg:')
        for Bs in Bs_avg_dict:
            print(Bs_avg_dict[Bs])

        for i in range(int(N_x/2)):
            TwoPointFunctions[i, deg2_index] = 3*B[i]/A

        print('\n Two Point Functions:')
        for func in TwoPointFunctions[:, deg2_index]:
            print(func)

        # Fits two points functions to exponentially decaying correlation functions
        # Extracts correlation length from log fit and exponential fit
        LogTwoPointFunctions = [np.log(np.abs(TwoPointFunctions[i, deg2_index])) for i in range(int(N_x/2))]
        m, b, r_value, p_value, std_err = \
            stats.linregress(range(1, len(LogTwoPointFunctions) + 1), LogTwoPointFunctions)
        correlation_lengths_log[deg2_index] = -1/m
        popt, pcov = curve_fit(exp_correlation_function, list(range(1, int(N_x/2) + 1)),
                               np.abs(TwoPointFunctions[0:int(N_x/2), deg2_index]), p0=[np.exp(b), -1/m])
        correlation_lengths_exp[deg2_index] = popt[1]

    # Prints results and writes results to a txt file
    t2 = time.time()
    print('\n \n \n \n ALL DONE!!!')
    print('Runtime: ' + str(t2 - t))
    for deg2_index in range(len(deg2_weights)):
        print('\n \n deg2_weight: ' + str(deg2_weights[deg2_index]))

        print('\n Two Point Functions:')
        for func in TwoPointFunctions[:, deg2_index]:
            print(func)

        print('\n Correlation Length (log fit):')
        print(correlation_lengths_log[deg2_index])
        print('\n Correlation Length (exp fit):')
        print(correlation_lengths_exp[deg2_index])


    with open(os.path.basename(__file__) + ".txt", "w") as text_file:
        print('N_x = ' + str(N_x), file=text_file)
        print('N_y = ' + str(N_y), file=text_file)
        print('deg2_weights = ', file=text_file)
        print(deg2_weights, file=text_file)
        print('CR_weight = ' + str(CR_weight), file=text_file)
        print('CP_weight = ' + str(CP_weight), file=text_file)
        print('loop_weight = ' + str(loop_weight), file=text_file)

        print('epsilon = ' + str(epsilon), file=text_file)
        print('samples = ' + str(samples), file=text_file)
        print('range samples = ' + str(range_samples), file=text_file)
        print('iterations = ' + str(iterations), file=text_file)
        print('Runtime: ' + str(t2 - t), file=text_file)

        for deg2_index in range(len(deg2_weights)):
            print('\n \n \n deg2_weight: ' + str(deg2_weights[deg2_index]), file=text_file)

            print('\n Two Point Functions:', file=text_file)
            for func in TwoPointFunctions[:, deg2_index]:
                print(func, file=text_file)

            print('\n Correlation Length (log fit):', file=text_file)
            print(correlation_lengths_log[deg2_index], file=text_file)
            print('\n Correlation Length (exp fit):', file=text_file)
            print(correlation_lengths_exp[deg2_index], file=text_file)
