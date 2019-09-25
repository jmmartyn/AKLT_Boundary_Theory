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
    Square_Lattice = ig.Graph()     # Vertices are indexed from 0 in igraph
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
            to_flip = np.array([[v, v+N_x], [v, N_x*(y-1)], [v+N_x, N_x*y], [N_x*(y-1), N_x*y]]).astype(int)
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

def ComputeAllPairs(lst):
    # Computes all possible pairs of elements in lst

    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in ComputeAllPairs(lst[:i] + lst[i + 1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1, len(lst)):
            pair = [a, lst[i]]
            for rest in ComputeAllPairs(lst[1:i] + lst[i + 1:]):
                yield pair + rest

def Compute_deg4_configs(deg4, deg4_samples):
    # Computes possible configurations of deg4 vertices

    if deg4_samples < 3**deg4:
        np.random.seed()
        deg4_configs = np.zeros([deg4_samples, deg4])
        i = 0
        while i < deg4_samples:
            deg4_configs[i, :] = np.random.randint(3, size=deg4) + 1
            i = i + 1
            if i == deg4_samples:
                deg4_configs = np.unique(deg4_configs, axis=0)
                i = np.shape(deg4_configs)[0]
                deg4_configs = np.pad(deg4_configs, ((0, deg4_samples - i), (0, 0)), 'constant')

    else:
        deg4_avg_factor = 1
        deg4_configs = np.reshape(list([p for p in product([1, 2, 3], repeat=deg4)]),
                                  (-1, deg4))

    return deg4_configs

def ComputeLoopProperties_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, CP_weight, CR_weight, loop_weight,
                                 n_low, n_high, samples, deg4_samples, iteration):
    # Computes average loop number and loop size

    print('Iteration: ' + str(iteration + 1))
    rung = np.mod(iteration, N_y + 1) + 1
    loop_numbers = np.zeros(n_high - n_low + 1)
    loop_sizes = np.zeros(n_high - n_low + 1)
    As_component = np.zeros(n_high - n_low + 1)

    for n in range(n_low, n_high + 1):
        if np.mod(iteration+1, 16) == 0:
            print('Iteration: ' + str(iteration + 1) + ' n: '+str(n))
        if n == 0:
            # Contribution from contractible configuration (no faces flipped)
            As_component[0] += 1

            # Contribution from noncontractible configuration
            Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
            deg2 = Lattice_nc.degree().count(2)
            loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                    if deg >= 1])).components())
            edges = Lattice_nc.ecount()
            w = (deg2_weight ** deg2) * (loop_weight ** loops)

            As_component[0] += w
            loop_numbers[0] += loops * w
            loop_sizes[0] += (edges / loops) * w
        else:
            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                    np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                combs = ComputeRandomUniqueCombinations(N_faces, n, samples)
            else:
                combs = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))

            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(0, np.shape(combs)[0]):
                # Finds coordinates of faces to be flipped in loop configuration
                coords = np.zeros([n, 2])
                for j in range(0, n):
                    coords[j, :] = [np.floor((combs[i, j] - 1) / N_y) + 1,  np.mod(combs[i, j] - 1, N_y) + 1]

                # Flips faces, contractible config
                Lattice_c = FlipSquareLatticeFaces(Lattice_Initial.copy(), coords, N_x)
                # Flips faces, noncontractible config
                Lattice_nc = AddNCLoop(Lattice_c.copy(), N_x, rung)

                # Contribution from contractible configuration
                deg2 = Lattice_c.degree().count(2)
                deg4 = Lattice_c.degree().count(4)
                loops0 = len((Lattice_c.subgraph(
                    [nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components())
                edges = Lattice_c.ecount()
                if deg4 >= 1:
                    deg4_configs = Compute_deg4_configs(deg4, deg4_samples)
                    for deg4_config in deg4_configs:
                        CP1 = list(deg4_config).count(1)    # Corner passes of type 1
                        CP2 = list(deg4_config).count(3)    # Corner passes of type 2
                        CP = CP1 + CP2
                        CR = list(deg4_config).count(2)     # Crossings

                        loops = loops0+CP1                  # One type of corner pass increases the number of loops
                        w = (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)*(loop_weight**loops)*\
                            (3**deg4/np.shape(deg4_configs)[0])

                        As_component[n - n_low] += w
                        loop_numbers[n - n_low] += loops*w
                        loop_sizes[n - n_low] += (edges/loops)*w
                else:
                    loops = loops0
                    w = (deg2_weight**deg2)*(loop_weight**loops)

                    As_component[n - n_low] += w
                    loop_numbers[n - n_low] += loops * w
                    loop_sizes[n - n_low] += (edges / loops) * w

                # Contribution from noncontractible configuration
                deg2 = Lattice_nc.degree().count(2)
                deg4 = Lattice_nc.degree().count(4)
                loops0 = len((Lattice_nc.subgraph(
                    [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components())
                edges = Lattice_nc.ecount()
                if deg4 >= 1:
                    deg4_configs = Compute_deg4_configs(deg4, deg4_samples)
                    for deg4_config in deg4_configs:
                        CP1 = list(deg4_config).count(1)    # Corner passes of type 1
                        CP2 = list(deg4_config).count(3)    # Corner passes of type 2
                        CP = CP1 + CP2
                        CR = list(deg4_config).count(2)     # Crossings

                        loops = loops0+CP1                  # One type of corner pass increases the number of loops
                        w = (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)*(loop_weight**loops)*\
                            (3**deg4/np.shape(deg4_configs)[0])

                        As_component[n - n_low] += w
                        loop_numbers[n - n_low] += loops*w
                        loop_sizes[n - n_low] += (edges/loops)*w
                else:
                    loops = loops0
                    w = (deg2_weight**deg2)*(loop_weight**loops)

                    As_component[n - n_low] += w
                    loop_numbers[n - n_low] += loops * w
                    loop_sizes[n - n_low] += (edges / loops) * w

    loop_number = sum(loop_numbers) / (sum(As_component))
    loop_size = sum(loop_sizes) / (sum(As_component))
    return loop_number, loop_size



if __name__ == '__main__':
    # Parameter specification
    t = time.time()
    N_x = 10                            # Number of squares in x direction; assumed to be even
    N_y = 10                            # Number of squares in y direction
    N_faces = N_x*N_y                   # Total number of squares being considered
    h = 1                               # Height of squares
    w = 1                               # Width of squares

    deg2_weights = np.array([0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3])
    # Weights of degree 2 vertex
    CR_weight = 1 / 15                  # Weight of crossing
    CP_weight = 1 / 15                  # Weight of a corner pass
    loop_weights = np.array([.01, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2])
    # Weight of a closed loop

    samples = 80                        # Maximum number of samples (loop configurations) evaluated
    iterations = 32                     # Number of iterations over which coefficients are averaged
    range_samples = 50                  # Number of samples used to determine n_range
    deg4_samples = 20                   # Maximum number of deg4 configs sampled over

    # Initializes lattices; sets the low number of face flips to 0 and high number of face flips to N_Faces
    Square_Lattice, Square_Lattice_Bare = ConstructSquareLattices(N_x, N_y, w, h)
    n_low = 0
    n_high = N_faces
    loop_numbers = np.zeros([len(deg2_weights), len(loop_weights)])
    loop_sizes = np.zeros([len(deg2_weights), len(loop_weights)])
    Lattice_Initial = Square_Lattice_Bare.copy()

    for loop_index in range(len(loop_weights)):
        for deg2_index in range(len(deg2_weights)):
            loop_weight = loop_weights[loop_index]  # Weight of a loop
            deg2_weight = deg2_weights[deg2_index]  # Weight of a degree 2 vertex
            print('\n \n \n loop_weight = ' + str(loop_weight) + '\n')
            print('\n \n \n deg2_weight = ' + str(deg2_weight) + '\n')

            def ComputeLoopProperties_Square_parallel(iteration):
                return ComputeLoopProperties_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, CP_weight,
                    CR_weight, loop_weight, n_low, n_high, samples, deg4_samples, iteration)

            pool = mp.Pool()
            # B coefficients for each number of faces and iterations
            loop_numbers_iters, loop_sizes_iters = np.transpose(
                pool.map(ComputeLoopProperties_Square_parallel, range(iterations)))
            pool.close()
            pool.join()

            # Averages loop properties and prints results
            loop_numbers[deg2_index, loop_index] = np.mean(loop_numbers_iters)
            loop_sizes[deg2_index, loop_index] = np.mean(loop_sizes_iters)
            print('Average loop number: ' + str(loop_numbers[deg2_index]))
            print('Average loop size (perimeter): ' + str(loop_sizes[deg2_index]) + '\n \n')

    # Prints results
    t2 = time.time()
    print('\n \n \n \n ALL DONE!!!')
    print('Runtime: ' + str(t2 - t))
    print('\n deg2_weights: \n')
    for i in range(len(deg2_weights)):
        print(deg2_weights[i])
    print('\n loop_weights: \n')
    for i in range(len(loop_weights)):
        print(loop_weights[i])
    print('\n Average loop numbers: ')
    for i in range(len(deg2_weights)):
        for j in range(len(loop_weights)):
            print(loop_numbers[i, j])
    print('\n Average loop sizes (perimeters): ')
    for i in range(len(deg2_weights)):
        for j in range(len(loop_weights)):
            print(loop_sizes[i, j])

    # Outputs results to text file
    with open(os.path.basename(__file__) + ".txt", "w") as text_file:
        print('N_x = ' + str(N_x), file=text_file)
        print('N_y = ' + str(N_y), file=text_file)
        print('deg2_weights = ', file=text_file)
        print(deg2_weights, file=text_file)
        print('loop_weights = ', file=text_file)
        print(loop_weights, file=text_file)
        print('CR_weight = ' + str(CR_weight), file=text_file)
        print('CP_weight = ' + str(CP_weight), file=text_file)
        print('samples = ' + str(samples), file=text_file)
        print('iterations = ' + str(iterations), file=text_file)
        print('deg4_samples = ' + str(deg4_samples) + '\n', file=text_file)
        print('Runtime: ' + str(t2 - t), file=text_file)

        print('\n \n \n deg2_weights: ', file=text_file)
        for i in range(len(deg2_weights)):
            print(str(deg2_weights[i]), file=text_file)
        print('\n Average loop numbers: ', file=text_file)
        print(loop_numbers, file=text_file)
        print('\n Average loop sizes (perimeters): ', file=text_file)
        print(loop_sizes, file=text_file)

    np.savetxt(str(os.path.basename(__file__) + "_LoopNumbers.txt"), loop_numbers, delimiter='\t')
    np.savetxt(str(os.path.basename(__file__) + "_LoopSizes.txt"), loop_sizes, delimiter='\t')
