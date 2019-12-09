# Import necessary packages
import numpy as np
import igraph as ig
from scipy.special import gammaln
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

def ComputeAs_component_0_Square(N_x, Lattice_Initial, deg2_weight, loop_weight, rung):
    # Computes 0th component of A (configuration with no strings)

    As_component_0 = 0

    # Contractible configuration (no faces flipped)
    As_component_0 += 1

    # Noncontractible configuration
    Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
    deg2 = Lattice_nc.degree().count(2)
    loops = 1
    As_component_0 += (deg2_weight**deg2)*(loop_weight**loops)

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
    loops = len([x for x in Lattice_c.components() if len(x) > 1])
    As_component_contribution += (deg2_weight)**(deg2)*gamma**(deg4)*loop_weight**(loops)

    # Adds contribution from noncontractible lattice configuration
    deg2 = Lattice_nc.degree().count(2)
    deg4 = Lattice_nc.degree().count(4)
    loops = len([x for x in Lattice_nc.components() if len(x) > 1])
    As_component_contribution += (deg2_weight)**(deg2)*gamma**(deg4)*loop_weight**(loops)

    return As_component_contribution

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
        deg4_configs = np.random.randint(3, size=(deg4_samples, deg4)) + 1
    else:
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
    Z0_components = np.zeros(n_high - n_low + 1)    # Components of Z with no strings

    for n in range(n_low, n_high + 1):
        if np.mod(iteration + 1, 16) == 0:
            print('Iteration: ' + str(iteration + 1) + ' n: '+str(n))
        if n == 0:
            # Contribution from contractible configuration (no faces flipped)
            Z0_components[0] += 1

            # Contribution from noncontractible configuration
            Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
            deg2 = Lattice_nc.degree().count(2)
            loops = len([x for x in Lattice_nc.components() if len(x) > 1])
            edges = Lattice_nc.ecount()
            w = (deg2_weight**deg2)*(loop_weight**loops)

            Z0_components[0] += w
            loop_numbers[0] += loops*w
            loop_sizes[0] += (edges/loops)*w
        else:
            n_index = n-n_low

            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces+1) - gammaln(n+1) - gammaln(N_faces-n+1)) > \
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
                loops0 = len([x for x in Lattice_c.components() if len(x) > 1])
                edges = Lattice_c.ecount()
                if deg4 >= 1:
                    deg4_configs = Compute_deg4_configs(deg4, deg4_samples)
                    deg4_avg_factor = (3**deg4/np.shape(deg4_configs)[0])
                    for deg4_config in deg4_configs:
                        CP1 = list(deg4_config).count(1)    # Corner passes of type 1
                        CP2 = list(deg4_config).count(3)    # Corner passes of type 2
                        CP = CP1 + CP2
                        CR = list(deg4_config).count(2)     # Crossings

                        loops = loops0+CP1                  # One type of corner pass increases the number of loops
                        w = (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)*(loop_weight**loops)*(deg4_avg_factor)

                        Z0_components[n_index] += w
                        loop_numbers[n_index] += loops*w
                        loop_sizes[n_index] += (edges/loops)*w
                else:
                    loops = loops0
                    w = (deg2_weight**deg2)*(loop_weight**loops)

                    Z0_components[n_index] += w
                    loop_numbers[n_index] += loops*w
                    loop_sizes[n_index] += (edges/loops)*w


                # Contribution from noncontractible configuration
                deg2 = Lattice_nc.degree().count(2)
                deg4 = Lattice_nc.degree().count(4)

                loops0 = len([x for x in Lattice_nc.components() if len(x) > 1])
                edges = Lattice_nc.ecount()
                if deg4 >= 1:
                    deg4_configs = Compute_deg4_configs(deg4, deg4_samples)
                    deg4_avg_factor = (3**deg4/np.shape(deg4_configs)[0])
                    for deg4_config in deg4_configs:
                        CP1 = list(deg4_config).count(1)    # Corner passes of type 1
                        CP2 = list(deg4_config).count(3)    # Corner passes of type 2
                        CP = CP1 + CP2
                        CR = list(deg4_config).count(2)     # Crossings

                        loops = loops0+CP1                  # One type of corner pass increases the number of loops
                        w = (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)*(loop_weight**loops)*deg4_avg_factor

                        Z0_components[n_index] += w
                        loop_numbers[n_index] += loops*w
                        loop_sizes[n_index] += (edges/loops)*w
                else:
                    loops = loops0
                    w = (deg2_weight**deg2)*(loop_weight**loops)

                    Z0_components[n_index] += w
                    loop_numbers[n_index] += loops*w
                    loop_sizes[n_index] += (edges/loops)*w

    loop_number = sum(loop_numbers)/(sum(Z0_components))
    loop_size = sum(loop_sizes)/(sum(Z0_components))
    return loop_number, loop_size




if __name__ == '__main__':
    # Parameter specification
    t = time.time()
    N_x = 10                            # Number of squares in x direction; assumed to be even
    N_y = 10                            # Number of squares in y direction
    N_faces = N_x*N_y                   # Total number of squares being considered
    h = 1                               # Height of squares
    w = 1                               # Width of squares

    # Sweeps over c_2 and c_4; c_\ell remains fixed
    # Weights of degree 2 vertex
    deg2_weights = np.array([.1, .2, .3])
    # Weight of crossing
    CR_weights = np.array([.2])
    CP_weights = CR_weights             # Weight of a corner pass
    loop_weight = 3                     # Weight of a closed loop

    epsilon = 0.00                      # Maximum admissible error in coefficients
    samples = 40                        # Maximum number of samples (loop configurations) evaluated
    iterations = 32                     # Number of iterations over which coefficients are averaged
    deg4_samples = 20                   # Maximum number of deg4 configs sampled over
    range_samples = 50                  # Number of samples used to determine n_range


    # Initializes lattices; sets the low number of face flips to 0 and high number of face flips to N_faces
    Square_Lattice, Square_Lattice_Bare = ConstructSquareLattices(N_x, N_y, w, h)
    n_low = 0
    n_high = N_faces
    loop_numbers = np.zeros([len(deg2_weights), len(CR_weights)])
    loop_sizes = np.zeros([len(deg2_weights), len(CR_weights)])
    Lattice_Initial = Square_Lattice_Bare.copy()

    for deg4_index in range(len(CR_weights)):
        for deg2_index in range(len(deg2_weights)):
            # Sets weights
            CR_weight = CR_weights[deg4_index]
            CP_weight = CP_weights[deg4_index]
            gamma = (loop_weight+1)*CP_weight+CR_weight  # Contribution from deg4 vertex if all deg4 configs are valid
            deg2_weight = deg2_weights[deg2_index]
            print('\n \n \n deg2_weight = ' + str(deg2_weight) + '\n')
            print('\n \n \n deg4_weight = ' + str(CR_weight) + '\n')

            # Determines low and high number of faces to be flipped, such that Z is accurate to within epsilon percent
            print('Determining n_range')
            if epsilon == 0:
                n_low = 0
                n_high = N_faces
            else:
                As_component = np.zeros(N_faces + 1)
                rung = np.random.randint(1, N_y + 1 + 1)

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

                    # Constructs list of combinations (loop configurations) to analyze
                    if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                            np.log(range_samples):  # equivalent to nchoosek(N_faces, n) > samples
                        combs = ComputeRandomUniqueCombinations(N_faces, n, range_samples)
                        avg = 1
                    else:
                        combs = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))
                        avg = 0


                    # Computes exp(-energy) for each loop config to be analyzed
                    def ComputeAs_component_contribution_Square_parallel(sample):
                        return ComputeAs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma,
                                                                       loop_weight, combs, rung, sample)


                    pool = mp.Pool()
                    As_component_contributions = \
                        np.transpose(
                            pool.map(ComputeAs_component_contribution_Square_parallel, range(np.shape(combs)[0])))
                    pool.close()
                    pool.join()

                    if avg == 1:
                        As_component[n] = np.mean(As_component_contributions) * \
                                          np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1))
                    else:
                        As_component[n] = np.sum(As_component_contributions[:])

                    if np.abs(As_component[n]) > max:
                        max = abs(As_component[n])
                        max_n_index = n
                    # elif np.abs(As_component[n]) * (N_faces - n) / max < epsilon:
                    elif np.abs(As_component[n]) * (N_faces - n) / np.sum(As_component) < epsilon:
                        high = n
                        break
                for n in range(max_n_index, 0, -1):
                    # if np.abs(As_component[n]) * (n + 1) / max < epsilon:
                    if np.abs(As_component[n]) * (n + 1) / np.sum(As_component) < epsilon:
                        low = n
                        break
                approx_error = (np.abs(As_component[high] * (N_faces - high)) + np.abs(As_component[low - 1])*(low))/\
                               np.abs(np.sum(As_component[low:high + 1]))
                if high == N_faces:
                    approx_error = 0
                As_determine_n = As_component[np.nonzero(As_component)]
                n_low = low
                n_high = high

                print('\n As for determining n_range: ')
                print(As_determine_n)
                print('\n n_high: ' + str(n_high))
                print('n_low: ' + str(n_low))
                print('error: <= ' + str(approx_error))




            # Computes average loop perimeter and number
            def ComputeLoopProperties_Square_parallel(iteration):
                return ComputeLoopProperties_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, CP_weight,
                    CR_weight, loop_weight, n_low, n_high, samples, deg4_samples, iteration)
            pool = mp.Pool()
            loop_numbers_iters, loop_sizes_iters = np.transpose(
                pool.map(ComputeLoopProperties_Square_parallel, range(iterations)))
            pool.close()
            pool.join()


            # Averages loop properties and prints results
            loop_numbers[deg2_index, deg4_index] = np.mean(loop_numbers_iters)
            loop_sizes[deg2_index, deg4_index] = np.mean(loop_sizes_iters)
            print('Average loop number: ' + str(loop_numbers[deg2_index]))
            print('Average loop size (perimeter): ' + str(loop_sizes[deg2_index]) + '\n \n')

    # Prints results
    t2 = time.time()
    print('\n \n \n \n ALL DONE!!!')
    print('Runtime: ' + str(t2 - t))
    print('\n deg2_weights: \n')
    for w in deg2_weights:
        print(w)
    print('\n loop_weight: \n')
    print(loop_weight)
    print('\n deg4_weights: \n')
    for w in CP_weights:
        print(w)
    print('\n Average loop numbers: ')
    for i in range(len(deg2_weights)):
        for j in range(len(CP_weights)):
            print(loop_numbers[i, j])
    print('\n Average loop sizes (perimeters): ')
    for i in range(len(deg2_weights)):
        for j in range(len(CP_weights)):
            print(loop_sizes[i, j])

    # Outputs results to text file
    with open(os.path.basename(__file__) + ".txt", "w") as text_file:
        print('N_x = ' + str(N_x), file=text_file)
        print('N_y = ' + str(N_y), file=text_file)
        print('deg2_weights = ', file=text_file)
        print(deg2_weights, file=text_file)
        print('deg4_weights = ', file=text_file)
        print(CP_weights, file=text_file)
        print('loop_weight = ' + str(loop_weight), file=text_file)
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
