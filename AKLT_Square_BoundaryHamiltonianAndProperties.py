# Import necessary packages
import numpy as np
from numpy import linalg as la
import igraph as ig
from scipy import linalg as LA
from scipy import sparse
from scipy.special import gammaln
from itertools import combinations, product
import multiprocessing as mp
import os
import time
import copy


def PlotGraph(G):
    # Plots graph G with appropriate positions of vertices

    layout = zip(G.vs["x"], -1 * np.array(G.vs["y"]))
    pl = ig.plot(G, layout=layout)
    pl.show()


def ConstructSquareLattices(N_x, N_y, w, h):
    # Constructs cylindrical square lattices

    # Square lattice
    Square_Lattice = ig.Graph()  # Vertices are indexed from 0 in igraph
    x_squares = np.arange(0.0, w * N_x, w)
    y_squares = 0 * h * np.ones(N_x)
    y_squares[-1] = y_squares[-1] + 0.35 * h
    Square_Lattice.add_vertices(N_x)
    Square_Lattice.vs["x"] = x_squares[0:N_x]
    Square_Lattice.vs["y"] = y_squares[0:N_x]
    Square_Lattice.add_edges(zip(range(0, N_x - 1), range(1, N_x)))
    Square_Lattice.add_edge(0, N_x - 1)
    for rung in range(2, N_y + 1 + 1):
        x_squares = np.append(x_squares, np.arange(0.0, w * N_x, w))
        y_squares = np.append(y_squares, (rung - 1) * h * np.ones(N_x))
        y_squares[-1] = y_squares[-1] + 0.35 * h
        Square_Lattice.add_vertices(N_x)
        Square_Lattice.vs.select(range((rung - 1) * N_x, rung * N_x))["x"] = x_squares[(rung - 1) * N_x:rung * N_x]
        Square_Lattice.vs.select(range((rung - 1) * N_x, rung * N_x))["y"] = y_squares[(rung - 1) * N_x:rung * N_x]
        Square_Lattice.add_edges(zip(range((rung - 1) * N_x, rung * N_x - 1), range((rung - 1) * N_x + 1, rung * N_x)))
        Square_Lattice.add_edge((rung - 1) * N_x, rung * N_x - 1)
        for ladder in range(0, N_x):
            Square_Lattice.add_edge(((rung - 1) * N_x) + ladder, ((rung - 2) * N_x) + ladder)

    x_squares = np.append(x_squares, np.arange(0.0, w * N_x, w))
    y_squares = np.append(y_squares, -1 * h * np.ones(N_x))
    y_squares[-1] = y_squares[-1] + 0.35 * h
    Square_Lattice.add_vertices(N_x)
    Square_Lattice.vs.select(range((N_y + 1) * N_x, (N_y + 2) * N_x))["x"] = x_squares[(N_y + 1) * N_x: (N_y + 2) * N_x]
    Square_Lattice.vs.select(range((N_y + 1) * N_x, (N_y + 2) * N_x))["y"] = y_squares[(N_y + 1) * N_x: (N_y + 2) * N_x]
    Square_Lattice.add_edges(zip(range(0, N_x), range(N_x * (N_y + 1), N_x * (N_y + 2))))
    Square_Lattice.es["weight"] = np.ones(Square_Lattice.ecount()).astype(int)

    # Square lattice with no edges
    Square_Lattice_Bare = Square_Lattice.copy()
    Square_Lattice_Bare.delete_edges(range(Square_Lattice_Bare.ecount()))

    return Square_Lattice, Square_Lattice_Bare


def ConstructSquare_qb_Lattices(N_x, N_y, w, h):
    # Constructs cylindrical square lattices of qubits

    # Square lattice of qubits
    Square_Lattice_qb = ig.Graph()  # Vertices in igraph are indexed from 0
    x_squares_qb = np.zeros(4 * N_x)
    y_squares_qb = np.zeros(4 * N_x)
    for i in range(0, N_x):
        x_squares_qb[(4 * i): 4 * (i + 1)] = i * w
        y_squares_qb[(4 * i): 4 * (i + 1)] = 0 * h
    x_squares_qb[1::4] += 0.2 * w
    x_squares_qb[3::4] += - 0.2 * w
    y_squares_qb[0::4] += 0.2 * h
    y_squares_qb[2::4] += - 0.2 * h
    y_squares_qb[-4:] += 0.35 * h
    y_squares_qb[-1] += - 0.05 * h
    y_squares_qb[-3] += 0.03 * h
    Square_Lattice_qb.add_vertices(4 * N_x)
    Square_Lattice_qb.vs["x"] = x_squares_qb[0:4 * N_x]
    Square_Lattice_qb.vs["y"] = y_squares_qb[0:4 * N_x]
    Square_Lattice_qb.add_edges(zip(range(1, (4 * N_x - 6), 4), range(7, 4 * N_x, 4)))
    Square_Lattice_qb.add_edge(3, 4 * N_x - 3)
    for rung in range(2, N_y + 1 + 1):
        x_squares_qb_temp = np.zeros(4 * N_x)
        y_squares_qb_temp = np.zeros(4 * N_x)
        for i in range(0, N_x):
            x_squares_qb_temp[(4 * i): 4 * (i + 1)] = i * w
            y_squares_qb_temp[(4 * i): 4 * (i + 1)] = (rung - 1) * h
        x_squares_qb_temp[1::4] += 0.2 * w
        x_squares_qb_temp[3::4] += - 0.2 * w
        y_squares_qb_temp[0::4] += 0.2 * h
        y_squares_qb_temp[2::4] += - 0.2 * h
        y_squares_qb_temp[-4:] += 0.35 * h
        y_squares_qb_temp[-1] += - 0.05 * h
        y_squares_qb_temp[-3] += 0.03 * h
        x_squares_qb = np.append(x_squares_qb, x_squares_qb_temp)
        y_squares_qb = np.append(y_squares_qb, y_squares_qb_temp)
        Square_Lattice_qb.add_vertices(4 * N_x)
        Square_Lattice_qb.vs.select(range((rung - 1) * 4 * N_x, rung * 4 * N_x))["x"] = x_squares_qb[(
                                                                                                                 rung - 1) * 4 * N_x: rung * 4 * N_x]
        Square_Lattice_qb.vs.select(range((rung - 1) * 4 * N_x, rung * 4 * N_x))["y"] = y_squares_qb[(
                                                                                                                 rung - 1) * 4 * N_x: rung * 4 * N_x]
        Square_Lattice_qb.add_edges(zip(range((rung - 1) * 4 * N_x + 1, rung * 4 * N_x - 6, 4),
                                        range((rung - 1) * 4 * N_x + 7, rung * 4 * N_x, 4)))
        Square_Lattice_qb.add_edge((rung - 1) * 4 * N_x + 3, rung * 4 * N_x - 3)
        for ladder in range(0, N_x):
            Square_Lattice_qb.add_edge((rung - 1) * 4 * N_x + 2 + 4 * ladder, (rung - 2) * 4 * N_x + 4 * ladder)
    nodes_to_add = np.linspace(3, 4 * N_x - 1, N_x) - 1
    for i in range(1, N_x + 1):
        Square_Lattice_qb.add_vertex(x=(i - 1) * w, y=-1 * h + 0.35 * h * int(i == N_x))
    for i in range(1, N_x + 1):
        Square_Lattice_qb.add_edge(int(nodes_to_add[i - 1]), 4 * N_x * (N_y + 1) + i - 1)
    Square_Lattice_qb.es["weight"] = np.ones(Square_Lattice_qb.ecount())

    # Square lattice of qubits with all neighboring qubits connected
    Square_Lattice_qb_all = Square_Lattice_qb.copy()
    for i in range(1, N_x * (N_y + 1) + 1):
        index = 4 * (i - 1)
        Square_Lattice_qb_all.add_edge(index, index + 1, weight=0)
        Square_Lattice_qb_all.add_edge(index, index + 2, weight=0)
        Square_Lattice_qb_all.add_edge(index, index + 3, weight=0)
        Square_Lattice_qb_all.add_edge(index + 1, index + 2, weight=0)
        Square_Lattice_qb_all.add_edge(index + 1, index + 3, weight=0)
        Square_Lattice_qb_all.add_edge(index + 2, index + 3, weight=0)

    # Square lattice of qubits with all neighboring qubits connected via kinked lines
    Square_Lattice_qb_all_kinked = Square_Lattice_qb.copy()
    for i in range(1, N_x * (N_y + 1) + 1):
        index = 4 * (i - 1)
        Square_Lattice_qb_all_kinked.add_edge(index, index + 1, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index, index + 3, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index + 1, index + 2, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index + 2, index + 3, weight=0)

    # Square lattice of qubits with no edges
    Square_Lattice_qb_Bare = Square_Lattice_qb.copy()
    Square_Lattice_qb_Bare.delete_edges(range(Square_Lattice_qb_Bare.ecount()))

    return Square_Lattice_qb, Square_Lattice_qb_all_kinked, Square_Lattice_qb_all, Square_Lattice_qb_Bare


def InitializeSquareLattice(nodes, Square_Lattice_Bare, Square_Lattice, N_x, N_y):
    # Constructs initial square lattice that connects the vertices in 'nodes'

    # Re-orders nodes according to shortest length between the two vertices
    nodes = nodes + N_x * (N_y + 1)
    path_length = np.zeros(np.shape(nodes)[0])
    for i in range(0, np.shape(nodes)[0]):
        path_length[i] = Square_Lattice.shortest_paths(nodes[i][0], nodes[i][1])[0][0]
    I = path_length.argsort()
    nodes = nodes[I][:]

    # Constructs initial lattice connecting nodes
    Lattice_Initial = Square_Lattice_Bare.copy()
    Lattice_temp = Square_Lattice.copy()
    for i in range(0, np.shape(nodes)[0]):
        paths = Lattice_temp.get_shortest_paths(nodes[i][0], to=nodes[i][1])
        path = paths[np.random.randint(np.array(paths).shape[0])]
        for j in range(0, len(path) - 1):
            Lattice_Initial.add_edge(path[j], path[j + 1],
                                     weight=Square_Lattice.es.select(Square_Lattice.get_eid(path[j], path[j + 1]))[
                                         'weight'])

    return nodes, Lattice_Initial


def InitializeSquareLattice_qb(nodes, Square_Lattice_qb_Bare, Square_Lattice_qb_all, N_x, N_y):
    # Constructs initial square lattice that connects the vertices in 'nodes'

    # Re-orders nodes according to shortest length between the two vertices
    nodes = nodes + 4 * N_x * (N_y + 1)
    path_length = np.zeros(np.shape(nodes)[0])
    for i in range(0, np.shape(nodes)[0]):
        path_length[i] = Square_Lattice_qb_all.shortest_paths(nodes[i][0], nodes[i][1])[0][0]
    I = path_length.argsort()
    nodes = nodes[I][:]

    # Constructs initial lattice connecting nodes
    Lattice_Initial = Square_Lattice_qb_Bare.copy()
    Lattice_temp = Square_Lattice_qb_all.copy()
    for i in range(0, np.shape(nodes)[0]):
        paths = Lattice_temp.get_shortest_paths(nodes[i][0], to=nodes[i][1])
        path = paths[np.random.randint(np.array(paths).shape[0])]
        for j in range(0, len(path) - 1):
            Lattice_Initial.add_edge(path[j], path[j + 1],
                                     weight=Square_Lattice_qb_all.es.select(Square_Lattice_qb_all.get_eid(
                                         path[j], path[j + 1]))['weight'][0])
        Lattice_temp.delete_edges(Lattice_temp.get_eids(zip(path[0:-1], path[1:])))
        for v in path:
            Lattice_temp.delete_edges(Lattice_temp.incident(v))

    # Determines the configuration of the deg4 vertices in Lattice_Initial
    Initial_deg4_config = {}
    Initial_deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_Initial, N_x, N_y)
    for vertex in Initial_deg4_clusters:
        if Lattice_Initial.are_connected(vertex, vertex + 1):
            Initial_deg4_config[vertex] = 1
        elif Lattice_Initial.are_connected(vertex, vertex + 3):
            Initial_deg4_config[vertex] = 3
        elif Lattice_Initial.are_connected(vertex, vertex + 2):
            Initial_deg4_config[vertex] = 2

    # Makes initial lattice kinked (needed for face flips)
    for i in range(N_x * (N_y + 1)):
        if len(Lattice_Initial.subgraph([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]).components()) < 4:
            S = Lattice_Initial.subgraph([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
            if len(S.components()) == 3 and Lattice_Initial.are_connected(4 * i, 4 * i + 2):
                Lattice_Initial.delete_edges((4 * i, 4 * i + 2))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i, 4 * i + 1 + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 1 + 2 * x, 4 * i + 2, weight=0)
            elif len(S.components()) == 3 and Lattice_Initial.are_connected(4 * i + 1, 4 * i + 3):
                Lattice_Initial.delete_edges((4 * i + 1, 4 * i + 3))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i + 1, 4 * i + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 2 * x, 4 * i + 3, weight=0)
            elif len(S.components()) == 2 and Lattice_Initial.are_connected(4 * i, 4 * i + 2) \
                    and Lattice_Initial.are_connected(4 * i + 1, 4 * i + 3):
                Lattice_Initial.delete_edges(((4 * i, 4 * i + 2), (4 * i + 1, 4 * i + 3)))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i, 4 * i + 1 + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 2, 4 * i + 3 - 2 * x, weight=0)

    return nodes, Lattice_Initial, Initial_deg4_config


def ComputeRandomUniqueCombinations(N_faces, n, samples):
    # Randomly determines a unique combination of faces to flip

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


def ComputeStringConfigs(N_x, strings):
    # Computes the start/end nodes of all possible unique configurations with 'strings' strings

    lst = list(range(N_x))
    lsts = list(combinations(lst, 2 * strings))
    for lst2 in lsts:
        lst2 = list(lst2)
        a = lst2[0]
        for i in range(1, len(lst2)):
            pair = [a, lst2[i]]
            for rest in ComputeAllPairs(lst2[1:i] + lst2[i + 1:]):
                yield pair + rest


def ComputeAs_component_0_Square(N_x, Lattice_Initial, deg2_weight, loop_weight, rung):
    # Computes 0th component of A (configuration with no strings)

    As_component_0 = 0

    # Contractible configuration (no faces flipped)
    As_component_0 += 1

    # Noncontractible configuration
    Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
    deg2 = Lattice_nc.degree().count(2)
    loops = 1
    As_component_0 += (deg2_weight ** deg2) * (loop_weight ** loops)

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
    As_component_contribution += (deg2_weight) ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

    # Adds contribution from noncontractible lattice configuration
    deg2 = Lattice_nc.degree().count(2)
    deg4 = Lattice_nc.degree().count(4)
    loops = len([x for x in Lattice_nc.components() if len(x) > 1])
    As_component_contribution += (deg2_weight) ** (deg2) * gamma ** (deg4) * loop_weight ** (loops)

    return As_component_contribution


def FlipSquareLatticeFaces(Lattice, coords, N_x):
    # Flips the faces in coords (acts on the square lattice, not the lattice of qubits)

    for x, y in coords:
        v = N_x * (y - 1) + x - 1  # Lower left vertex of face to be flipped

        # Edges to be flipped
        if np.mod(v + 1, N_x) == 0:
            to_flip = np.array([[v, v + N_x], [v, N_x * (y - 1)], [v + N_x, N_x * y], [N_x * (y - 1), N_x * y]]).astype(
                int)
        else:
            to_flip = np.array([[v, v + 1], [v, v + N_x], [v + N_x, v + 1 + N_x], [v + 1, v + 1 + N_x]]).astype(int)

        # Flips edges
        for v1, v2 in to_flip:
            if Lattice.are_connected(v1, v2):
                Lattice.delete_edges((v1, v2))
            else:
                Lattice.add_edge(v1, v2, weight=1)

    return Lattice


def FlipSquareLatticeFaces_qb(Lattice_qb, coords, N_x):
    # Flips the faces in coords (acts on the lattice of qubits, not the square lattice)

    for x, y in coords:
        v = 4 * N_x * (y - 1) + 4 * (x - 1)  # Lower left vertex of face to be flipped

        # Edges with weight 0 and weight 1 to be flipped
        if x == N_x:
            to_flip0 = np.array([[v, v + 1], [4 * N_x * (y - 1), 4 * N_x * (y - 1) + 3],
                                 [4 * N_x * y + 2, 4 * N_x * y + 3], [v + 4 * N_x + 1, v + 4 * N_x + 2]]).astype(int)
            to_flip1 = np.array([[v + 1, 4 * N_x * (y - 1) + 3], [4 * N_x * (y - 1), 4 * N_x * y + 2],
                                 [v + 1 + 4 * N_x, 4 * N_x * y + 3], [v, v + 4 * N_x + 2]]).astype(int)
        else:
            to_flip0 = np.array([[v, v + 1], [v + 7, v + 4], [v + 4 + 4 * N_x + 2, v + 4 + 4 * N_x + 3],
                                 [v + 4 * N_x + 1, v + 4 * N_x + 2]]).astype(int)
            to_flip1 = np.array([[v + 1, v + 7], [v + 4, v + 4 + 4 * N_x + 2], [v + 4 * N_x + 1, v + 4 + 4 * N_x + 3],
                                 [v, v + 4 * N_x + 2]]).astype(int)

        # Flips edges
        for vs_0, vs_1 in zip(to_flip0, to_flip1):
            if Lattice_qb.are_connected(vs_0[0], vs_0[1]):
                Lattice_qb.delete_edges((vs_0[0], vs_0[1]))
            else:
                Lattice_qb.add_edge(vs_0[0], vs_0[1], weight=0)

            if Lattice_qb.are_connected(vs_1[0], vs_1[1]):
                Lattice_qb.delete_edges((vs_1[0], vs_1[1]))
            else:
                Lattice_qb.add_edge(vs_1[0], vs_1[1], weight=1)

    return Lattice_qb


def DetermineSquareLatticeValid(Lattice, nodes):
    # Determines if the configuration of Lattice (a square lattice) is valid

    valid = not (Lattice.degree().count(3) >= 1)
    for v1, v2 in nodes:
        if Lattice.shortest_paths(v1, v2)[0][0] == float('inf'):
            valid = False
    return valid


def DetermineSquareLatticeValid_qb(Lattice_qb, nodes):
    # Determines if the configuration of Lattice (a lattice of qubits) is valid

    for v1, v2 in nodes:
        if Lattice_qb.shortest_paths(v1, v2)[0][0] == float('inf'):
            return False
    return True


def DetermineSquareLatticeValid_NoStrings(Lattice):
    # Determines if the configuration of Lattice (square lattice with no strings, just loops) is valid

    valid = not (Lattice.degree().count(3) >= 3)
    return valid


def DetermineSquareLatticeValid_NoStrings_qb(Lattice_qb):
    # Determines if the configuration of Lattice (lattice of qubits with no strings, just loops) is valid

    valid = not (len([deg for deg in Lattice_qb.degree() if deg >= 3]) >= 1)
    return valid


def MakeCP1(Lattice_qb, vertex):
    # Makes a corner pass at vertex; Lattice_qb is a lattice of qubits

    if Lattice_qb.are_connected(vertex, vertex + 3):
        Lattice_qb.delete_edges(((vertex, vertex + 3), (vertex + 1, vertex + 2)))
        Lattice_qb.add_edge(vertex, vertex + 1, weight=0)
        Lattice_qb.add_edge(vertex + 2, vertex + 3, weight=0)
    return Lattice_qb


def MakeCP2(Lattice_qb, vertex):
    # Makes alternate corner pass at vertex; Lattice_qb is a lattice of qubits

    if Lattice_qb.are_connected(vertex, vertex + 1):
        Lattice_qb.delete_edges(((vertex, vertex + 1), (vertex + 2, vertex + 3)))
        Lattice_qb.add_edge(vertex, vertex + 3, weight=0)
        Lattice_qb.add_edge(vertex + 1, vertex + 2, weight=0)
    return Lattice_qb


def MakeCR(Lattice_qb, vertex):
    # Makes crossing at vertex; Lattice_qb is a lattice of qubits

    if Lattice_qb.are_connected(vertex, vertex + 1):
        Lattice_qb.delete_edges(((vertex, vertex + 1), (vertex + 2, vertex + 3)))
        Lattice_qb.add_edge(vertex, vertex + 2, weight=0)
        Lattice_qb.add_edge(vertex + 1, vertex + 3, weight=0)
    else:
        Lattice_qb.delete_edges(((vertex, vertex + 3), (vertex + 1, vertex + 2)))
        Lattice_qb.add_edge(vertex, vertex + 2, weight=0)
        Lattice_qb.add_edge(vertex + 1, vertex + 3, weight=0)
    return Lattice_qb


def AddNCLoop(Lattice, N_x, rung):
    # Adds to Lattice a noncontractible loop at rung

    for i in range(0, N_x - 1):
        if Lattice.are_connected(int((rung - 1) * N_x + i), int((rung - 1) * N_x + i + 1)):
            Lattice.delete_edges((int((rung - 1) * N_x + i), int((rung - 1) * N_x + i + 1)))
        else:
            Lattice.add_edge(int((rung - 1) * N_x + i), int((rung - 1) * N_x + i + 1), weight=1)

    if Lattice.are_connected(int((rung - 1) * N_x), int(rung * N_x - 1)):
        Lattice.delete_edges((int((rung - 1) * N_x), int(rung * N_x - 1)))
    else:
        Lattice.add_edge(int((rung - 1) * N_x), int(rung * N_x - 1), weight=1)

    return Lattice


def AddNCLoop_qb(Lattice_qb, N_x, rung):
    # Adds to Lattice_qb a noncontractible loop at rung

    vertex = (rung - 1) * 4 * N_x

    if Lattice_qb.are_connected(vertex + 1, vertex + 2):
        Lattice_qb.delete_edges((vertex + 1, vertex + 2))
    else:
        Lattice_qb.add_edge(vertex + 1, vertex + 2, weight=0)
    if Lattice_qb.are_connected(vertex + 2, vertex + 3):
        Lattice_qb.delete_edges((vertex + 2, vertex + 3))
    else:
        Lattice_qb.add_edge(vertex + 2, vertex + 3, weight=0)
    if Lattice_qb.are_connected(vertex + 3, rung * 4 * N_x - 3):
        Lattice_qb.delete_edges((vertex + 3, rung * 4 * N_x - 3))
    else:
        Lattice_qb.add_edge(vertex + 3, rung * 4 * N_x - 3, weight=1)

    for i in range(1, N_x):
        if Lattice_qb.are_connected(vertex + 2 + 4 * (i - 1) - 1, vertex + 8 + 4 * (i - 1) - 1):
            Lattice_qb.delete_edges((vertex + 2 + 4 * (i - 1) - 1, vertex + 8 + 4 * (i - 1) - 1))
        else:
            Lattice_qb.add_edge(vertex + 2 + 4 * (i - 1) - 1, vertex + 8 + 4 * (i - 1) - 1, weight=1)
        if Lattice_qb.are_connected(vertex + 4 * i + 1, vertex + 4 * i + 2):
            Lattice_qb.delete_edges((vertex + 4 * i + 1, vertex + 4 * i + 2))
        else:
            Lattice_qb.add_edge(vertex + 4 * i + 1, vertex + 4 * i + 2, weight=0)
        if Lattice_qb.are_connected(vertex + 4 * i + 2, vertex + 4 * i + 3):
            Lattice_qb.delete_edges((vertex + 4 * i + 2, vertex + 4 * i + 3))
        else:
            Lattice_qb.add_edge(vertex + 4 * i + 2, vertex + 4 * i + 3, weight=0)

    return Lattice_qb


def AddInitialStrings_SquareLattice(nodes, Flipped_Square_Lattice, Square_Lattice, N_x, N_y):
    # Adds to Flipped_Square_Lattice the initial strings that connects the vertices in 'nodes'

    nodes = nodes + N_x * (N_y + 1)

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


def Find_Initial_deg4_config_and_InitialConfigEdges_qb(nodes, Square_Lattice_qb_Bare,
                                                       Square_Lattice_qb_all, N_x, N_y):
    # Finds paths of initial config

    # Re-orders nodes according to shortest length between the two vertices
    nodes = nodes + 4 * N_x * (N_y + 1)
    path_length = np.zeros(np.shape(nodes)[0])
    for i in range(0, np.shape(nodes)[0]):
        path_length[i] = Square_Lattice_qb_all.shortest_paths(nodes[i][0], nodes[i][1])[0][0]
    I = path_length.argsort()
    nodes = nodes[I][:]

    # Constructs initial lattice connecting nodes
    Lattice_Initial = Square_Lattice_qb_Bare.copy()
    Lattice_temp = Square_Lattice_qb_all.copy()
    for i in range(0, np.shape(nodes)[0]):
        paths = Lattice_temp.get_shortest_paths(nodes[i][0], to=nodes[i][1])
        path = paths[np.random.randint(np.array(paths).shape[0])]
        for j in range(0, len(path) - 1):
            Lattice_Initial.add_edge(path[j], path[j + 1],
                                     weight=Square_Lattice_qb_all.es.select(
                                         Square_Lattice_qb_all.get_eid(path[j], path[j + 1]))['weight'][0])
        Lattice_temp.delete_edges(Lattice_temp.get_eids(zip(path[0:-1], path[1:])))
        for v in path:
            Lattice_temp.delete_edges(Lattice_temp.incident(v))

    # Determines the configuration of deg4 vertices in Lattice_Initial
    Initial_deg4_config = {}
    Initial_deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_Initial, N_x, N_y)
    for vertex in Initial_deg4_clusters:
        if Lattice_Initial.are_connected(vertex, vertex + 1):
            Initial_deg4_config[vertex] = 1
        elif Lattice_Initial.are_connected(vertex, vertex + 3):
            Initial_deg4_config[vertex] = 3
        elif Lattice_Initial.are_connected(vertex, vertex + 2):
            Initial_deg4_config[vertex] = 2

    # Makes initial lattice kinked (needed for face flips)
    for i in range(N_x * (N_y + 1)):
        if len(Lattice_Initial.subgraph([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]).components()) < 4:
            S = Lattice_Initial.subgraph([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
            if len(S.components()) == 3 and Lattice_Initial.are_connected(4 * i, 4 * i + 2):
                Lattice_Initial.delete_edges((4 * i, 4 * i + 2))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i, 4 * i + 1 + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 1 + 2 * x, 4 * i + 2, weight=0)
            elif len(S.components()) == 3 and Lattice_Initial.are_connected(4 * i + 1, 4 * i + 3):
                Lattice_Initial.delete_edges((4 * i + 1, 4 * i + 3))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i + 1, 4 * i + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 2 * x, 4 * i + 3, weight=0)
            elif len(S.components()) == 2 and Lattice_Initial.are_connected(4 * i, 4 * i + 2) \
                    and Lattice_Initial.are_connected(4 * i + 1, 4 * i + 3):
                Lattice_Initial.delete_edges(((4 * i, 4 * i + 2), (4 * i + 1, 4 * i + 3)))
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4 * i, 4 * i + 1 + 2 * x, weight=0)
                Lattice_Initial.add_edge(4 * i + 2, 4 * i + 3 - 2 * x, weight=0)

    InitialConfigEdges = [e.tuple for e in Lattice_Initial.es]

    return Initial_deg4_config, InitialConfigEdges


def AddInitialStrings_SquareLattice_qb(InitialConfigEdges, Square_Lattice_qb_all, Flipped_Square_Lattice_qb):
    # Adds initial strings to Flipped_Square_Lattice_qb

    Lattice_qb = Flipped_Square_Lattice_qb.copy()

    lst = [e.tuple for e in Lattice_qb.es]
    for edge in InitialConfigEdges:
        if edge in lst:
            Lattice_qb.delete_edges((edge[0], edge[1]))
        else:
            Lattice_qb.add_edge(edge[0], edge[1], weight=
            Square_Lattice_qb_all.es.select(Square_Lattice_qb_all.get_eid(edge[0], edge[1]))['weight'][0])

    return Lattice_qb


def Compute_rho(N_x, N_y, N_faces, deg2_weight, gamma, loop_weight, n_low, n_high, samples, AllCoords, iteration,
                Square_Lattice, UniqueCoefficientDictionary_zeros, ConjugacyClasses, UniqueStringConfigs,
                UniqueInitial_deg4_configs, UniqueInitialConfigEdges, sX, sY, sZ, max_strings):
    # Computes the density matrix of the boundary (rho)

    # Determines the rung of the noncontractible loop to analyze in the current iteration
    print('Iteration: ' + str(iteration + 1))
    rung = np.mod(iteration, N_y + 1) + 1
    UniqueCoefficientDictionary = copy.deepcopy(UniqueCoefficientDictionary_zeros)

    # Estimates coefficients by looking at configurations with n faces flipped
    for n in range(n_low, n_high + 1):
        if np.mod(iteration + 1, 16) == 0 or iteration == 0:
            # if 0 == 0:
            print('Iteration: ' + str(iteration + 1) + ', n: ' + str(n))
        if n == 0:
            # Computes Z for configurations with no strings

            # Contractible and noncontractible configurations
            Lattice_c_NoStrings = Square_Lattice_Bare.copy()
            Lattice_nc_NoStrings = AddNCLoop(Square_Lattice_Bare.copy(), N_x, rung)

            # Contribution from contractible configuration
            UniqueCoefficientDictionary[0][0][0] += 1

            # Contribution from noncontractible configuration
            deg2 = Lattice_nc_NoStrings.degree().count(2)
            deg4 = Lattice_nc_NoStrings.degree().count(4)
            loops = 1
            UniqueCoefficientDictionary[0][0][0] += (deg2_weight ** deg2) * (gamma ** deg4) * (loop_weight ** loops)

            # Computes Z for configurations with 1 string
            strings = 1
            for string_config_index in (UniqueStringConfigs[strings]).keys():
                nodes = np.array(UniqueStringConfigs[strings][string_config_index]).reshape(strings, 2)

                # Contractible and noncontractible configurations
                Lattice_c = AddInitialStrings_SquareLattice(nodes, Lattice_c_NoStrings.copy(),
                                                            Square_Lattice, N_x, N_y)
                Lattice_nc = AddInitialStrings_SquareLattice(nodes, Lattice_nc_NoStrings.copy(),
                                                             Square_Lattice, N_x, N_y)
                parity = np.mod(np.sum(Lattice_c.es['weight']), 2)

                # Contribution from contractible configuration
                deg2 = Lattice_c.degree().count(2)
                deg4 = Lattice_c.degree().count(4)
                loops = len([x for x in Lattice_c.components() if len(x) > 1]) - strings
                UniqueCoefficientDictionary[strings][string_config_index][0] += \
                    (-1) ** (parity) * (deg2_weight ** deg2) * (gamma ** deg4) * (loop_weight ** loops)

                # Contribution from noncontractible configuration
                deg2 = Lattice_nc.degree().count(2)
                deg4 = Lattice_nc.degree().count(4)
                loops = len([x for x in Lattice_nc.components() if len(x) > 1]) - strings
                UniqueCoefficientDictionary[strings][string_config_index][0] += \
                    (-1) ** (parity) * (deg2_weight ** deg2) * (gamma ** deg4) * (loop_weight ** loops)

            # Computes Z for configurations with strings \in [2, N_x/2]
            for strings in range(2, max_strings + 1):
                for string_config_index in (UniqueStringConfigs[strings]).keys():
                    # Finds nodes and initial string configurations
                    nodes = np.array(UniqueStringConfigs[strings][string_config_index]).reshape(strings, 2)
                    nodes1 = nodes + 4 * N_x * (N_y + 1)
                    InitialConfigEdges = UniqueInitialConfigEdges[strings][string_config_index]
                    Initial_deg4_config = UniqueInitial_deg4_configs[strings][string_config_index]

                    # Constructs contractible and noncontractible lattice
                    Lattice_qb_c = AddInitialStrings_SquareLattice_qb(InitialConfigEdges, Square_Lattice_qb_all,
                                                                      Square_Lattice_qb_Bare)
                    Lattice_qb_nc = AddNCLoop_qb(Lattice_qb_c.copy(), N_x, rung)
                    parity = np.mod(np.sum(Lattice_qb_c.es['weight']), 2)

                    # Contribution from contractible configuration
                    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_qb_c, N_x, N_y)
                    deg4 = len(deg4_clusters)
                    Lattice_qb_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                        Process_deg4(Lattice_qb_c.copy(), deg4_clusters, Initial_deg4_config)
                    weight1_edges = np.sum(Lattice_qb_c_Before_deg4_Flips.es['weight'])
                    deg2 = weight1_edges - 2 * deg4 - strings
                    if len(deg4_clusters) >= 1:
                        # Analyzes different degree 4 configurations
                        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                        for j in range(np.shape(deg4_configs)[0]):
                            Lattice_qb_c, CP, CR = Flip_deg4_clusters(Lattice_qb_c_Before_deg4_Flips.copy(),
                                                                      deg4_clusters, deg4_configs[j], CP0, CR0)
                            if DetermineSquareLatticeValid_qb(Lattice_qb_c, nodes1):
                                # Computes energy
                                loops = len([x for x in Lattice_qb_c.components() if len(x) > 1]) \
                                        - interior_loops - strings
                                UniqueCoefficientDictionary[strings][string_config_index][n - n_low] += \
                                    ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) * \
                                    (loop_weight ** loops) * (deg4_avg_factor)
                    else:
                        Lattice_qb_c = Lattice_qb_c_Before_deg4_Flips.copy()
                        if DetermineSquareLatticeValid_qb(Lattice_qb_c, nodes1):
                            # Computes energy
                            CP = CP0
                            CR = CR0
                            loops = len([x for x in Lattice_qb_c.components() if len(x) > 1]) \
                                    - interior_loops - strings
                            UniqueCoefficientDictionary[strings][string_config_index][0] += \
                                ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) * (
                                            loop_weight ** loops)

                    # Contribution from noncontractible configuration
                    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_qb_nc, N_x, N_y)
                    deg4 = len(deg4_clusters)
                    Lattice_qb_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                        Process_deg4(Lattice_qb_nc.copy(), deg4_clusters, Initial_deg4_config)
                    weight1_edges = np.sum(Lattice_qb_nc_Before_deg4_Flips.es['weight'])
                    deg2 = weight1_edges - 2 * deg4 - strings
                    if len(deg4_clusters) >= 1:
                        # Analyzes different degree 4 configurations
                        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                        for j in range(np.shape(deg4_configs)[0]):
                            Lattice_qb_nc, CP, CR = Flip_deg4_clusters(Lattice_qb_nc_Before_deg4_Flips.copy(),
                                                                       deg4_clusters, deg4_configs[j], CP0, CR0)
                            if DetermineSquareLatticeValid_qb(Lattice_qb_nc, nodes1):
                                # Computes energy
                                loops = len([x for x in Lattice_qb_nc.components() if len(x) > 1]) \
                                        - interior_loops - strings
                                UniqueCoefficientDictionary[strings][string_config_index][n - n_low] += \
                                    ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) * \
                                    (loop_weight ** loops) * (deg4_avg_factor)
                    else:
                        Lattice_qb_nc = Lattice_qb_nc_Before_deg4_Flips.copy()
                        if DetermineSquareLatticeValid_qb(Lattice_qb_nc, nodes1):
                            # Computes energy
                            CP = CP0
                            CR = CR0
                            loops = len([x for x in Lattice_qb_nc.components() if len(x) > 1]) \
                                    - interior_loops - strings
                            UniqueCoefficientDictionary[strings][string_config_index][0] += ((-1) ** parity) * \
                                                                                            (deg2_weight ** deg2) * (
                                                                                                        CP_weight ** CP) * (
                                                                                                        CR_weight ** CR) * (
                                                                                                        loop_weight ** loops)
        else:
            n_index = n - n_low
            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(len(AllCoords[iteration, n])):
                # print('Iteration: ' + str(iteration + 1) + ', n: ' + str(n) + ', i: ' + str(i))
                # Coordinates of faces to be flipped in loop configuration
                coords = AllCoords[iteration, n][i]

                # Computes Z for configurations with no strings
                # Flips faces, contractible config
                Lattice_c_NoStrings = FlipSquareLatticeFaces(Square_Lattice_Bare.copy(), coords, N_x)
                Lattice_qb_c_NoStrings = FlipSquareLatticeFaces_qb(Square_Lattice_qb_Bare.copy(), coords, N_x)

                # Flips faces, noncontractible config
                Lattice_nc_NoStrings = AddNCLoop(Lattice_c_NoStrings.copy(), N_x, rung)

                # Contribution from contractible configuration
                deg2 = Lattice_c_NoStrings.degree().count(2)
                deg4 = Lattice_c_NoStrings.degree().count(4)
                loops = len([x for x in Lattice_c_NoStrings.components() if len(x) > 1])
                UniqueCoefficientDictionary[0][0][n_index] += deg2_weight ** (deg2) * gamma ** (deg4) * loop_weight ** (
                    loops)

                # Contribution from noncontractible configuration
                deg2 = Lattice_nc_NoStrings.degree().count(2)
                deg4 = Lattice_nc_NoStrings.degree().count(4)
                loops = len([x for x in Lattice_nc_NoStrings.components() if len(x) > 1])
                UniqueCoefficientDictionary[0][0][n_index] += deg2_weight ** (deg2) * gamma ** (deg4) * loop_weight ** (
                    loops)

                # Computes Z for configurations with 1 string
                strings = 1
                for string_config_index in (UniqueStringConfigs[strings]).keys():
                    nodes = np.array(UniqueStringConfigs[strings][string_config_index]).reshape(strings, 2)

                    # Contractible and noncontractible configurations
                    Lattice_c = AddInitialStrings_SquareLattice(nodes, Lattice_c_NoStrings.copy(),
                                                                Square_Lattice, N_x, N_y)
                    Lattice_nc = AddInitialStrings_SquareLattice(nodes, Lattice_nc_NoStrings.copy(),
                                                                 Square_Lattice, N_x, N_y)
                    parity = np.mod(np.sum(Lattice_c.es['weight']), 2)

                    # Contribution from contractible configuration
                    deg2 = Lattice_c.degree().count(2)
                    deg4 = Lattice_c.degree().count(4)
                    loops = len([x for x in Lattice_c.components() if len(x) > 1]) - strings
                    UniqueCoefficientDictionary[strings][string_config_index][n_index] += (-1) ** (parity) * \
                                                                                          deg2_weight ** (
                                                                                              deg2) * gamma ** (
                                                                                              deg4) * loop_weight ** (
                                                                                              loops)

                    # Contribution from noncontractible configuration
                    deg2 = Lattice_nc.degree().count(2)
                    deg4 = Lattice_nc.degree().count(4)
                    loops = len([x for x in Lattice_nc.components() if len(x) > 1]) - strings
                    UniqueCoefficientDictionary[strings][string_config_index][n_index] += (-1) ** (parity) * \
                                                                                          deg2_weight ** (
                                                                                              deg2) * gamma ** (
                                                                                              deg4) * loop_weight ** (
                                                                                              loops)

                # Computes Z for configurations with strings \in [2, N_x/2]
                for strings in range(2, max_strings + 1):
                    # print('Iteration: ' + str(iteration + 1) + ', n: ' + str(n) + ', strings: ' + str(strings))
                    for string_config_index in (UniqueStringConfigs[strings]).keys():
                        # Finds nodes and initial string configurations
                        nodes = np.array(UniqueStringConfigs[strings][string_config_index]).reshape(strings, 2)
                        nodes1 = nodes + 4 * N_x * (N_y + 1)
                        InitialConfigEdges = UniqueInitialConfigEdges[strings][string_config_index]
                        Initial_deg4_config = UniqueInitial_deg4_configs[strings][string_config_index]

                        # Contractible and noncontractible configurations
                        Lattice_qb_c = AddInitialStrings_SquareLattice_qb(InitialConfigEdges, Square_Lattice_qb_all,
                                                                          Lattice_qb_c_NoStrings)
                        Lattice_qb_nc = AddNCLoop_qb(Lattice_qb_c.copy(), N_x, rung)
                        parity = np.mod(np.sum(Lattice_qb_c.es['weight']), 2)

                        # Contribution from contractible configuration
                        deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_qb_c, N_x, N_y)
                        deg4 = len(deg4_clusters)
                        Lattice_qb_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                            Process_deg4(Lattice_qb_c.copy(), deg4_clusters, Initial_deg4_config)
                        weight1_edges = np.sum(Lattice_qb_c_Before_deg4_Flips.es['weight'])
                        deg2 = weight1_edges - 2 * deg4 - strings
                        if len(deg4_clusters) >= 1:
                            # Analyzes different degree 4 configurations
                            deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                            for j in range(np.shape(deg4_configs)[0]):
                                Lattice_qb_c, CP, CR = Flip_deg4_clusters(Lattice_qb_c_Before_deg4_Flips.copy(),
                                                                          deg4_clusters, deg4_configs[j], CP0, CR0)
                                if DetermineSquareLatticeValid_qb(Lattice_qb_c, nodes1):
                                    # Computes energy
                                    loops = len([x for x in Lattice_qb_c.components() if len(x) > 1]) \
                                            - interior_loops - strings
                                    UniqueCoefficientDictionary[strings][string_config_index][n_index] += \
                                        ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (
                                                    CR_weight ** CR) * \
                                        (loop_weight ** loops) * (deg4_avg_factor)
                        else:
                            Lattice_qb_c = Lattice_qb_c_Before_deg4_Flips.copy()
                            if DetermineSquareLatticeValid_qb(Lattice_qb_c, nodes1):
                                # Computes energy
                                CP = CP0
                                CR = CR0
                                loops = len([x for x in Lattice_qb_c.components() if len(x) > 1]) \
                                        - interior_loops - strings
                                UniqueCoefficientDictionary[strings][string_config_index][n_index] += ((-1) ** parity) * \
                                                                                                      (
                                                                                                                  deg2_weight ** deg2) * (
                                                                                                                  CP_weight ** CP) * (
                                                                                                                  CR_weight ** CR) * (
                                                                                                                  loop_weight ** loops)

                        # Contribution from noncontractible configuration
                        deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_qb_nc, N_x, N_y)
                        deg4 = len(deg4_clusters)
                        Lattice_qb_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                            Process_deg4(Lattice_qb_nc.copy(), deg4_clusters, Initial_deg4_config)
                        weight1_edges = np.sum(Lattice_qb_nc_Before_deg4_Flips.es['weight'])
                        deg2 = weight1_edges - 2 * deg4 - strings
                        if len(deg4_clusters) >= 1:
                            # Analyzes different degree 4 configurations
                            deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                            for j in range(np.shape(deg4_configs)[0]):
                                Lattice_qb_nc, CP, CR = Flip_deg4_clusters(Lattice_qb_nc_Before_deg4_Flips.copy(),
                                                                           deg4_clusters, deg4_configs[j], CP0, CR0)
                                if DetermineSquareLatticeValid_qb(Lattice_qb_nc, nodes1):
                                    # Computes energy
                                    loops = len([x for x in Lattice_qb_nc.components() if len(x) > 1]) \
                                            - interior_loops - strings
                                    UniqueCoefficientDictionary[strings][string_config_index][n - n_low] += \
                                        ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (
                                                    CR_weight ** CR) * \
                                        (loop_weight ** loops) * (deg4_avg_factor)
                        else:
                            Lattice_qb_nc = Lattice_qb_nc_Before_deg4_Flips.copy()
                            if DetermineSquareLatticeValid_qb(Lattice_qb_nc, nodes1):
                                # Computes energy
                                CP = CP0
                                CR = CR0
                                loops = len([x for x in Lattice_qb_nc.components() if len(x) > 1]) \
                                        - interior_loops - strings
                                UniqueCoefficientDictionary[strings][string_config_index][n_index] += ((-1) ** parity) * \
                                                                                                      (
                                                                                                                  deg2_weight ** deg2) * (
                                                                                                                  CP_weight ** CP) * (
                                                                                                                  CR_weight ** CR) * (
                                                                                                                  loop_weight ** loops)

            # Averages terms if not all combinations are sampled
            if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > np.log(samples):
                UniqueCoefficientDictionary[0][0][n_index] = UniqueCoefficientDictionary[0][0][n_index] / samples * \
                                                             np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(
                                                                 N_faces - n + 1))
                for strings in range(1, max_strings + 1):
                    for string_config_index in (UniqueStringConfigs[strings]).keys():
                        UniqueCoefficientDictionary[strings][string_config_index][n_index] = \
                            UniqueCoefficientDictionary[strings][string_config_index][n_index] / samples * \
                            np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1))

    # After coefficients are estimated, rho is constructed
    if np.mod(iteration + 1, 16) == 0 or iteration == 0:
        print('Iteration: ' + str(iteration + 1) + ', Estimating rho')
    rho_unnormalized = np.sum(UniqueCoefficientDictionary[0][0]) * sparse.identity(2 ** N_x) / (2 ** N_x)
    for strings in range(1, max_strings + 1):
        if np.mod(iteration + 1, 16) == 0 or iteration == 0:
            print('Iteration: ' + str(iteration + 1) + ', Estimating rho; strings: ' + str(strings))
        for all_string_config_index, StringConfig in enumerate(AllStringConfigs[strings]):
            nodes = np.array(StringConfig).reshape(strings, 2)
            string_config_index = ConjugacyClasses[strings][all_string_config_index]
            rho_unnormalized = rho_unnormalized + np.sum(UniqueCoefficientDictionary[strings][string_config_index]) * \
                               1 / (2 ** N_x) * ComputeInteraction(nodes, N_x, sX, sY, sZ)

    # Normalizes rho
    rho_normalized = rho_unnormalized / (rho_unnormalized.diagonal().sum())
    if np.mod(iteration + 1, 16) == 0 or iteration == 0:
        print('Iteration: ' + str(iteration + 1) + ', Estimated rho')

    # Returns unnormalized and normalized rho
    return {0: rho_unnormalized, 1: rho_normalized}


def ComputeSquareLattice_qb_deg4_clusters(Lattice, N_x, N_y):
    # Determines the location of deg4 clusters in Lattice

    # Finds potential deg4 vertices
    deg4_clusters = np.array([4 * cl for cl in range(N_x * (N_y + 1))
                              if Lattice.degree()[4 * cl: 4 * cl + 4] == [2, 2, 2, 2]]).astype(int)
    # Counts interior loops
    interior_loops = len([cluster for cluster in deg4_clusters if
                          min(Lattice.subgraph([cluster, cluster + 1, cluster + 2, cluster + 3]).degree()) == 2])
    # Removes interior loops and false deg4 vertices
    deg4_clusters = [cluster for cluster in deg4_clusters
                     if len(Lattice.subgraph([cluster, cluster + 1, cluster + 2, cluster + 3]).components()) == 2]

    return deg4_clusters, interior_loops


def Compute_deg4_configs(deg4_clusters, deg4_samples):
    # Computes random configurations of deg4 vertices, and a factor used to average over deg4 contributions

    if deg4_samples < 3 ** len(deg4_clusters):
        np.random.seed()
        deg4_configs = np.random.randint(3, size=(deg4_samples, len(deg4_clusters))) + 1
        deg4_avg_factor = 3 ** len(deg4_clusters) / deg4_samples
    else:
        deg4_avg_factor = 1
        deg4_configs = np.reshape(list([p for p in product([1, 2, 3], repeat=len(deg4_clusters))]),
                                  (-1, len(deg4_clusters)))

    return deg4_configs, deg4_avg_factor


def Process_deg4(Lattice, deg4_clusters, Initial_deg4_config):
    # Makes initial deg4 vertices have their initial orientation
    # Calculates the number of initial corner passes (CP) and crossings (CR)

    CP0 = 0
    CR0 = 0
    Lattice_Before_deg4_Flips = Lattice

    # Enforces deg4 configurations of initial strings
    for vertex in list(Initial_deg4_config.keys()):
        if vertex in deg4_clusters:
            if Initial_deg4_config[vertex] == 1:
                Lattice_Before_deg4_Flips = MakeCP1(Lattice_Before_deg4_Flips, vertex)
                CP0 += 1
            elif Initial_deg4_config[vertex] == 3:
                Lattice_Before_deg4_Flips = MakeCP2(Lattice_Before_deg4_Flips, vertex)
                CP0 += 1
            elif Initial_deg4_config[vertex] == 2:
                Lattice_Before_deg4_Flips = MakeCR(Lattice_Before_deg4_Flips, vertex)
                CR0 += 1
            deg4_clusters = np.delete(deg4_clusters, np.argwhere(deg4_clusters == vertex))

    # Removes edges at remaining deg4 vertices
    for vertex in deg4_clusters:
        if Lattice_Before_deg4_Flips.are_connected(vertex, vertex + 1):
            Lattice_Before_deg4_Flips.delete_edges(((vertex, vertex + 1), (vertex + 2, vertex + 3)))
        else:
            Lattice_Before_deg4_Flips.delete_edges(((vertex, vertex + 3), (vertex + 1, vertex + 2)))

    return Lattice_Before_deg4_Flips, deg4_clusters, CP0, CR0


def Flip_deg4_clusters(Lattice, deg4_clusters, deg4_config, CP0, CR0):
    # Configures deg4 vertices according to deg4_configs[j, :]
    # Computes number of corner passes and crossings

    CP = CP0 + list(deg4_config).count(1) + list(deg4_config).count(3)
    CR = CR0 + list(deg4_config).count(2)
    others_list = np.array([[2, 3], [1, 3], [1, 2]])

    for k in range(len(deg4_config)):
        vertex_config = int(deg4_config[k])
        vertex = int(deg4_clusters[k])
        Lattice.add_edges([(vertex, vertex + vertex_config),
                           (vertex + others_list[vertex_config - 1, 0], vertex + others_list[vertex_config - 1, 1])])

    return Lattice, CP, CR


def ComputeInteraction(nodes0, N_x, sX, sY, sZ):
    # Computes the sigma dot sigma interaction of nodes in 'nodes0'

    nodes0 += 1
    Interaction = sparse.identity(2 ** N_x)
    for [i, j] in nodes0:
        Interaction = Interaction * ComputeSigmaDotSigma(i, j, N_x, sX, sY, sZ)
    return Interaction


def ComputeSigmaDotSigma(i, j, N_x, sX, sY, sZ):
    # Computes Sigma_i dot Sigma_j

    sX_i = sparse.kron(sparse.kron(sparse.identity(2 ** (i - 1)), sX, 'csr'), sparse.identity(2 ** (N_x - i)), 'csr')
    sY_i = sparse.kron(sparse.kron(sparse.identity(2 ** (i - 1)), sY, 'csr'), sparse.identity(2 ** (N_x - i)), 'csr')
    sZ_i = sparse.kron(sparse.kron(sparse.identity(2 ** (i - 1)), sZ, 'csr'), sparse.identity(2 ** (N_x - i)), 'csr')
    sX_j = sparse.kron(sparse.kron(sparse.identity(2 ** (j - 1)), sX, 'csr'), sparse.identity(2 ** (N_x - j)), 'csr')
    sY_j = sparse.kron(sparse.kron(sparse.identity(2 ** (j - 1)), sY, 'csr'), sparse.identity(2 ** (N_x - j)), 'csr')
    sZ_j = sparse.kron(sparse.kron(sparse.identity(2 ** (j - 1)), sZ, 'csr'), sparse.identity(2 ** (N_x - j)), 'csr')

    SigmaDotSigma = sX_i * sX_j + sY_i * sY_j + sZ_i * sZ_j

    return SigmaDotSigma


def Compute_A_r(H, N_x):
    # Computes amplitudes A_r of Hamiltonian H

    Z = np.array([[1, 0], [0, -1]])
    A_r = np.zeros((N_x, 1), dtype=np.complex)
    # A_0 = np.trace(H) / (N_x * 2 ** N_x)
    # X_squared = np.trace(H.dot(H))/(N_x*2**N_x)-A_0**2*10-3/16*(A_rs[1]**2+A_rs[2]**2+A_rs[3]**2+A_rs[4]**2+A_rs[5]**2)
    for r in range(0, N_x):
        for k in range(1, N_x + 1):
            Z_k = np.kron(np.kron(np.identity(2 ** (k - 1)), Z), np.identity(2 ** (N_x - k)))
            Z_kPlusr = np.kron(np.kron(np.identity(2 ** (np.mod(k + r - 1, N_x) + 1 - 1)), Z),
                               np.identity(2 ** (N_x - np.mod(k + r - 1, N_x) - 1)))
            A_r[r, 0] += 1 / (N_x * 3 * 2 ** N_x) * np.trace(np.dot(np.dot(H, Z_k), Z_kPlusr))

    return A_r[:, 0]


def Compute_d_n(H, N_x, sX, sY, sZ):
    # Computes n-qubit interaction strengths d_n of Hamiltonian H (assumed to be sparse)

    d_n = np.zeros((N_x + 1, 1), dtype=np.complex)
    h = (H.diagonal()).sum() / 2 ** N_x * sparse.identity(2 ** N_x)
    d_n[0, 0] = ((h * h).diagonal()).sum()

    for n in range(1, N_x+1):
        print('n: ' + str(n))
        h = sparse.csr_matrix((2 ** N_x, 2 ** N_x))
        for strings in range(1, int(n/2)+1):
            for nodes in list(ComputeStringConfigs(n, strings)):
                if (0 in nodes) and (n-1 in nodes):
                    # Uses orthogonality of Pauli operators to determine contribution to h
                    op = ComputeInteraction(np.array(nodes).reshape(strings, 2), N_x, sX, sY, sZ)
                    h = h + op * ((H * op).diagonal()).sum() / (3 ** strings * 2 ** N_x)
        h = h * N_x       # Interaction can begin at any of the N_x qubits
        d_n[n, 0] = ((h * h).diagonal()).sum()


    return d_n[:, 0]


def Compute_TwoPointFunctions(rho, N_x):
    # Computes two point functions

    Z = np.array([[1, 0], [0, -1]])
    TwoPointFunctions = np.zeros((N_x - 1, 1), dtype=np.complex)
    Z_1 = np.kron(Z, np.identity(2 ** (N_x - 1)))

    for r in range(1, N_x):
        Z_1Plusr = np.kron(np.kron(np.identity(2 ** (np.mod(r, N_x) + 1 - 1)), Z),
                           np.identity(2 ** (N_x - np.mod(r, N_x) - 1)))
        TwoPointFunctions[r - 1, 0] = 3 * np.trace(np.dot(np.dot(rho, Z_1), Z_1Plusr))

    return TwoPointFunctions[:, 0]


def nearestPD(A):
    B = (A + A.conj().T) / 2
    U, s, V = la.svd(B)

    H = np.dot(V.conj().T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.conj().T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False






if __name__ == '__main__':
    # Parameter specification
    t = time.time()
    N_x = 8                     # Number of squares in periodic direction, assumed to be even
    N_y = 8                     # Number of squares in y direction
    N_faces = N_x*N_y           # Total number of squares being considered
    h = 1                       # Height of squares
    w = 1                       # Width of squares

    deg2_weight = 1/3
    CR_weight = 1/15
    CP_weight = CR_weight                                                           # Weight of a corner pass
    loop_weight = 3                                                                 # Weight of a closed loop
    gamma = (loop_weight+1)*CP_weight+CR_weight  # Contribution from deg4 vertex if all deg4 configurations are valid

    epsilon = 0.001                 # Maximum admissible error in coefficients
    samples = 30                    # Maximum number of samples (loop configurations) evaluated
    epochs = 2                      # Number of epochs (iteration over noncontractible loop rungs)
    iterations = epochs*(N_y+1)     # Total number of iterations over which coefficients are averaged
    comb_method = 2                 # Method used to construct combinations of face flips that are analyzed
    range_samples = 50              # Number of samples used to determine n_range
    deg4_samples = 15               # Maximum number of deg4 configs sampled over
    max_strings = int(N_x/2)        # Max number of strings analyzed; an integer >=1 and <= int(N_x/2)

    # Creates square lattices used in later calculations
    Square_Lattice, Square_Lattice_Bare = ConstructSquareLattices(N_x, N_y, w, h)
    Square_Lattice_qb, Square_Lattice_qb_all_kinked, \
    Square_Lattice_qb_all, Square_Lattice_qb_Bare = ConstructSquare_qb_Lattices(N_x, N_y, w, h)

    # Pauli matrices, full and sparse
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    sX = sparse.csr_matrix(X)
    sY = sparse.csr_matrix(Y)
    sZ = sparse.csr_matrix(Z)


    # Prints weights
    print('\n \n \n deg2_weight = ' + str(deg2_weight))
    print('deg4_weight = ' + str(CR_weight) + '\n')

    # Constructs combinations of faces to be flipped
    if comb_method == 0:
        # Single method, same combinations for all iterations
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(1):
            print('Iteration: ' + str(iteration + 1))
            if iteration == 0:
                for n in range(1, N_faces + 1):
                    if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                            np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                        AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                    else:
                        avg = 0
                        AllCombinations[iteration, n] = np.reshape(list(combinations(range(1, N_faces + 1), n)),
                                                                   (-1, n))
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
            else:
                for n in range(1, N_faces + 1):
                    AllCombinations[iteration, n] = AllCombinations[0, n]
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
    elif comb_method == 1:
        # Epoch method, same combinations for each epoch
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(1):
            print('Iteration: ' + str(iteration + 1))
            if np.mod(iteration, epochs) == 0:
                for n in range(1, N_faces + 1):
                    if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                            np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                        AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                    else:
                        avg = 0
                        AllCombinations[iteration, n] = np.reshape(
                            list(combinations(range(1, N_faces + 1), n)), (-1, n))
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
            else:
                for n in range(1, N_faces + 1):
                    AllCombinations[iteration, n] = AllCombinations[int(np.floor(iteration / epochs)), n]
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
    elif comb_method == 2:
        # Full Method, different combinations for each iteration
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(1):
            print('Iteration: ' + str(iteration + 1))
            for n in range(1, N_faces + 1):
                if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                        np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                    AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                else:
                    avg = 0
                    AllCombinations[iteration, n] = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))
                combs = AllCombinations[iteration, n]
                AllCoords[iteration, n] = {}
                for i in range(np.shape(combs)[0]):
                    # Finds coordinates of faces to be flipped in loop configuration
                    AllCoords[iteration, n][i] = np.zeros([n, 2])
                    for j in range(0, n):
                        AllCoords[iteration, n][i][j, :] = \
                            [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]

    # Determines the low and high number of faces to be flipped, such that Z is accurate to within epsilon percent
    print('Determining n_range')
    iteration = 0
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
            combs = AllCombinations[iteration, n]
            if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                    np.log(range_samples):  # equivalent to nchoosek(N_faces, n) > range_samples
                avg = 1
            else:
                avg = 0


            # Computes exp(-energy) for each loop config to be analyzed
            def ComputeAs_component_contribution_Square_parallel(sample):
                return ComputeAs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma,
                                                               loop_weight, combs, rung, sample)


            pool = mp.Pool()
            As_component_contributions = \
                np.transpose(pool.map(ComputeAs_component_contribution_Square_parallel, range(np.shape(combs)[0])))
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
            elif np.abs(As_component[n]) * (N_faces - n) / max < epsilon:
                high = n
                break
        for n in range(max_n_index, 0, -1):
            if np.abs(As_component[n]) * (n + 1) / max < epsilon:
                low = n
                break
        low = 0
        approx_error = (np.abs(As_component[high] * (N_faces - high)) + np.abs(As_component[low - 1]) * (low)) / \
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

    # Constructs combinations of faces to be flipped
    if comb_method == 0:
        # Single method, same combinations for all iterations
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(iterations):
            print('Iteration: ' + str(iteration + 1))
            if iteration == 0:
                for n in range(n_low + int(n_low == 0), n_high + 1):
                    if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                            np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                        AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                    else:
                        avg = 0
                        AllCombinations[iteration, n] = np.reshape(list(combinations(range(1, N_faces + 1), n)),
                                                                   (-1, n))
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
            else:
                for n in range(1, N_faces + 1):
                    AllCombinations[iteration, n] = AllCombinations[0, n]
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
    elif comb_method == 1:
        # Epoch method, same combinations for each epoch
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(iterations):
            print('Iteration: ' + str(iteration + 1))
            if np.mod(iteration, epochs) == 0:
                for n in range(n_low + int(n_low == 0), n_high + 1):
                    if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                            np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                        AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                    else:
                        avg = 0
                        AllCombinations[iteration, n] = np.reshape(
                            list(combinations(range(1, N_faces + 1), n)), (-1, n))
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
            else:
                for n in range(1, N_faces + 1):
                    AllCombinations[iteration, n] = AllCombinations[int(np.floor(iteration / epochs)), n]
                    combs = AllCombinations[iteration, n]
                    AllCoords[iteration, n] = {}
                    for i in range(np.shape(combs)[0]):
                        # Finds coordinates of faces to be flipped in loop configuration
                        AllCoords[iteration, n][i] = np.zeros([n, 2])
                        for j in range(0, n):
                            AllCoords[iteration, n][i][j, :] = \
                                [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]
    elif comb_method == 2:
        # Full Method, different combinations for each iteration
        print('Determining Combinations: ')
        AllCombinations = {}
        AllCoords = {}
        for iteration in range(iterations):
            print('Iteration: ' + str(iteration + 1))
            for n in range(n_low + int(n_low == 0), n_high + 1):
                if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                        np.log(samples):  # equivalent to nchoosek(N_faces, n) > samples
                    AllCombinations[iteration, n] = ComputeRandomUniqueCombinations(N_faces, n, samples)
                else:
                    avg = 0
                    AllCombinations[iteration, n] = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))
                combs = AllCombinations[iteration, n]
                AllCoords[iteration, n] = {}
                for i in range(np.shape(combs)[0]):
                    # Finds coordinates of faces to be flipped in loop configuration
                    AllCoords[iteration, n][i] = np.zeros([n, 2])
                    for j in range(0, n):
                        AllCoords[iteration, n][i][j, :] = \
                            [np.floor((combs[i, j] - 1) / N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]


    # Determines which string configurations have unique Z's (configurations not related by translational symmetry)
    print('Determining Unique Configs')
    UniqueCoefficientDictionary_zeros = {0: {0: np.zeros(n_high - n_low + 1)}}
    AllStringConfigs = {}
    UniqueStringConfigs = {}
    UniqueInitial_deg4_configs = {}
    UniqueInitialConfigEdges = {}
    ConjugacyClasses = {}
    for strings in range(1, max_strings + 1):
        print('Strings: ' + str(strings))
        AllStringConfigs[strings] = list(ComputeStringConfigs(N_x, strings))
        AllStringConfigs_temp = list(ComputeStringConfigs(N_x, strings))
        UniqueStringConfigs[strings] = {}
        UniqueCoefficientDictionary_zeros[strings] = {}
        UniqueInitial_deg4_configs[strings] = {}
        UniqueInitialConfigEdges[strings] = {}
        ConjugacyClasses[strings] = -1 * np.ones(np.shape(AllStringConfigs[strings])[0])
        ConjugacyClassIndex = 0
        while AllStringConfigs_temp:
            nodes_vec_OfInterest = AllStringConfigs_temp[0]
            nodes_OfInterest = np.array(nodes_vec_OfInterest).reshape(strings, 2)
            UniqueStringConfigs[strings][ConjugacyClassIndex] = nodes_vec_OfInterest
            UniqueCoefficientDictionary_zeros[strings][ConjugacyClassIndex] = np.zeros(n_high - n_low + 1)
            Initial_deg4_config, InitialConfigEdges = Find_Initial_deg4_config_and_InitialConfigEdges_qb(
                nodes_OfInterest, Square_Lattice_qb_Bare, Square_Lattice_qb_all, N_x, N_y)
            UniqueInitial_deg4_configs[strings][ConjugacyClassIndex] = Initial_deg4_config
            UniqueInitialConfigEdges[strings][ConjugacyClassIndex] = InitialConfigEdges
            for shift in range(N_x):
                nodes = np.mod(np.array(nodes_vec_OfInterest) + shift, N_x).reshape(strings, 2)
                nodes.sort(axis=1)
                nodes = nodes[nodes[:, 0].argsort()]
                nodes = (nodes.reshape(1, 2 * strings))[0].tolist()
                if nodes in AllStringConfigs_temp:
                    ConjugacyClasses[strings][AllStringConfigs[strings].index(nodes)] = ConjugacyClassIndex
                    AllStringConfigs_temp.remove(nodes)
            '''[AllStringConfigs[strings][j] for j in range(len(ConjugacyClasses[strings])) if
            ConjugacyClasses[strings][j] == ConjugacyClassIndex]'''
            ConjugacyClassIndex += 1

    # Computes boundary state
    print('Computing rho')
    def Compute_rho_parallel(iteration):
        return Compute_rho(N_x, N_y, N_faces, deg2_weight, gamma, loop_weight, n_low, n_high, samples,
            AllCoords, iteration, Square_Lattice, UniqueCoefficientDictionary_zeros, ConjugacyClasses,
            UniqueStringConfigs, UniqueInitial_deg4_configs, UniqueInitialConfigEdges, sX, sY, sZ, max_strings)
    pool = mp.Pool()
    rhos = pool.map(Compute_rho_parallel, range(iterations))  # Unnormalized and normalized rho's
    pool.close()
    pool.join()

    # Averages the unnormalized and normalized boundary states
    print('Averaging rhos')
    rho = sparse.csr_matrix((2 ** N_x, 2 ** N_x))  # Average of unnormalized boundary states
    for i in range(iterations):
        rho = rho + rhos[i][0] / iterations
    rho = rho / iterations
    rho = rho / (rho.diagonal().sum())
    rho = rho.toarray()
    # rho = nearestPD(rho)
    # rho = rho/np.trace(rho)

    rho2 = sparse.csr_matrix((2 ** N_x, 2 ** N_x))  # Average of normalized boundary states
    for i in range(iterations):
        rho2 = rho2 + rhos[i][1]
    rho2 = rho2/iterations
    rho2 = rho2 / (rho2.diagonal().sum())
    rho2 = rho2.toarray()
    # rho2 = nearestPD(rho2)
    # rho2 = rho2/np.trace(rho2)

    # Finds boundary Hamiltonians
    # D, U = LA.eig(rho)
    # H = -1/2*U.dot(LA.logm(np.diag(D))+LA.logm(np.diag(D)).conj().T).dot(LA.inv(U))
    D, U = np.linalg.eigh(rho)
    H = -1 * U.dot(np.diag(np.log(D.astype(complex)))).dot(LA.inv(U))

    # D2, U2 = LA.eig(rho2)
    # H2 = -1/2*U2.dot(LA.logm(np.diag(D2))+LA.logm(np.diag(D2)).conj().T).dot(LA.inv(U2))
    D2, U2 = np.linalg.eigh(rho2)
    H2 = -1 * U2.dot(np.diag(np.log(D2.astype(complex)))).dot(LA.inv(U2))


    # Determines properties of boundary Hamiltonian
    print('Determining Hamiltonian Properties')
    TwoPointFunctions = Compute_TwoPointFunctions(rho, N_x)     # Two point functions
    A_r = Compute_A_r(H, N_x)                                   # Heisenberg amplitudes
    d_n = Compute_d_n(H, N_x, sX, sY, sZ)                       # n-qubit interaction strength

    TwoPointFunctions2 = Compute_TwoPointFunctions(rho2, N_x)   # Two point functions
    A_r2 = Compute_A_r(H2, N_x)                                 # Heisenberg amplitudes
    d_n2 = Compute_d_n(H2, N_x, sX, sY, sZ)                     # n-qubit interaction strength


    # Prints results
    t2 = time.time()
    print('\n \n \n \n ALL DONE!!!')
    print('max_strings = ' + str(max_strings))
    print('Runtime: ' + str(t2 - t))
    print('\n \n Two Point Functions:')
    for func in TwoPointFunctions:
        print(np.real_if_close(func))
    print('\n Two Point Functions2:')
    for func in TwoPointFunctions2:
        print(np.real_if_close(func))

    print('\n Heisenberg Amplitudes:')
    for amp in A_r:
        print(np.real_if_close(amp))
    print('\n Heiseberg Amplitudes2:')
    for amp in A_r2:
        print(np.real_if_close(amp))

    print('\n n-qubit Interaction Strengths:')
    for strength in d_n:
        print(np.real_if_close(strength))
    print('\n n-qubit Interaction Strengths2:')
    for strength in d_n2:
        print(np.real_if_close(strength))


    # Prints results to text file
    with open(os.path.basename(__file__) + ".txt", "w") as text_file:
        print('N_x = ' + str(N_x), file=text_file)
        print('N_y = ' + str(N_y), file=text_file)
        print('deg2_weight = ' + str(deg2_weight), file=text_file)
        print('CR_weight = ' + str(CR_weight), file=text_file)
        print('CP_weight = ' + str(CP_weight), file=text_file)
        print('loop_weight = ' + str(loop_weight), file=text_file)

        print('epsilon = ' + str(epsilon), file=text_file)
        print('n_low = ' + str(n_low), file=text_file)
        print('n_high = ' + str(n_high), file=text_file)
        print('samples = ' + str(samples), file=text_file)
        print('epochs = ' + str(epochs), file=text_file)
        print('iterations = ' + str(iterations), file=text_file)
        print('range_samples = ' + str(range_samples), file=text_file)
        print('deg4_samples = ' + str(deg4_samples), file=text_file)
        print('max_strings = ' + str(max_strings), file=text_file)
        print('Runtime: ' + str(t2 - t), file=text_file)

        print('\n Two Point Functions:', file=text_file)
        for func in TwoPointFunctions:
            print(np.real_if_close(func), file=text_file)
        print('\n Two Point Functions2:', file=text_file)
        for func in TwoPointFunctions2:
            print(np.real_if_close(func), file=text_file)
        print('\n Heisenberg Amplitudes:', file=text_file)
        for amp in A_r:
            print(np.real_if_close(amp), file=text_file)
        print('\n Heisenberg Amplitudes2:', file=text_file)
        for amp in A_r2:
            print(np.real_if_close(amp), file=text_file)
        print('\n n-qubit Interaction Strengths:', file=text_file)
        for strength in d_n:
            print(np.real_if_close(strength), file=text_file)
        print('\n n-qubit Interaction Strengths2:', file=text_file)
        for strength in d_n2:
            print(np.real_if_close(strength), file=text_file)

    # Prints boundary states to text files
    np.savetxt(str(os.path.basename(__file__) + "_rho.txt"), rho, delimiter='\t')
    np.savetxt(str(os.path.basename(__file__) + "_rho2.txt"), rho2, delimiter='\t')
