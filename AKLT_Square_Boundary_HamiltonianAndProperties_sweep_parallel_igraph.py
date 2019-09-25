# Import necessary packages
import numpy as np
from matplotlib import pyplot as plt
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
    for rung in range(2, N_y + 1 + 1):
        x_squares = np.append(x_squares, np.arange(0.0, w*N_x, w))
        y_squares = np.append(y_squares, (rung - 1)*h*np.ones(N_x))
        y_squares[-1] = y_squares[-1] + 0.35*h
        Square_Lattice.add_vertices(N_x)
        Square_Lattice.vs.select(range((rung - 1)*N_x, rung*N_x))["x"] = x_squares[(rung - 1)*N_x:rung*N_x]
        Square_Lattice.vs.select(range((rung - 1)*N_x, rung*N_x))["y"] = y_squares[(rung - 1)*N_x:rung*N_x]
        Square_Lattice.add_edges(zip(range((rung - 1)*N_x, rung*N_x-1), range((rung - 1)*N_x + 1, rung*N_x)))
        Square_Lattice.add_edge((rung - 1)*N_x, rung*N_x - 1)
        for ladder in range(0, N_x):
            Square_Lattice.add_edge(((rung - 1)*N_x) + ladder, ((rung - 2)*N_x) + ladder)

    x_squares = np.append(x_squares, np.arange(0.0, w*N_x, w))
    y_squares = np.append(y_squares, -1*h*np.ones(N_x))
    y_squares[-1] = y_squares[-1] + 0.35*h
    Square_Lattice.add_vertices(N_x)
    Square_Lattice.vs.select(range((N_y + 1)*N_x, (N_y + 2)*N_x))["x"] = x_squares[(N_y + 1)*N_x: (N_y + 2)*N_x]
    Square_Lattice.vs.select(range((N_y + 1)*N_x, (N_y + 2)*N_x))["y"] = y_squares[(N_y + 1)*N_x: (N_y + 2)*N_x]
    Square_Lattice.add_edges(zip(range(0, N_x), range(N_x*(N_y + 1), N_x*(N_y + 2))))
    Square_Lattice.es["weight"] = np.ones(Square_Lattice.ecount()).astype(int)

    # Square lattice with no edges
    Square_Lattice_Bare = Square_Lattice.copy()
    Square_Lattice_Bare.delete_edges(range(Square_Lattice_Bare.ecount()))

    return Square_Lattice, Square_Lattice_Bare

def ConstructSquare_qb_Lattices(N_x, N_y, w, h):
    # Constructs cylindrical square lattices of qubits

    # Square lattice of qubits
    Square_Lattice_qb = ig.Graph()     # Vertices in igraph are indexed from 0
    x_squares_qb = np.zeros(4*N_x)
    y_squares_qb = np.zeros(4*N_x)
    for i in range(0, N_x):
        x_squares_qb[(4*i): 4*(i+1)] = i*w
        y_squares_qb[(4*i): 4*(i+1)] = 0*h
    x_squares_qb[1::4] += 0.2*w
    x_squares_qb[3::4] += - 0.2*w
    y_squares_qb[0::4] += 0.2*h
    y_squares_qb[2::4] += - 0.2*h
    y_squares_qb[-4:] += 0.35*h
    y_squares_qb[-1] += - 0.05*h
    y_squares_qb[-3] += 0.03*h
    Square_Lattice_qb.add_vertices(4*N_x)
    Square_Lattice_qb.vs["x"] = x_squares_qb[0:4*N_x]
    Square_Lattice_qb.vs["y"] = y_squares_qb[0:4*N_x]
    Square_Lattice_qb.add_edges(zip(range(1, (4*N_x-6), 4), range(7, 4*N_x, 4)))
    Square_Lattice_qb.add_edge(3, 4*N_x-3)
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
        Square_Lattice_qb.vs.select(range((rung - 1)*4*N_x, rung*4*N_x))["x"] = x_squares_qb[(rung-1)*4*N_x: rung*4*N_x]
        Square_Lattice_qb.vs.select(range((rung - 1)*4*N_x, rung*4*N_x))["y"] = y_squares_qb[(rung-1)*4*N_x: rung*4*N_x]
        Square_Lattice_qb.add_edges(zip(range((rung-1)*4*N_x+1, rung*4*N_x-6, 4),
                                        range((rung-1)*4*N_x+7, rung*4*N_x, 4)))
        Square_Lattice_qb.add_edge((rung-1)*4*N_x+3, rung*4*N_x-3)
        for ladder in range(0, N_x):
            Square_Lattice_qb.add_edge((rung-1)*4*N_x+2+4*ladder, (rung-2)*4*N_x+4*ladder)
    nodes_to_add = np.linspace(3, 4*N_x-1, N_x)-1
    for i in range(1, N_x+1):
        Square_Lattice_qb.add_vertex(x=(i-1)*w, y=-1*h + 0.35*h*int(i == N_x) )
    for i in range(1, N_x+1):
        Square_Lattice_qb.add_edge(int(nodes_to_add[i-1]), 4*N_x*(N_y+1)+i-1)
    Square_Lattice_qb.es["weight"] = np.ones(Square_Lattice_qb.ecount())

    # Square lattice of qubits with all neighboring qubits connected
    Square_Lattice_qb_all = Square_Lattice_qb.copy()
    for i in range(1, N_x*(N_y+1)+1):
        index = 4*(i-1)
        Square_Lattice_qb_all.add_edge(index, index+1, weight=0)
        Square_Lattice_qb_all.add_edge(index, index+2, weight=0)
        Square_Lattice_qb_all.add_edge(index, index+3, weight=0)
        Square_Lattice_qb_all.add_edge(index+1, index+2, weight=0)
        Square_Lattice_qb_all.add_edge(index+1, index+3, weight=0)
        Square_Lattice_qb_all.add_edge(index+2, index+3, weight=0)

    # Square lattice of qubits with all neighboring qubits connected via kinked lines
    Square_Lattice_qb_all_kinked = Square_Lattice_qb.copy()
    for i in range(1, N_x*(N_y+1)+1):
        index = 4*(i-1)
        Square_Lattice_qb_all_kinked.add_edge(index, index+1, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index, index+3, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index+1, index+2, weight=0)
        Square_Lattice_qb_all_kinked.add_edge(index+2, index+3, weight=0)

    #Square lattice of qubits with no edges
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
                weight=Square_Lattice.es.select(Square_Lattice.get_eid(path[j], path[j + 1]))['weight'])

    return nodes, Lattice_Initial

def InitializeSquareLattice_qb(nodes, Square_Lattice_qb_Bare, Square_Lattice_qb_all, N_x, N_y):
    # Constructs initial square lattice that connects the vertices in 'nodes'

    # USE SHORTEST_PATHS
    # Re-orders nodes according to shortest length between the two vertices
    nodes = nodes + 4*N_x*(N_y+1)
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
        for j in range(0, len(path)-1):
            Lattice_Initial.add_edge(path[j], path[j + 1],
            weight=Square_Lattice_qb_all.es.select(Square_Lattice_qb_all.get_eid(path[j], path[j + 1]))['weight'][0])
        Lattice_temp.delete_edges(Lattice_temp.get_eids(zip(path[0:-1], path[1:])))
        for v in path:
            Lattice_temp.delete_edges(Lattice_temp.incident(v))

    # Determines the configuration of deg4 vertices in Lattice_Initial
    Initial_deg4_config = {}
    Initial_deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_Initial, N_x, N_y)
    for vertex in Initial_deg4_clusters:
        if Lattice_Initial.are_connected(vertex, vertex+1):
            Initial_deg4_config[vertex] = 1
        elif Lattice_Initial.are_connected(vertex, vertex+3):
            Initial_deg4_config[vertex] = 3
        elif Lattice_Initial.are_connected(vertex, vertex+2):
            Initial_deg4_config[vertex] = 2

    # Makes initial lattice kinked (needed for face flips)
    for i in range(N_x*(N_y+1)):
        if len(Lattice_Initial.subgraph([4*i, 4*i+1, 4*i+2, 4*i+3]).components()) < 4:
            S = Lattice_Initial.subgraph([4*i, 4*i+1, 4*i+2, 4*i+3])
            if len(S.components()) == 3 and Lattice_Initial.are_connected(4*i, 4*i+2):
                Lattice_Initial.delete_edges( (4*i, 4*i+2) )
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4*i, 4*i+1+2*x, weight=0)
                Lattice_Initial.add_edge(4*i+1+2*x, 4*i+2, weight=0)
            elif len(S.components()) == 3 and Lattice_Initial.are_connected(4*i+1, 4*i+3):
                Lattice_Initial.delete_edges( (4*i+1, 4*i+3) )
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4*i+1, 4*i+2*x, weight=0)
                Lattice_Initial.add_edge(4*i+2*x, 4*i+3, weight=0)
            elif len(S.components()) == 2 and Lattice_Initial.are_connected(4*i, 4*i+2) \
            and Lattice_Initial.are_connected(4*i+1, 4*i+3):
                Lattice_Initial.delete_edges( ((4*i, 4*i+2), (4*i+1, 4*i+3)) )
                np.random.seed()
                x = np.random.randint(2)
                Lattice_Initial.add_edge(4*i, 4*i+1+2*x, weight=0)
                Lattice_Initial.add_edge(4*i+2, 4*i+3-2*x, weight=0)

    return nodes, Lattice_Initial, Initial_deg4_config

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

def FlipSquareLatticeFaces_qb(Lattice, coords, N_x):
    # Flips the faces in coords

    for x, y in coords:
        v = 4*N_x*(y-1) + 4*(x-1)   # Lower left vertex of face to be flipped

        # Edges with weight 0 and weight 1 to be flipped
        if x == N_x:
            to_flip0 = np.array([[v, v+1], [4*N_x*(y-1), 4*N_x*(y-1)+3],
                                [4*N_x*y+2, 4*N_x*y+3], [v+4*N_x+1, v+4*N_x+2]]).astype(int)
            to_flip1 = np.array([[v+1, 4*N_x*(y-1)+3], [4*N_x*(y-1), 4*N_x*y+2],
                                 [v+1+4*N_x, 4*N_x*y+3], [v, v+4*N_x+2]]).astype(int)
        else:
            to_flip0 = np.array([[v, v+1], [v+7, v+4], [v+4+4*N_x+2, v+4+4*N_x+3], [v+4*N_x+1, v+4*N_x+2]]).astype(int)
            to_flip1 = np.array([[v+1, v+7], [v+4, v+4+4*N_x+2], [v+4*N_x+1, v+4+4*N_x+3], [v, v+4*N_x+2]]).astype(int)

        # Flips edges
        for vs_0, vs_1 in zip(to_flip0, to_flip1):
            if Lattice.are_connected(vs_0[0], vs_0[1]):
                Lattice.delete_edges((vs_0[0], vs_0[1]))
            else:
                Lattice.add_edge(vs_0[0], vs_0[1], weight=0)

            if Lattice.are_connected(vs_1[0], vs_1[1]):
                Lattice.delete_edges((vs_1[0], vs_1[1]))
            else:
                Lattice.add_edge(vs_1[0], vs_1[1], weight=1)

    return Lattice

def DetermineSquareLatticeValid(Lattice, nodes):
    # Determines if the configuration of Lattice is valid

    valid = not (Lattice.degree().count(3) >= 1)
    for v1, v2 in nodes:
        if Lattice.shortest_paths(v1, v2)[0][0] == float('inf'):
            valid = False
    return valid

def DetermineSquareLatticeValid_qb(Lattice, nodes):
    # Determines if the configuration of Lattice is valid

    valid = not (len([deg for deg in Lattice.degree() if deg >= 3]) >= 1)
    for v1, v2 in nodes:
        if Lattice.shortest_paths(v1, v2)[0][0] == float('inf'):
            valid = False
    return valid

def DetermineSquareLatticeValid_NoStrings(Lattice):
    # Determines if the configuration of Lattice (no strings, just loops) is valid

    valid = not (Lattice.degree().count(3) >= 3)
    return valid

def DetermineSquareLatticeValid_NoStrings_qb(Lattice):
    # Determines if the configuration of Lattice (no strings, just loops) is valid

    valid = not (len([deg for deg in Lattice.degree() if deg >= 3]) >= 1)
    return valid

def ComputeSquareLattice_qb_deg4_clusters(Lattice, N_x, N_y):
    # Determines the location of deg4 clusters in Lattice

    # Finds clusters with all vertices having degree 2
    deg_equals2 = [int(deg == 2) for deg in Lattice.degree()[0:4*N_x*(N_y+1)]]
    deg4_clusters = np.array([4*cl for cl in range(N_x*(N_y+1)) if deg_equals2[4*cl:4*cl+4] == [1, 1, 1, 1]])

    # Removes clusters that aren't true degree 4 clusters, counts interior loops
    interior_loops = 0
    for index, cluster in enumerate(deg4_clusters):
        S = Lattice.subgraph([cluster, cluster+1, cluster+2, cluster+3])
        if len(S.components()) != 2:
            deg4_clusters[index] = -1
        if np.min(S.degree()) == 2:
            interior_loops += 1
    deg4_clusters = [cluster for cluster in deg4_clusters if cluster != -1]

    return np.array(deg4_clusters).astype(int), interior_loops

def MakeCP1(Lattice, vertex):
    # Makes a corner pass at vertex

    if Lattice.are_connected(vertex, vertex + 3):
        Lattice.delete_edges( ((vertex, vertex + 3), (vertex + 1, vertex + 2)) )
        Lattice.add_edge(vertex, vertex + 1, weight=0)
        Lattice.add_edge(vertex + 2, vertex + 3, weight=0)
    return Lattice

def MakeCP2(Lattice, vertex):
    # Makes alternate corner pass at vertex

    if Lattice.are_connected(vertex, vertex + 1):
        Lattice.delete_edges( ((vertex, vertex + 1), (vertex + 2, vertex + 3)) )
        Lattice.add_edge(vertex, vertex + 3, weight=0)
        Lattice.add_edge(vertex + 1, vertex + 2, weight=0)
    return Lattice

def MakeCR(Lattice, vertex):
    # Makes crossing at vertex

    if Lattice.are_connected(vertex, vertex + 1):
        Lattice.delete_edges( ((vertex, vertex + 1), (vertex + 2, vertex + 3)) )
        Lattice.add_edge(vertex, vertex + 2, weight=0)
        Lattice.add_edge(vertex + 1, vertex + 3, weight=0)
    else:
        Lattice.delete_edges( ((vertex, vertex + 3), (vertex + 1, vertex + 2)) )
        Lattice.add_edge(vertex, vertex + 2, weight=0)
        Lattice.add_edge(vertex + 1, vertex + 3, weight=0)
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

def AddNCLoop_qb(Lattice, N_x, rung):
    # Adds to Lattice a noncontractible loop at rung

    if Lattice.are_connected((rung-1)*4*N_x+1, (rung-1)*4*N_x+2):
        Lattice.delete_edges(((rung-1)*4*N_x+1, (rung-1)*4*N_x+2))
    else:
        Lattice.add_edge((rung-1)*4*N_x+1, (rung-1)*4*N_x+2, weight=0)
    if Lattice.are_connected((rung-1)*4*N_x+2, (rung-1)*4*N_x+3):
        Lattice.delete_edges(((rung-1)*4*N_x+2, (rung-1)*4*N_x+3))
    else:
        Lattice.add_edge((rung-1)*4*N_x+2, (rung-1)*4*N_x+3, weight=0)

    if Lattice.are_connected((rung - 1) * 4 * N_x + 3, rung * 4 * N_x - 3):
        Lattice.delete_edges(((rung - 1) * 4 * N_x + 3, rung * 4 * N_x - 3))
    else:
        Lattice.add_edge((rung - 1) * 4 * N_x + 3, rung * 4 * N_x - 3, weight=1)

    for i in range(1, N_x):
        if Lattice.are_connected((rung-1)*4*N_x+2+4*(i-1)-1, (rung-1)*4*N_x+8+4*(i-1)-1):
            Lattice.delete_edges( ((rung-1)*4*N_x+2+4*(i-1)-1, (rung-1)*4*N_x+8+4*(i-1)-1) )
        else:
            Lattice.add_edge((rung-1)*4*N_x+2+4*(i-1)-1, (rung-1)*4*N_x+8+4*(i-1)-1, weight=1)

        if Lattice.are_connected((rung-1)*4*N_x+4*i+1, (rung-1)*4*N_x+4*i+2):
            Lattice.delete_edges( ((rung-1)*4*N_x+4*i+1, (rung-1)*4*N_x+4*i+2) )
        else:
            Lattice.add_edge((rung-1)*4*N_x+4*i+1, (rung-1)*4*N_x+4*i+2, weight=0)
        if Lattice.are_connected((rung-1)*4*N_x+4*i+2, (rung-1)*4*N_x+4*i+3):
            Lattice.delete_edges( ((rung-1)*4*N_x+4*i+2, (rung-1)*4*N_x+4*i+3) )
        else:
            Lattice.add_edge((rung-1)*4*N_x+4*i+2, (rung-1)*4*N_x+4*i+3, weight=0)




    return Lattice

def ComputeAllPairs(lst):
    # Computes all possible pairs of elements in lst

    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in ComputeAllPairs(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1,len(lst)):
            pair = [a,lst[i]]
            for rest in ComputeAllPairs(lst[1:i]+lst[i+1:]):
                yield pair + rest

def ComputeStringConfigs(N_x, k):
    # Computes the start/end nodes of all possible configurations with k strings

    lst = list(range(N_x))
    lsts = list(combinations(lst, 2*k))
    for lst2 in lsts:
        lst2 = list(lst2)
        a = lst2[0]
        for i in range(1, len(lst2)):
            pair = [a, lst2[i]]
            for rest in ComputeAllPairs(lst2[1:i]+lst2[i+1:]):
                yield pair + rest

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

def ComputeAs_component_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma, loop_weight,
                               n_low, n_high, samples, iteration):
    # Computes components of A (configuration with no strings)

    print('Iteration: ' + str(iteration+1))
    rung = np.mod(iteration, N_y+1) + 1
    As_component = np.zeros(n_high-n_low+1)

    for n in range(n_low, n_high+1):
        if n == 0:
            # Contribution from contractible configuration (no faces flipped)
            As_component[0] += 1

            # Contribution from noncontractible configuration
            Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)
            deg2 = Lattice_nc.degree().count(2)
            loops = len((Lattice_nc.subgraph(
                [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components())
            As_component[0] += (deg2_weight)**(deg2)*loop_weight**(loops)
        else:
            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1)) > np.log(samples):  # (N_faces choose n) > samples
                avg = 1
                combs = ComputeRandomUniqueCombinations(N_faces, n, samples)
            else:
                avg = 0
                combs = np.reshape(list(combinations(range(1, N_faces+1), n)), (-1, n))

            contributions = np.zeros(np.shape(combs)[0])

            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(0, np.shape(combs)[0]):
                # Finds coordinates of faces to be flipped in loop configuration
                coords = np.zeros([n, 2])
                for j in range(0, n):
                    coords[j, :] = [np.floor((combs[i, j] - 1)/N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]

                # Flips faces, contractible configuration
                Lattice_c = FlipSquareLatticeFaces(Lattice_Initial.copy(), coords, N_x)
                # Flips faces, noncontractible configuration
                Lattice_nc = AddNCLoop(Lattice_c.copy(), N_x, rung)

                # Contribution from contractible configuration
                if DetermineSquareLatticeValid_NoStrings(Lattice_c):
                    deg2 = Lattice_c.degree().count(2)
                    deg4 = Lattice_c.degree().count(4)
                    loops = len((Lattice_c.subgraph(
                        [nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components())
                    contributions[i] += (deg2_weight)**(deg2)*gamma**(deg4)*loop_weight**(loops)

                # Contribution from noncontractible configuration
                if DetermineSquareLatticeValid_NoStrings(Lattice_nc):
                    deg2 = Lattice_nc.degree().count(2)
                    deg4 = Lattice_nc.degree().count(4)
                    loops = len((Lattice_nc.subgraph(
                        [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components())
                    contributions[i] += (deg2_weight)**(deg2)*gamma**(deg4)*loop_weight**(loops)

            #Averages As terms if necessary
            if avg == 1:
                As_component[n-n_low] = np.exp(np.mean(np.log(contributions)))*\
                    np.exp(gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1))
                # equivalent to log_avg(As(n,iter))*nchoosek(N_faces,n)
            else:
                As_component[n - n_low] = sum(contributions)


    return As_component

def ComputeBs_component_0_Square(N_x, Lattice_Initial, deg2_weight, gamma, loop_weight, parity, strings, rung):
    # Computes 0th component of B (configurations with 1 string)

    Bs_component_0 = 0

    # Contractible configuration
    Lattice_c = Lattice_Initial.copy()
    # Noncontractible configuration
    Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)

    # Adds contribution from contractible configuration
    deg2 = Lattice_c.degree().count(2)
    deg4 = Lattice_c.degree().count(4)
    loops = len((Lattice_c.subgraph(
        [nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components()) - strings
    Bs_component_0 += (-1)**(parity)*(deg2_weight**deg2)*(gamma**deg4)*(loop_weight**loops)  # Sum of Boltzmann weights

    # Adds contribution from noncontractible configuration
    deg2 = Lattice_nc.degree().count(2)
    deg4 = Lattice_nc.degree().count(4)
    loops = len((Lattice_nc.subgraph(
        [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components()) - strings
    Bs_component_0 += (-1)**(parity)*(deg2_weight**deg2)*(gamma**deg4)*(loop_weight**loops)  # Sum of Boltzmann weights

    return Bs_component_0

def ComputeBs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma, loop_weight, parity,
                                            nodes, strings, combs, rung, sample):
    # Computes contribution to B (configurations with 1 string) by flipping faces of combs

    Bs_component_contribution = 0

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
    if DetermineSquareLatticeValid(Lattice_c, nodes):
        deg2 = Lattice_c.degree().count(2)
        deg4 = Lattice_c.degree().count(4)
        loops = len((Lattice_c.subgraph(
            [nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components()) - strings
        Bs_component_contribution += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

    # Adds contribution from noncontractible lattice configuration
    if DetermineSquareLatticeValid(Lattice_nc, nodes):
        deg2 = Lattice_nc.degree().count(2)
        deg4 = Lattice_nc.degree().count(4)
        loops = len((Lattice_nc.subgraph(
            [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components()) - strings
        Bs_component_contribution += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

    return Bs_component_contribution

def ComputeBs_component_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma, loop_weight, parity,
                               nodes, strings, n_low, n_high, samples, iteration):
    # Computes components of B (configuration with 1 string)

    print('Iteration: ' + str(iteration+1))
    rung = np.mod(iteration, N_y+1) + 1
    Bs_component = np.zeros(n_high-n_low+1)

    for n in range(n_low, n_high+1):
        if n == 0:
            # Contractible configuration
            Lattice_c = Lattice_Initial.copy()
            # Noncontractible configuration
            Lattice_nc = AddNCLoop(Lattice_Initial.copy(), N_x, rung)

            # Contribution from contractible configuration
            deg2 = Lattice_c.degree().count(2)
            deg4 = Lattice_c.degree().count(4)
            loops = len((Lattice_c.subgraph(
                [nod for (nod, deg) in enumerate(Lattice_c.degree()) if deg >= 1])).components()) - strings
            Bs_component[0] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

            # Contribution from noncontractible configuration
            deg2 = Lattice_nc.degree().count(2)
            deg4 = Lattice_nc.degree().count(4)
            loops = len((Lattice_nc.subgraph(
                [nod for (nod, deg) in enumerate(Lattice_nc.degree()) if deg >= 1])).components()) - strings
            Bs_component[0] += (-1)**(parity)*deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)
        else:
            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1)) > \
                    np.log(samples):    # equivalent to nchoosek(N_faces, n) > samples
                avg = 1
                combs = ComputeRandomUniqueCombinations(N_faces, n, samples)
            else:
                avg = 0
                combs = np.reshape(list(combinations(range(1, N_faces+1), n)), (-1, n))

            contributions = np.zeros(np.shape(combs)[0])

            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(0, np.shape(combs)[0]):
                # Finds coordinates of faces to be flipped in loop configuration
                coords = np.zeros([n, 2])
                for j in range(0, n):
                    coords[j, :] = [np.floor((combs[i, j] - 1)/N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]

                # Flips faces, contractible config
                Lattice_c = FlipSquareLatticeFaces(Lattice_Initial.copy(), coords, N_x)
                # Flips faces, noncontractible config
                Lattice_nc = AddNCLoop(Lattice_c.copy(), N_x, rung)

                # Contribution from contractible configuration
                if DetermineSquareLatticeValid(Lattice_c, nodes):
                    deg2 = Lattice_c.degree().count(2)
                    deg4 = Lattice_c.degree().count(4)
                    loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                          if deg >= 1])).components()) - strings
                    contributions[i] += deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

                # Contribution from noncontractible configuration
                if DetermineSquareLatticeValid(Lattice_nc, nodes):
                    deg2 = Lattice_nc.degree().count(2)
                    deg4 = Lattice_nc.degree().count(4)
                    loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                           if deg >= 1])).components()) - strings
                    contributions[i] += deg2_weight**(deg2)*gamma**(deg4)*loop_weight**(loops)

            if avg == 1:
                Bs_component[n-n_low] = (-1)**parity*np.exp(np.mean(np.log(contributions)))*\
                    np.exp(gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1))
                # equivalent to Bs(n,iter)/samples*nchoosek(N_faces,n)
            else:
                Bs_component[n - n_low] = (-1)**parity*sum(contributions)

    return Bs_component

def ComputeCs_component_0_Square(N_x, N_y, Lattice_Initial, Initial_deg4_config, deg4_samples, deg2_weight,
                                 CP_weight, CR_weight, loop_weight, parity, rung):
    # Computes 0th component of C (configurations with k strings)

    Cs_component_0 = 0

    # Contractible configuration
    Lattice_c = Lattice_Initial.copy()
    # Nonontractible configuration
    Lattice_nc = AddNCLoop_qb(Lattice_Initial.copy(), N_x, rung)

    # Contribution of contractible configuration
    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_c, N_x, N_y)
    deg4 = len(deg4_clusters)
    Lattice_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
        Process_deg4(Lattice_c.copy(), deg4_clusters, Initial_deg4_config)
    weight1_edges = np.sum(Lattice_c_Before_deg4_Flips.es['weight'])
    if len(deg4_clusters) >= 1:
        # Analyzes different degree 4 configurations
        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
        deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
        for j in range(np.shape(deg4_configs)[0]):
            Lattice_c, CP, CR = Flip_deg4_clusters(Lattice_c_Before_deg4_Flips.copy(), deg4_clusters, deg4_configs,
                                                   j, CP0, CR0)
            if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                # Computes energy
                deg2 = weight1_edges - strings - 2 * deg4
                loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                      if deg >= 1])).components()) - interior_loops - strings
                deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                         * (loop_weight ** loops) * deg4_avg_factor  # Sum of Boltzmann weights
            if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                # Calculates log average
                Cs_component_0 += ((-1)**parity)*3**len(deg4_clusters)* \
                                  np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
            else:
                Cs_component_0 += ((-1) ** parity) * np.sum(deg4_contributions)
    else:
        Lattice_c = Lattice_c_Before_deg4_Flips.copy()
        if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
            # Computes energy
            CP = CP0
            CR = CR0
            deg2 = weight1_edges - strings - 2*deg4
            loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                  if deg >= 1])).components()) - interior_loops - strings
            Cs_component_0 += ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) * \
                              (loop_weight ** loops)  # Sum of Boltzmann weights

    # Contribution of noncontractible configuration
    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_nc, N_x, N_y)
    deg4 = len(deg4_clusters)
    Lattice_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
        Process_deg4(Lattice_nc.copy(), deg4_clusters, Initial_deg4_config)
    weight1_edges = np.sum(Lattice_nc_Before_deg4_Flips.es['weight'])
    if len(deg4_clusters) >= 1:
        # Analyzes different degree 4 configurations
        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
        deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
        for j in range(np.shape(deg4_configs)[0]):
            Lattice_nc, CP, CR = Flip_deg4_clusters(Lattice_nc_Before_deg4_Flips.copy(), deg4_clusters, deg4_configs,
                                                    j, CP0, CR0)
            if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                # Computes energy
                deg2 = weight1_edges - strings - 2 * deg4
                loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                      if deg >= 1])).components()) - interior_loops - strings
                deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                         * (loop_weight ** loops) * deg4_avg_factor  # Sum of Boltzmann weights
            if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                # Calculates log average
                Cs_component_0 += ((-1) ** parity) * 3 ** len(deg4_clusters) * \
                                   np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
            else:
                Cs_component_0 += ((-1) ** parity) * np.sum(deg4_contributions)
    else:
        Lattice_nc = Lattice_nc_Before_deg4_Flips.copy()
        if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
            # Computes energy
            CP = CP0
            CR = CR0
            deg2 = weight1_edges - strings - 2*deg4
            loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                  if deg >= 1])).components()) - interior_loops - strings
            Cs_component_0 += ((-1) ** parity) * (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) *\
                              (loop_weight ** loops)  # Sum of Boltzmann weights

    return Cs_component_0

def ComputeCs_component_contribution_Square(N_x, N_y, Lattice_Initial, Initial_deg4_config, deg4_samples,
        deg2_weight, CP_weight, CR_weight, loop_weight, parity, combs, rung, sample):
    # Computes contribution to C (configurations with k strings) by flipping faces of combs

    Cs_component_contribution = 0

    # Finds coordinates of faces to be flipped in loop configuration
    n = np.shape(combs)[1]
    coords = np.zeros([n, 2])
    for j in range(0, n):
        coords[j, :] = [np.floor((combs[sample, j] - 1) / N_y) + 1, np.mod(combs[sample, j] - 1, N_y) + 1]

    # Flips faces of contractible config
    Lattice_c = FlipSquareLatticeFaces_qb(Lattice_Initial.copy(), coords, N_x)
    # Flips faces of noncontractible config
    Lattice_nc = AddNCLoop_qb(Lattice_c.copy(), N_x, rung)

    # Contribution of contractible configuration
    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_c, N_x, N_y)
    deg4 = len(deg4_clusters)
    Lattice_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
        Process_deg4(Lattice_c.copy(), deg4_clusters, Initial_deg4_config)
    weight1_edges = np.sum(Lattice_c_Before_deg4_Flips.es['weight'])
    if len(deg4_clusters) >= 1:
        # Analyzes different degree 4 configurations
        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
        deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
        for j in range(np.shape(deg4_configs)[0]):
            Lattice_c, CP, CR = Flip_deg4_clusters(Lattice_c_Before_deg4_Flips.copy(), deg4_clusters, deg4_configs, j,
                                                   CP0, CR0)
            if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                # Computes energy
                deg2 = weight1_edges - 2*deg4 - strings
                loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                       if deg >= 1])).components()) - interior_loops - strings
                deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                         * (loop_weight ** loops)  # Sum of Boltzmann weights
            if 3 ** len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                # Calculates log average
                Cs_component_contribution += ((-1)**parity)*3 ** len(deg4_clusters) * \
                                    np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
            else:
                Cs_component_contribution += ((-1)**parity)*np.sum(deg4_contributions)
    else:
        Lattice_c = Lattice_c_Before_deg4_Flips.copy()
        if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
            # Computes energy
            CP = CP0
            CR = CR0
            deg2 = weight1_edges - strings - 2*deg4
            loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                                   if deg >= 1])).components()) - interior_loops - strings
            Cs_component_contribution += ((-1)**parity)*(deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)\
                                         *(loop_weight**loops) # Sum of Boltzmann weights


    # Contribution of noncontractible configuration
    deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_nc, N_x, N_y)
    deg4 = len(deg4_clusters)
    Lattice_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
        Process_deg4(Lattice_nc.copy(), deg4_clusters, Initial_deg4_config)
    weight1_edges = np.sum(Lattice_nc_Before_deg4_Flips.es['weight'])
    if len(deg4_clusters) >= 1:
        # Analyzes different degree 4 configurations
        deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
        deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
        for j in range(np.shape(deg4_configs)[0]):
            Lattice_nc, CP, CR = Flip_deg4_clusters(Lattice_nc_Before_deg4_Flips.copy(), deg4_clusters, deg4_configs, j,
                                                    CP0, CR0)
            if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                # Computes energy
                deg2 = weight1_edges - 2*deg4 - strings
                loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                       if deg >= 1])).components()) - interior_loops - strings
                deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                         * (loop_weight ** loops)  # Sum of Boltzmann weights
        if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
            # Calculates log average
            Cs_component_contribution += ((-1) ** parity) * 3 ** len(deg4_clusters) * \
                                         np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
        else:
            Cs_component_contribution += ((-1) ** parity) * np.sum(deg4_contributions)
    else:
        Lattice_nc = Lattice_nc_Before_deg4_Flips.copy()
        if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
            # Computes energy
            CP = CP0
            CR = CR0
            deg2 = weight1_edges - strings - 2*deg4
            loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                                   if deg >= 1])).components()) - interior_loops - strings
            Cs_component_contribution += ((-1)**parity)*(deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR) \
                                         *(loop_weight**loops)  # Sum of Boltzmann weights


    return Cs_component_contribution

def ComputeCs_component_Square(N_x, N_y, N_faces, Lattice_Initial, Initial_deg4_config, deg4_samples, deg2_weight,
        CP_weight, CR_weight, loop_weight, parity, nodes, strings, n_low, n_high, samples, iteration):
    # Computes components of B (configurations with k strings)

    print('Iteration: ' + str(iteration+1))
    rung = np.mod(iteration, N_y+1) + 1
    Cs_component = np.zeros(n_high-n_low+1)

    for n in range(n_low, n_high+1):
        # print('Iteration: ' + str(iteration+1) + ', n: ' + str(n))
        if n == 0:
            # Contractible configuration
            Lattice_c = Lattice_Initial.copy()
            # Noncontractible configuration
            Lattice_nc = AddNCLoop_qb(Lattice_Initial.copy(), N_x, rung)

            # Contribution from contractible configuration
            deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_c, N_x, N_y)
            deg4 = len(deg4_clusters)
            Lattice_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                Process_deg4(Lattice_c.copy(), deg4_clusters, Initial_deg4_config)
            weight1_edges = np.sum(Lattice_c_Before_deg4_Flips.es['weight'])
            if len(deg4_clusters) >= 1:
                # Analyzes different degree 4 configurations
                deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
                for j in range(np.shape(deg4_configs)[0]):
                    Lattice_c, CP, CR = Flip_deg4_clusters(Lattice_c_Before_deg4_Flips.copy(), deg4_clusters,
                                                           deg4_configs, j, CP0, CR0)
                    if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                        # Computes energy
                        deg2 = weight1_edges - 2*deg4 - strings
                        loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                if deg >= 1])).components()) - interior_loops - strings
                        deg4_contributions[j] += (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR) \
                                                 *(loop_weight**loops)*deg4_avg_factor  # Sum of Boltzmann weights
                if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                    # Calculates log average
                    Cs_component[0] += ((-1)**parity)*3**len(deg4_clusters)*\
                                        np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
                else:
                    Cs_component[0] += ((-1)**parity)*np.sum(deg4_contributions)
            else:
                Lattice_c = Lattice_c_Before_deg4_Flips.copy()
                if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                    # Computes energy
                    CP = CP0
                    CR = CR0
                    deg2 = weight1_edges - 2*deg4 - strings
                    loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                            if deg >= 1])).components()) - interior_loops - strings
                    Cs_component[0] += ((-1)**parity)*(deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR) \
                                       *(loop_weight ** loops)  # Sum of Boltzmann weights

            # Contribution from noncontractible configuration
            deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_nc, N_x, N_y)
            deg4 = len(deg4_clusters)
            Lattice_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                Process_deg4(Lattice_nc.copy(), deg4_clusters, Initial_deg4_config)
            weight1_edges = np.sum(Lattice_nc_Before_deg4_Flips.es['weight'])
            if len(deg4_clusters) >= 1:
                # Analyzes different degree 4 configurations
                deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
                for j in range(np.shape(deg4_configs)[0]):
                    Lattice_nc, CP, CR = Flip_deg4_clusters(Lattice_nc_Before_deg4_Flips.copy(), deg4_clusters,
                                                            deg4_configs, j, CP0, CR0)
                    if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                        # Computes energy
                        deg2 = weight1_edges - 2*deg4 - strings
                        loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                if deg >= 1])).components()) - interior_loops - strings
                        deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                                 * (loop_weight ** loops) * deg4_avg_factor  # Sum of Boltzmann weights
                if 3 ** len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                    # Calculates log average
                    Cs_component[0] += ((-1) ** parity) * 3 ** len(deg4_clusters) * \
                                       np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
                else:
                    Cs_component[0] += ((-1) ** parity) * np.sum(deg4_contributions)
            else:
                Lattice_nc = Lattice_nc_Before_deg4_Flips.copy()
                if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                    # Computes energy
                    CP = CP0
                    CR = CR0
                    deg2 = weight1_edges - strings - 2*deg4
                    loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                            if deg >= 1])).components()) - interior_loops - strings
                    Cs_component[0] += ((-1)**parity)*(deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR) \
                                       *(loop_weight ** loops)  # Sum of Boltzmann weights
        else:
            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1)) > \
                    np.log(samples): # equivalent to nchoosek(N_faces, n) > samples
                avg = 1
                combs = ComputeRandomUniqueCombinations(N_faces, n, samples)
            else:
                avg = 0
                combs = np.reshape(list(combinations(range(1, N_faces+1), n)), (-1, n))

            contributions = np.zeros(np.shape(combs)[0])

            # Computes exp(-energy) for each loop config to be analyzed
            for i in range(0, np.shape(combs)[0]):
                # Finds coordinates of faces to be flipped in loop configuration
                coords = np.zeros([n, 2])
                for j in range(0, n):
                    coords[j, :] = [np.floor((combs[i, j] - 1)/N_y) + 1, np.mod(combs[i, j] - 1, N_y) + 1]

                Lattice_c = FlipSquareLatticeFaces_qb(Lattice_Initial.copy(), coords, N_x)
                Lattice_nc = AddNCLoop_qb(Lattice_c.copy(), N_x, rung)
                deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_c, N_x, N_y)
                deg4 = len(deg4_clusters)
                Lattice_c_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                    Process_deg4(Lattice_c.copy(), deg4_clusters, Initial_deg4_config)
                weight1_edges = np.sum(Lattice_c_Before_deg4_Flips.es['weight'])
                if len(deg4_clusters) >= 1:
                    # Analyzes different degree 4 configurations
                    deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                    deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
                    for j in range(np.shape(deg4_configs)[0]):
                        Lattice_c, CP, CR = Flip_deg4_clusters(Lattice_c_Before_deg4_Flips.copy(), deg4_clusters,
                                                               deg4_configs, j, CP0, CR0)
                        if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                            # Computes energy
                            deg2 = weight1_edges - 2 * deg4 - strings
                            loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                    if deg >= 1])).components()) - interior_loops - strings
                            deg4_contributions[j] += (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)\
                                                     *(loop_weight**loops)  # Sum of Boltzmann weights
                    if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                        # Calculates log average
                        contributions[i] += 3**len(deg4_clusters)*\
                                            np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
                    else:
                        contributions[i] += np.sum(deg4_contributions)
                else:
                    Lattice_c = Lattice_c_Before_deg4_Flips.copy()
                    if DetermineSquareLatticeValid_qb(Lattice_c, nodes):
                        # Computes energy
                        CP = CP0
                        CR = CR0
                        deg2 = weight1_edges - strings - 2*deg4
                        loops = len((Lattice_c.subgraph([nod for (nod, deg) in enumerate(Lattice_c.degree())
                                if deg >= 1])).components()) - interior_loops - strings
                        contributions[i] += (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)\
                                            *(loop_weight**loops)  # Sum of Boltzmann weights


                deg4_clusters, interior_loops = ComputeSquareLattice_qb_deg4_clusters(Lattice_nc, N_x, N_y)
                deg4 = len(deg4_clusters)
                Lattice_nc_Before_deg4_Flips, deg4_clusters, CP0, CR0 = \
                    Process_deg4(Lattice_nc.copy(), deg4_clusters, Initial_deg4_config)
                weight1_edges = np.sum(Lattice_nc_Before_deg4_Flips.es['weight'])
                if len(deg4_clusters) >= 1:
                    deg4_configs, deg4_avg_factor = Compute_deg4_configs(deg4_clusters, deg4_samples)
                    deg4_contributions = np.zeros(np.shape(deg4_configs)[0])
                    for j in range(np.shape(deg4_configs)[0]):
                        Lattice_nc, CP, CR = Flip_deg4_clusters(Lattice_nc_Before_deg4_Flips.copy(), deg4_clusters,
                                                                deg4_configs, j, CP0, CR0)
                        if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                            # Computes energy
                            deg2 = weight1_edges - 2*deg4 - strings
                            loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                    if deg >= 1])).components()) - interior_loops - strings
                            deg4_contributions[j] += (deg2_weight ** deg2) * (CP_weight ** CP) * (CR_weight ** CR) \
                                                     *(loop_weight**loops) # Sum of Boltzmann weights
                            '''
                            contributions[i] += (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)\
                                                *(loop_weight**loops)*deg4_avg_factor  # Sum of Boltzmann weights
                            '''
                    if 3**len(deg4_clusters) > deg4_samples and np.shape(np.nonzero(deg4_contributions))[1] > 0:
                        # Calculates log average
                        contributions[i] += 3**len(deg4_clusters)*\
                                            np.exp(np.mean(np.log(deg4_contributions[np.nonzero(deg4_contributions)])))
                    else:
                        contributions[i] += np.sum(deg4_contributions)
                else:
                    Lattice_nc = Lattice_nc_Before_deg4_Flips.copy()
                    if DetermineSquareLatticeValid_qb(Lattice_nc, nodes):
                        # Computes energy
                        CP = CP0
                        CR = CR0
                        deg2 = weight1_edges - strings - 2*deg4
                        loops = len((Lattice_nc.subgraph([nod for (nod, deg) in enumerate(Lattice_nc.degree())
                                if deg >= 1])).components()) - interior_loops - strings
                        contributions[i] += (deg2_weight**deg2)*(CP_weight**CP)*(CR_weight**CR)\
                                            *(loop_weight**loops)  # Sum of Boltzmann weights

            if avg == 1 and len(contributions[np.nonzero(contributions)]) > 0:
                Cs_component[n-n_low] = (-1)**parity*np.exp(np.mean(np.log(contributions[np.nonzero(contributions)])))*\
                    np.exp(gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1))
                # equivalent to Cs/samples*nchoosek(N_faces,n)
            else:
                Cs_component[n - n_low] = (-1)**parity*sum(contributions)

    return Cs_component

def Compute_deg4_configs(deg4_clusters, deg4_samples):
    # Computes possible configurations of deg4 vertices

    if deg4_samples < 3**len(deg4_clusters):
        np.random.seed()
        deg4_configs = np.zeros([deg4_samples, len(deg4_clusters)])
        i = 0
        while i < deg4_samples:
            deg4_configs[i, :] = np.random.randint(3, size=len(deg4_clusters))+1
            i = i + 1
            if i == deg4_samples:
                deg4_configs = np.unique(deg4_configs, axis=0)
                i = np.shape(deg4_configs)[0]
                deg4_configs = np.pad(deg4_configs, ((0, deg4_samples - i), (0, 0)), 'constant')
        deg4_avg_factor = 3 ** len(deg4_clusters) / deg4_samples

    else:
        deg4_avg_factor = 1
        deg4_configs = np.reshape(list([p for p in product([1, 2, 3], repeat=len(deg4_clusters))]),
                                  (-1, len(deg4_clusters)))

    return deg4_configs, deg4_avg_factor

def Process_deg4(Lattice, deg4_clusters, Initial_deg4_config):
    # Makes initial deg4 vertices have their initial orientation;
    # Calculates the number of initial corner passes (CP) and crossings (CR)

    CP0 = 0
    CR0 = 0
    Lattice_Before_deg4_Flips = Lattice.copy()

    for vertex in list(Initial_deg4_config.keys()):
        if vertex in deg4_clusters:
            if Initial_deg4_config[vertex] == 1:
                Lattice_Before_deg4_Flips = MakeCP1(Lattice_Before_deg4_Flips, vertex)
                CP0 = CP0 + 1
            elif Initial_deg4_config[vertex] == 3:
                Lattice_Before_deg4_Flips = MakeCP2(Lattice_Before_deg4_Flips, vertex)
                CP0 = CP0 + 1
            elif Initial_deg4_config[vertex] == 2:
                Lattice_Before_deg4_Flips = MakeCR(Lattice_Before_deg4_Flips, vertex)
                CR0 = CR0 + 1
            deg4_clusters = np.delete(deg4_clusters, np.argwhere(deg4_clusters == vertex))

    for vertex in deg4_clusters:
        if Lattice_Before_deg4_Flips.are_connected(vertex, vertex+1):
            Lattice_Before_deg4_Flips.delete_edges( ((vertex, vertex + 1), (vertex + 2, vertex + 3)) )
        else:
            Lattice_Before_deg4_Flips.delete_edges( ((vertex, vertex + 3), (vertex + 1, vertex + 2)) )

    return Lattice_Before_deg4_Flips, deg4_clusters, CP0, CR0

def Flip_deg4_clusters(Lattice, deg4_clusters, deg4_configs, j, CP0, CR0):
    # Configures deg4 vertices according to deg4_configs[j, :]
    # Computes number of corner passes and crossings

    CP = CP0 + list(deg4_configs[j, :]).count(1) + list(deg4_configs[j, :]).count(3)
    CR = CR0 + list(deg4_configs[j, :]).count(2)
    for k in range(np.shape(deg4_configs)[1]):
        vertex_config = int(deg4_configs[j, k])
        vertex = int(deg4_clusters[k])
        others = [1, 2, 3]
        others.remove(vertex_config)
        Lattice.add_edge(vertex, vertex + vertex_config, weight=0)
        Lattice.add_edge(vertex+others[0], vertex + others[1], weight=0)
    return Lattice, CP, CR

def ModularLength(a, b, N_x):
    # Computes length between a and b on a cylinder

    dist = min(np.abs(b-a), N_x - np.abs(b-a))
    return dist

def ComputeSeparationsAndLengths(nodes, N_x):
    # Computes the separations between nodes, and the length of the strings in nodes

    for i in range(np.shape(nodes)[0]):
        if (N_x - np.abs(nodes[i, 1] - nodes[i, 0])) < np.abs(nodes[i, 1] - nodes[i, 0]):
            nod = nodes[i, 1]
            nodes[i, 1] = nodes[i, 0]
            nodes[i, 0] = nod

    SepsAndLens = []
    for i in range(np.shape(nodes)[0]):
        SepsAndLens.append([set([int(ModularLength(nodes[i, 0], nodes[j, 0], N_x)) for j in range(np.shape(nodes)[0])]),
                           ModularLength(nodes[i, 0], nodes[i, 1], N_x)])

    return SepsAndLens

def ComputeInteraction(nodes0, N_x):
    # Computes the sigma dot sigma interaction of nodes in 'nodes0'

    nodes0 += 1
    Interaction = np.identity(2**N_x)
    for [i, j] in nodes0:
        Interaction = np.dot(Interaction, ComputeSigmaDotSigma(i, j, N_x))
    return Interaction

def ComputeSigmaDotSigma(i, j, N_x):
    # Computes Sigma_i dot Sigma_j

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    X_i = np.kron(np.kron(np.identity(2**(i-1)), X), np.identity(2**(N_x-i)))
    Y_i = np.kron(np.kron(np.identity(2**(i-1)), Y), np.identity(2**(N_x-i)))
    Z_i = np.kron(np.kron(np.identity(2**(i-1)), Z), np.identity(2**(N_x-i)))
    X_j = np.kron(np.kron(np.identity(2**(j-1)), X), np.identity(2**(N_x-j)))
    Y_j = np.kron(np.kron(np.identity(2**(j-1)), Y), np.identity(2**(N_x-j)))
    Z_j = np.kron(np.kron(np.identity(2**(j-1)), Z), np.identity(2**(N_x-j)))

    SigmaDotSigma = np.dot(X_i, X_j)+np.dot(Y_i, Y_j)+np.dot(Z_i, Z_j)
    return SigmaDotSigma

def FNorm(mat):
    # Computes Frobenius norm of mat

    FNorm = np.sqrt(np.abs(np.trace(np.dot(mat, np.transpose(np.conjugate(mat))))))
    return FNorm

def Compute_A_rs(H, N_x):
    # Computes amplitudes A_r of Hamiltonian H

    Z = np.array([[1, 0], [0, -1]])
    A_rs = np.zeros(N_x)

    for r in range(0, N_x):
        for k in range(1, N_x+1):
            Z_k = np.kron(np.kron(np.identity(2 ** (k - 1)), Z), np.identity(2 ** (N_x - k)))
            Z_kPlusr = np.kron(np.kron(np.identity(2**(np.mod(k+r-1, N_x)+1-1)), Z),
                               np.identity(2**(N_x-np.mod(k+r-1, N_x)-1)))
            # A_rs[r] += np.real(4/(N_x*2**N_x)*np.trace(np.dot(np.dot(H, Z_k), Z_kPlusr)))
            A_rs[r] += (4 / (N_x * 2 ** N_x) * np.trace(np.dot(np.dot(H, Z_k), Z_kPlusr)))

    return A_rs

def Compute_d_ns(H, N_x):
    # Computes interaction distance d_n of Hamiltonian H

    Z = np.array([[1, 0], [0, -1]])
    d_ns = np.zeros(N_x+1)
    # d_ns[0] = np.real(np.trace(H)**2/2**N_x)
    d_ns[0] = (np.trace(H) ** 2 / 2 ** N_x)
    for n in range(1, N_x+1):
        combs = np.reshape(list(combinations(range(1, N_x + 1), n)), (-1, n))
        for comb in combs:
            op = np.identity(2**N_x)
            for k in comb:
                op = op*np.kron(np.kron(np.identity(2 ** (k - 1)), Z), np.identity(2 ** (N_x - k)))
            # d_ns[n] += np.real(np.trace(np.dot(H, op))**2/(2**N_x))
            d_ns[n] += (np.trace(np.dot(H, op)) ** 2 / (2 ** N_x))

    return d_ns




if __name__ == '__main__':
    # Parameter specification
    t = time.time()
    N_x = 6                                         # Number of squares in x direction; assumed to be even
    N_y = 2                                         # Number of squares in y direction
    N_faces = N_x*N_y                               # Total number of squares being considered
    h = 1                                           # Height of squares
    w = 1                                           # Width of squares

    deg2_weights = np.array([1.2])                  # Weights of degree 2 vertex
    CR_weight = 1/15                                # Weight of crossing
    CP_weight = 1/15                                # Weight of a corner pass
    loop_weight = 2.5                                 # Weight of a closed loop
    gamma = (loop_weight+1)*CP_weight + CR_weight   # Total contribution from a degree 4 vertex

    epsilon = 0.02                                  # Maximum admissible error in coefficients
    epsilon2 = 0.01                                 # Maximum admissible error in rho
    samples = 80                                    # Maximum number of samples (loop configurations) evaluated
    iterations = 32                                 # Number of iterations over which coefficients are averaged
    range_samples = 50                              # Number of samples used to determine n_range
    deg4_samples = 20                               # Maximum number of deg4 configs sampled over

    Square_Lattice, Square_Lattice_Bare = ConstructSquareLattices(N_x, N_y, w, h)
    Square_Lattice_qb, Square_Lattice_qb_all_kinked, \
        Square_Lattice_qb_all, Square_Lattice_qb_Bare = ConstructSquare_qb_Lattices(N_x, N_y, w, h)

    A_rs = np.zeros([N_x, len(deg2_weights)])
    d_ns = np.zeros([N_x+1, len(deg2_weights)])
    A_rs2 = np.zeros([N_x, len(deg2_weights)])
    d_ns2 = np.zeros([N_x + 1, len(deg2_weights)])
    TwoPointFunctions = np.zeros([N_x-1, len(deg2_weights)])

    for deg2_index in range(len(deg2_weights)):
        deg2_weight = deg2_weights[deg2_index]  # Weight of a degree 2 vertex
        print('\n \n \n deg2_weight = ' + str(deg2_weight) + '\n')

        rho = np.zeros([2**N_x, 2**N_x])


        print('k=0')
        print('Determining n_range')
        As_component = np.zeros(N_faces + 1)
        rung = np.random.randint(1, N_y+1 + 1)

        Lattice_Initial = Square_Lattice_Bare.copy()
        As_component[0] = ComputeAs_component_0_Square(N_x, Lattice_Initial, deg2_weight, loop_weight, rung)

        max = As_component[0]
        max_n_index = 0
        # Highest and lowest number of 'on' faces to be considered
        high = N_faces
        low = 0
        for n in range(1, N_faces+1):
            if np.mod(n, 1) == 0:
                print('n: ' + str(n))

            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces+1)-gammaln(n+1)-gammaln(N_faces-n+1)) > \
                    np.log(range_samples):  # equivalent to nchoosek(N_faces, n) > range_samples
                avg = 1
                combs = ComputeRandomUniqueCombinations(N_faces, n, range_samples)
            else:
                avg = 0
                combs = np.reshape(list(combinations(range(1, N_faces+1), n)), (-1, n))

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
                As_component[n] = np.exp(np.mean(np.log(As_component_contributions))) * \
                    np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1))
                # equivalent to log_avg(As(n,iter))*nchoosek(N_faces,n)
            else:
                As_component[n] = np.sum(As_component_contributions[:])

            if np.abs(As_component[n]) > max:
                max = abs(As_component[n])
                max_n_index = n
            elif np.abs(As_component[n])*(N_faces-n)/max < epsilon:
                high = n
                break
        for n in range(max_n_index, 0, -1):
            if np.abs(As_component[n])*(n+1)/max < epsilon:
                low = n
                break
        approx_error = (np.abs(As_component[high]*(N_faces-high)) + np.abs(As_component[low-1])*(low)) / \
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

        # Compute A
        print('\n Calculating k=0 coefficient')
        Lattice_Initial = Square_Lattice_Bare.copy()

        def ComputeAs_component_Square_parallel(iteration):
            return ComputeAs_component_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma, loop_weight,
                                              n_low, n_high, samples, iteration)

        pool = mp.Pool()
        # A coefficient for each number of faces and iterations
        As = np.transpose(pool.map(ComputeAs_component_Square_parallel, range(iterations)))
        pool.close()
        pool.join()

        # Averages As over iterations and constructs A from this average
        As_avg = np.zeros([n_high - n_low + 1, 1])
        for n in range(0, n_high - n_low + 1):
            As_avg[n, 0] = np.mean(As[n, :])

        # Computes A
        A = np.real(np.sum(As_avg))
        rho = rho + A*np.identity(2**N_x)/(2**N_x)

        print('\n A_avg:')
        print(As_avg)
        print('\n A: ' + str(A) + '\n')




        # Computes n_range for B
        print('k=1')
        print('Determining n_range')
        StringConfigs = list(ComputeStringConfigs(N_x, 1))
        # nodes = np.array(StringConfigs[np.random.randint(0, len(StringConfigs))]).reshape(1, 2)
        nodes = np.array(StringConfigs[0]).reshape(1, 2)
        print('nodes: ')
        print(nodes)
        nodes, Lattice_Initial = InitializeSquareLattice(nodes, Square_Lattice_Bare, Square_Lattice, N_x, N_y)


        Bs_component = np.zeros(N_faces + 1)
        rung = np.random.randint(1, N_y + 1 + 1)
        strings = 1
        parity = np.mod(np.sum(Lattice_Initial.es['weight']), 2)

        Bs_component[0] = ComputeBs_component_0_Square(N_x, Lattice_Initial, deg2_weight, gamma,
                                                       loop_weight, parity, strings, rung)

        max = Bs_component[0]
        max_n_index = 0
        # Highest and lowest number of 'on' faces to be considered
        high = N_faces
        low = 0
        for n in range(1, N_faces + 1):
            if np.mod(n, 1) == 0:
                print('n: ' + str(n))

            # Constructs list of combinations (loop configurations) to analyze
            if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                    np.log(range_samples):  # equivalent to nchoosek(N_faces, n) > range_samples
                avg = 1
                combs = ComputeRandomUniqueCombinations(N_faces, n, range_samples)
            else:
                avg = 0
                combs = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))


            # Computes exp(-energy) for each loop config to be analyzed
            def ComputeBs_component_contribution_Square_parallel(sample):
                return ComputeBs_component_contribution_Square(N_x, N_y, Lattice_Initial, deg2_weight, gamma,
                                                               loop_weight, parity, nodes, strings, combs, rung, sample)


            pool = mp.Pool()
            Bs_component_contributions = \
                np.transpose(pool.map(ComputeBs_component_contribution_Square_parallel, range(np.shape(combs)[0])))
            pool.close()
            pool.join()

            if avg == 1:
                Bs_component[n] = (-1)**parity*np.exp(np.mean(np.log((-1)**parity*Bs_component_contributions))) * \
                    np.exp(gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1))
            else:
                Bs_component[n] = np.sum(Bs_component_contributions[:])

            if np.abs(Bs_component[n]) > max:
                max = abs(Bs_component[n])
                max_n_index = n
            elif np.abs(Bs_component[n])*(N_faces - n)/max < epsilon:
                high = n
                break
        for n in range(max_n_index, 0, -1):
            if np.abs(Bs_component[n])*(n + 1) / max < epsilon:
                low = n
                break
        approx_error = (np.abs(Bs_component[high] * (N_faces - high)) + np.abs(Bs_component[low - 1]) * (low)) / \
                        np.abs(np.sum(Bs_component[low:high + 1]))
        if high == N_faces:
            approx_error = 0
        Bs_determine_n = Bs_component[np.nonzero(Bs_component)]
        n_low = low
        n_high = high

        print('\n Bs for determining n_range: ')
        print(Bs_determine_n)
        print('\n n_high: ' + str(n_high))
        print('n_low: ' + str(n_low))
        print('error: <= ' + str(approx_error))


        print('\n Calculating k=1 coefficients')
        CoefficientDictionary = {}
        for i in range(len(StringConfigs)):
            print('i: ' + str(i))
            nodes = np.array(StringConfigs[i]).reshape(1, 2)
            SepsAndLens = ComputeSeparationsAndLengths(nodes, N_x)
            if SepsAndLens in CoefficientDictionary.values():
                B = [x for x in CoefficientDictionary if CoefficientDictionary[x] == SepsAndLens][0]
                rho = rho + B/(2**N_x)*ComputeInteraction(nodes, N_x)

                if i < N_x-1:
                    TwoPointFunctions[i, deg2_index] = 3*B/A
            else:
                nodes0 = nodes
                nodes, Lattice_Initial = InitializeSquareLattice(nodes, Square_Lattice_Bare, Square_Lattice, N_x, N_y)
                strings = 1
                parity = np.mod(np.sum(Lattice_Initial.es['weight']), 2)

                def ComputeBs_component_Square_parallel(iteration):
                    return ComputeBs_component_Square(N_x, N_y, N_faces, Lattice_Initial, deg2_weight, gamma,
                           loop_weight, parity, nodes, strings, n_low, n_high, samples, iteration)

                pool = mp.Pool()
                # B coefficient for each number of faces and iterations
                Bs = np.transpose(pool.map(ComputeBs_component_Square_parallel, range(iterations)))
                pool.close()
                pool.join()

                # Averages Bs over iterations and constructs B from this average
                Bs_avg = np.zeros([n_high - n_low + 1, 1])
                for n in range(0, n_high - n_low + 1):
                    Bs_avg[n, 0] = np.mean(Bs[n, :])

                # Computes B
                B = np.real(np.sum(Bs_avg))
                rho = rho + B/(2**N_x)*ComputeInteraction(nodes0, N_x)
                CoefficientDictionary[B] = SepsAndLens

                if i < N_x-1:
                    TwoPointFunctions[i, deg2_index] = 3*B/A

                print('\n B_avg:')
                print(Bs_avg)
                print('\n B: ' + str(B) + '\n')
        print('\n Two Point Functions:')
        print(TwoPointFunctions[:, deg2_index])

        for k in range(2, int(N_x/2)+1):

            # Computes n_range for C
            print('k='+str(k))
            print('Determining n_range')
            StringConfigs = list(ComputeStringConfigs(N_x, k))
            # nodes = np.array(StringConfigs[np.random.randint(0, len(StringConfigs))]).reshape(k, 2)
            nodes = np.array(StringConfigs[0]).reshape(k, 2)
            print('nodes: ')
            print(nodes)
            nodes0 = nodes
            nodes, Lattice_Initial, Initial_deg4_config = \
                InitializeSquareLattice_qb(nodes, Square_Lattice_qb_Bare, Square_Lattice_qb_all, N_x, N_y)

            Cs_component = np.zeros(N_faces + 1)
            rung = np.random.randint(1, N_y + 1 + 1)
            strings = k
            parity = np.mod(np.sum(Lattice_Initial.es['weight']), 2)


            Cs_component[0] = ComputeCs_component_0_Square(N_x, N_y, Lattice_Initial, Initial_deg4_config,
                              deg4_samples, deg2_weight, CP_weight, CR_weight, loop_weight, parity, rung)

            max = Cs_component[0]
            max_n_index = 0
            # Highest and lowest number of 'on' faces to be considered
            high = N_faces
            low = 0
            for n in range(1, N_faces + 1):
                if np.mod(n, 1) == 0:
                    print('n: ' + str(n))

                # Constructs list of combinations (loop configurations) to analyze
                if (gammaln(N_faces + 1) - gammaln(n + 1) - gammaln(N_faces - n + 1)) > \
                        np.log(range_samples):  # equivalent to nchoosek(N_faces, n) > range_samples
                    avg = 1
                    combs = ComputeRandomUniqueCombinations(N_faces, n, range_samples)
                else:
                    avg = 0
                    combs = np.reshape(list(combinations(range(1, N_faces + 1), n)), (-1, n))


                # Computes exp(-energy) for each loop config to be analyzed
                def ComputeCs_component_contribution_Square_parallel(sample):
                    return ComputeCs_component_contribution_Square(N_x, N_y, Lattice_Initial, Initial_deg4_config,
                        deg4_samples, deg2_weight, CP_weight, CR_weight, loop_weight, parity, combs, rung, sample)

                pool = mp.Pool()
                Cs_component_contributions = \
                    np.transpose(pool.map(ComputeCs_component_contribution_Square_parallel, range(np.shape(combs)[0])))
                pool.close()
                pool.join()

                if avg == 1 and len(Cs_component_contributions[np.nonzero(Cs_component_contributions)]) > 0:
                    Cs_component[n] = (-1)**parity*np.exp(np.mean(np.log(
                        (-1)**parity*Cs_component_contributions[np.nonzero(Cs_component_contributions)])))*\
                        np.exp(gammaln(N_faces+1) - gammaln(n+1) - gammaln(N_faces-n+1))
                else:
                    Cs_component[n] = np.sum(Cs_component_contributions[:])

                if np.abs(Cs_component[n]) > max:
                    max = abs(Cs_component[n])
                    max_n_index = n
                elif np.abs(Cs_component[n]) * (N_faces - n) / max < epsilon:
                    high = n
                    break
            for n in range(max_n_index, 0, -1):
                if np.abs(Cs_component[n]) * (n + 1) / max < epsilon:
                    low = n
                    break
            approx_error = (np.abs(Cs_component[high]*(N_faces - high)) + np.abs(Cs_component[low - 1])*(low)) / \
                           np.abs(np.sum(Cs_component[low:high + 1]))
            if high == N_faces:
                approx_error = 0
            Cs_determine_n = Cs_component[np.nonzero(Cs_component)]
            n_low = low
            n_high = high


            print('\n Cs for determining n_range: ')
            print(Cs_determine_n)
            print('\n n_high: ' + str(n_high))
            print('n_low: ' + str(n_low))
            print('error: <= ' + str(approx_error))


            print('\n Calculating k=' + str(k) + ' coefficients')
            CoefficientDictionary = {}
            for i in range(len(StringConfigs)):
                print('Configuration: ' + str(i+1) + '/'+str(len(StringConfigs)))
                nodes = np.array(StringConfigs[i]).reshape(k, 2)
                SepsAndLens = ComputeSeparationsAndLengths(nodes, N_x)
                if SepsAndLens in CoefficientDictionary.values():
                    C = [x for x in CoefficientDictionary if CoefficientDictionary[x] == SepsAndLens][0]
                    rho = rho + C/(2**N_x)*ComputeInteraction(nodes, N_x)
                else:
                    nodes0 = nodes
                    nodes, Lattice_Initial, Initial_deg4_config = \
                        InitializeSquareLattice_qb(nodes, Square_Lattice_qb_Bare, Square_Lattice_qb_all, N_x, N_y)
                    strings = k
                    parity = np.mod(np.sum(Lattice_Initial.es['weight']), 2)


                    def ComputeCs_component_Square_parallel(iteration):
                        return ComputeCs_component_Square(N_x, N_y, N_faces, Lattice_Initial, Initial_deg4_config,
                            deg4_samples, deg2_weight, CP_weight, CR_weight, loop_weight, parity, nodes, strings,
                            n_low, n_high, samples, iteration)


                    pool = mp.Pool()
                    # B coefficient for each number of faces and iterations
                    Cs = np.transpose(pool.map(ComputeCs_component_Square_parallel, range(iterations)))
                    pool.close()
                    pool.join()

                    # Averages Cs over iterations and constructs C from this average
                    Cs_avg = np.zeros([n_high - n_low + 1, 1])
                    for n in range(0, n_high - n_low + 1):
                        Cs_avg[n, 0] = np.mean(Cs[n, :])

                    # Computes C
                    C = np.real(np.sum(Cs_avg))
                    rho = rho + C/(2**N_x)*ComputeInteraction(nodes0, N_x)
                    CoefficientDictionary[C] = SepsAndLens

                    print('\n C_avg:')
                    print(Cs_avg)
                    print('\n C: ' + str(C) + '\n')


        rho = rho/np.trace(rho)
        rho = (rho + np.transpose(np.conjugate(rho))) / 2
        # min_rho_value = np.min(np.abs(rho[np.nonzero(rho)]))
        # min_rho_value = abs(np.mean(list(rho[np.nonzero(rho)])))
        # H = -1*(LA.logm(rho/min_rho_value) + np.identity(2**N_x)*np.log(min_rho_value))
        D, U = LA.eig(rho)
        H = -1/2*U.dot(LA.logm(np.diag(D))+LA.logm(np.diag(D)).conj().T).dot(LA.inv(U))
        # H = -1 * (LA.logm(rho))
        # H = (H + np.transpose(np.conjugate(H)))/2
        A_rs[:, deg2_index] = Compute_A_rs(H, N_x)
        d_ns[:, deg2_index] = Compute_d_ns(H, N_x)
        A_rs2[:, deg2_index] = Compute_A_rs(rho, N_x)
        d_ns2[:, deg2_index] = Compute_d_ns(rho, N_x)

        print('\n \n Amplitudes:')
        for amp in A_rs[:, deg2_index]:
            print(amp)

        print('\n \n Amplitudes2:')
        for amp in A_rs2[:, deg2_index]:
            print(amp)

        print('\n \n Distances:')
        for dist in d_ns[:, deg2_index]:
            print(dist)

        print('\n \n Distances:')
        for dist in d_ns2[:, deg2_index]:
            print(dist)

    t2 = time.time()
    print('\n \n \n \n ALL DONE!!!')
    print('Runtime: ' + str(t2-t))
    for deg2_index in range(len(deg2_weights)):
        print('\n \n deg2_weight: ' + str(deg2_weights[deg2_index]))

        print('\n Two Point Functions:')
        for func in TwoPointFunctions[:, deg2_index]:
            print(func)

        print('\n Amplitudes:')
        for amp in A_rs[:, deg2_index]:
            print(amp)

        print('\n Amplitudes2:')
        for amp in A_rs2[:, deg2_index]:
            print(amp)

        print('\n Interaction Distances (all):')
        for dist in d_ns[:, deg2_index]:
            print(dist)

        print('\n Interaction Distances (even terms):')
        for dist in d_ns[0::2, deg2_index]:
            print(dist)

        print('\n Interaction Distances2 (all):')
        for dist in d_ns2[:, deg2_index]:
            print(dist)

        print('\n Interaction Distances2 (even terms):')
        for dist in d_ns2[0::2, deg2_index]:
            print(dist)

    with open(os.path.basename(__file__)+".txt", "w") as text_file:
        print('N_x = ' + str(N_x), file=text_file)
        print('N_y = ' + str(N_y), file=text_file)
        print('deg2_weights = ', file=text_file)
        print(deg2_weights, file=text_file)
        print('CR_weight = ' + str(CR_weight), file=text_file)
        print('CP_weight = ' + str(CP_weight), file=text_file)
        print('loop_weight = ' + str(loop_weight), file=text_file)

        print('epsilon = ' + str(epsilon), file=text_file)
        print('epsilon2 = ' + str(epsilon2), file=text_file)
        print('samples = ' + str(samples), file=text_file)
        print('iterations = ' + str(iterations), file=text_file)
        print('range_samples = ' + str(range_samples), file=text_file)
        print('deg4_samples = ' + str(deg4_samples), file=text_file)
        print('Runtime: ' + str(t2 - t), file=text_file)

        for deg2_index in range(len(deg2_weights)):
            print('\n \n \n deg2_weight: ' + str(deg2_weights[deg2_index]), file=text_file)

            print('\n Two Point Functions:', file=text_file)
            for func in TwoPointFunctions[:, deg2_index]:
                print(func, file=text_file)

            print('\n Amplitudes:', file=text_file)
            for amp in A_rs[:, deg2_index]:
                print(amp, file=text_file)

            print('\n Amplitudes2:', file=text_file)
            for amp in A_rs2[:, deg2_index]:
                print(amp, file=text_file)

            print('\n Interaction Distances (all):', file=text_file)
            for dist in d_ns[:, deg2_index]:
                print(dist, file=text_file)

            print('\n Interaction Distances (even terms):', file=text_file)
            for dist in d_ns[0::2, deg2_index]:
                print(dist, file=text_file)

            print('\n Interaction Distances2 (all):', file=text_file)
            for dist in d_ns2[:, deg2_index]:
                print(dist, file=text_file)

            print('\n Interaction Distances2 (even terms):', file=text_file)
            for dist in d_ns2[0::2, deg2_index]:
                print(dist, file=text_file)

        np.savetxt(str(os.path.basename(__file__) + "_rho.txt"), rho, delimiter='\t')
