import numpy as np
import h5py
import pytenet as ptn


def bipartite_graph_maximum_cardinality_matching_data():

    # random number generator
    rng = np.random.default_rng(281)

    # generate a random bipartite graph
    num_u = rng.integers(1, 101)
    num_v = rng.integers(1, 101)
    edges = []
    for u in range(num_u):
        for v in range(num_v):
            if rng.uniform() < 0.05:
                edges.append((u, v))
    graph = ptn.BipartiteGraph(num_u, num_v, edges)

    # run Hopcroft-Karp algorithm
    hopcroft_karp = ptn.HopcroftKarp(graph)
    matching = hopcroft_karp()

    with h5py.File("data/test_bipartite_graph_maximum_cardinality_matching.hdf5", "w") as file:
        file.attrs["num_u"] = num_u
        file.attrs["num_v"] = num_v
        file["edges"] = np.array(edges)
        file["matching"] = np.array(matching)


def bipartite_graph_minimum_vertex_cover_data():

    # random number generator
    rng = np.random.default_rng(736)

    # generate a random bipartite graph
    num_u = rng.integers(1, 101)
    num_v = rng.integers(1, 101)
    edges = []
    for u in range(num_u):
        for v in range(num_v):
            if rng.uniform() < 0.02:
                edges.append((u, v))
    graph = ptn.BipartiteGraph(num_u, num_v, edges)

    # obtain a minimum vertex cover
    u_cover, v_cover = ptn.minimum_vertex_cover(graph)

    with h5py.File("data/test_bipartite_graph_minimum_vertex_cover.hdf5", "w") as file:
        file.attrs["num_u"] = num_u
        file.attrs["num_v"] = num_v
        file["edges"] = np.array(edges)
        file["u_cover"] = u_cover
        file["v_cover"] = v_cover


def main():
    bipartite_graph_maximum_cardinality_matching_data()
    bipartite_graph_minimum_vertex_cover_data()


if __name__ == "__main__":
    main()
