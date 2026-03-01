from typing import Sequence, Mapping
import numpy as np
from opchain import OpChain
from bipartite_graph import BipartiteGraph, minimum_vertex_cover


class OpGraphNode:
    """
    Operator graph node, corresponding to a virtual bond in an MPO.
    """
    def __init__(self, nid: int, eids_in: Sequence[int], eids_out: Sequence[int], qnum: int):
        assert len(eids_in)  == len(set(eids_in)),  \
            f"incoming edge indices must be pairwise different, received {eids_in}"
        assert len(eids_out) == len(set(eids_out)), \
            f"outgoing edge indices must be pairwise different, received {eids_out}"
        self.nid = nid
        self.eids = (list(eids_in), list(eids_out))
        self.qnum = qnum

    def add_edge_id(self, eid: int, direction: int):
        """
        Add an edge identified by ID 'eid' in the specified direction.
        """
        eids = self.eids[direction]
        assert eid not in eids
        eids.append(eid)


class OpGraphEdge:
    """
    Operator graph edge, representing a weighted sum of local operators
    which are indexed by their IDs.
    """
    def __init__(self, eid: int, nids: Sequence[int], opics: Sequence[tuple[int, float]]):
        if len(nids) != 2:
            raise ValueError(f"expecting exactly two node IDs per edge, received {len(nids)}")
        self.eid   = eid
        self.nids  = list(nids)
        self.opics = []
        for i, c in opics:
            # ensure that each index is unique
            for k in range(len(self.opics)):
                if self.opics[k][0] == i:
                    j, d = self.opics.pop(k)
                    assert i == j
                    # re-insert tuple with added coefficients
                    self.opics.append((i, c + d))
                    break
            else:
                # index 'i' not found so far
                self.opics.append((i, c))
        # sort by index
        self.opics = sorted(self.opics)

    def add(self, other):
        """
        Logical addition of the operation represented by another edge.
        """
        assert self.nids == other.nids
        for i, c in other.opics:
            for k in range(len(self.opics)):
                if self.opics[k][0] == i:
                    j, d = self.opics.pop(k)
                    assert i == j
                    # re-insert tuple with added coefficients
                    self.opics.append((i, c + d))
                    break
            else:
                # index 'i' not found so far
                self.opics.append((i, c))
        # sort by index
        self.opics = sorted(self.opics)


class OpGraph:
    """
    Operator graph: internal data structure for generating MPO representations.

    The layout consists of alternating layers of nodes (corresponding to virtual bonds)
    and edges (corresponding to local operators in the MPO tensors).
    """
    def __init__(self, nodes: Sequence[OpGraphNode], edges: Sequence[OpGraphEdge],
                 nid_terminal: Sequence[int]):
        # dictionary of nodes
        self.nodes = {}
        for node in nodes:
            self.add_node(node)
        # terminal node IDs
        if len(nid_terminal) != 2:
            raise ValueError(f"expecting two terminal node IDs, received {len(nid_terminal)}")
        if nid_terminal[0] not in self.nodes or nid_terminal[1] not in self.nodes:
            raise ValueError(f"terminal node IDs {nid_terminal} not found")
        self.nid_terminal = list(nid_terminal)
        # dictionary of edges
        self.edges = {}
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: OpGraphNode):
        """
        Add a node to the graph.
        """
        if node.nid in self.nodes:
            raise ValueError(f"node with ID {node.nid} already exists")
        self.nodes[node.nid] = node

    def remove_node(self, nid: int) -> OpGraphNode:
        """
        Remove a node from the graph, and return the removed node.
        """
        return self.nodes.pop(nid)

    def add_edge(self, edge: OpGraphEdge):
        """
        Add an edge to the graph.
        """
        if edge.eid in self.edges:
            raise ValueError(f"edge with ID {edge.eid} already exists")
        self.edges[edge.eid] = edge

    def add_connect_edge(self, edge: OpGraphEdge):
        """
        Add an edge to the graph, and connect nodes referenced by the edge to it.
        """
        self.add_edge(edge)
        # connect nodes back to edge
        for direction in (0, 1):
            if edge.nids[direction] in self.nodes:
                self.nodes[edge.nids[direction]].add_edge_id(edge.eid, 1 - direction)

    def node_depth(self, nid: int, direction: int) -> int:
        """
        Determine the depth of a node (distance to terminal node in specified direction),
        assuming that a corresponding path exists within the graph.
        """
        depth = 0
        node = self.nodes[nid]
        while node.eids[direction]:
            # follow first connection
            edge = self.edges[node.eids[direction][0]]
            node = self.nodes[edge.nids[direction]]
            depth += 1
        return depth

    @property
    def length(self) -> int:
        """
        Length of the graph (distance between terminal nodes).
        """
        return self.node_depth(self.nid_terminal[0], 1)

    @classmethod
    def from_opchains(cls, chains: Sequence[OpChain], length: int, oid_identity: int):
        """
        Construct an operator graph from a list of operator chains,
        implementing the algorithm in:
            Jiajun Ren, Weitang Li, Tong Jiang, Zhigang Shuai
            A general automatic method for optimal construction of matrix product operators
            using bipartite graph theory
            J. Chem. Phys. 153, 084118 (2020)

        Args:
            chains: list of operator chains
            length: overall length of the operator graph
            oid_identity: operator ID for identity map

        Returns:
            OpGraph: the constructed operator graph
        """
        if not chains:
            raise ValueError("list of operator chains cannot be empty")

        # construct graph with start node and dummy end node
        node_start = OpGraphNode(0, [], [], 0)
        graph = cls([node_start, OpGraphNode(-1, [], [], 0)],
                    [], [0, -1])
        nid_next = 1
        eid_next = 0

        # pad identities and filter out chains with zero coefficients
        chains = [chain.padded(length, oid_identity) for chain in chains if chain.coeff != 0]

        # convert to half-chains and add a dummy identity operator
        vlist_next  = [OpHalfchain(chain.oids + [oid_identity], chain.qnums + [0], node_start.nid)
                       for chain in chains]
        coeffs_next = [chain.coeff for chain in chains]

        # sweep from left to right
        for _ in range(length):

            ulist, vlist, edges, gamma = _site_partition_halfchains(vlist_next, coeffs_next)

            bigraph = BipartiteGraph(len(ulist), len(vlist), edges)
            u_cover, v_cover = minimum_vertex_cover(bigraph)

            vlist_next  = []
            coeffs_next = []

            for i in u_cover:
                u = ulist[i]
                # add a new operator edge
                graph.add_edge(OpGraphEdge(eid_next, [u.nidl, nid_next], [(u.oid, 1.0)]))
                # connect edge to previous node
                node_prev = graph.nodes[u.nidl]
                node_prev.add_edge_id(eid_next, 1)
                assert node_prev.qnum == u.qnum0
                # add a new node
                node = OpGraphNode(nid_next, [eid_next], [], u.qnum1)
                graph.add_node(node)
                nid_next += 1
                eid_next += 1
                # assemble operator half-chains for next iteration
                for j in bigraph.adj_u[i]:
                    vlist_next.append(OpHalfchain(vlist[j].oids, vlist[j].qnums, node.nid))
                    # pass gamma coefficient
                    coeffs_next.append(gamma[(i, j)])
                    # avoid double-counting
                    edges.remove((i, j))

            for j in v_cover:
                # add a new node
                node = OpGraphNode(nid_next, [], [], vlist[j].qnums[0])
                graph.add_node(node)
                nid_next += 1
                # add operator half-chain for next iteration with reference to node
                vlist_next.append(OpHalfchain(vlist[j].oids, vlist[j].qnums, node.nid))
                coeffs_next.append(1.0)
                # create a "complementary operator"
                for i in bigraph.adj_v[j]:
                    if (i, j) not in edges:
                        continue
                    u = ulist[i]
                    graph.add_edge(OpGraphEdge(eid_next, [u.nidl, node.nid],
                                               [(u.oid, gamma[(i, j)])]))
                    assert u.qnum1 == node.qnum
                    # keep track of handled edges
                    edges.remove((i, j))
                    # connect edge to previous node
                    node_prev = graph.nodes[u.nidl]
                    node_prev.add_edge_id(eid_next, 1)
                    assert node_prev.qnum == u.qnum0
                    # connect edge to new node
                    node.add_edge_id(eid_next, 0)
                    eid_next += 1

            assert not edges

        # dummy trailing half-chain
        assert len(vlist_next) == 1
        assert coeffs_next[0] == 1.0

        # make left node the new end node of the graph
        graph.nid_terminal[1] = vlist_next[0].nidl
        graph.remove_node(-1)

        assert graph.is_consistent()
        return graph

    def to_matrix(self, opmap: Mapping, direction: int = 1) -> np.ndarray:
        """
        Represent the logical operation of the operator graph as a matrix.
        """
        return _subgraph_to_matrix(self, self.nid_terminal[1-direction], opmap, direction)

    def is_consistent(self, verbose: bool = False) -> bool:
        """
        Perform an internal consistency check.
        """
        for k, node in self.nodes.items():
            if k != node.nid:
                if verbose:
                    print(f"Consistency check failed: dictionary key {k} "
                          f"does not match node ID {node.nid}.")
                return False
            for direction in (0, 1):
                for eid in node.eids[direction]:
                    # edge with ID 'eid' must exist
                    if eid not in self.edges:
                        if verbose:
                            print(f"Consistency check failed: edge with ID {eid} "
                                  f"referenced by node {k} does not exist.")
                        return False
                    # edge must refer back to node
                    edge = self.edges[eid]
                    if edge.nids[1-direction] != node.nid:
                        if verbose:
                            print(f"Consistency check failed: edge with ID {eid} "
                                  f"does not refer to node {node.nid}.")
                        return False
        for k, edge in self.edges.items():
            if k != edge.eid:
                if verbose:
                    print(f"Consistency check failed: dictionary key {k} "
                          f"does not match edge ID {edge.eid}.")
                return False
            for direction in (0, 1):
                if edge.nids[direction] not in self.nodes:
                    if verbose:
                        print(f"Consistency check failed: node with ID {edge.nids[direction]} "
                              f"referenced by edge {k} does not exist.")
                    return False
                node = self.nodes[edge.nids[direction]]
                if edge.eid not in node.eids[1-direction]:
                    if verbose:
                        print(f"Consistency check failed: node {node.nid} "
                              f"does not refer to edge {edge.eid}.")
                    return False
            if edge.opics != sorted(edge.opics):
                if verbose:
                    print(f"Consistency check failed: list of operator IDs "
                          f"of edge {edge.eid} is not sorted.")
                return False
        for direction in (0, 1):
            if self.nid_terminal[direction] not in self.nodes:
                if verbose:
                    print(f"Consistency check failed: terminal node ID "
                          f"{self.nid_terminal[direction]} not found.")
                return False
            node = self.nodes[self.nid_terminal[direction]]
            if node.eids[direction]:
                if verbose:
                    print(f"Consistency check failed: terminal node in direction {direction} "
                          f"cannot have edges in that direction.")
                return False
        for direction in (0, 1):
            # node levels (distance from start and end node)
            node_level_map = {}
            nid_start = self.nid_terminal[direction]
            nid_queue = [(nid_start, 0)]
            while nid_queue:
                nid, level = nid_queue.pop(0)
                if nid in node_level_map:
                    if level != node_level_map[nid]:
                        if verbose:
                            print(f"Consistency check failed: level of node {nid} is inconsistent.")
                        return False
                else:
                    node_level_map[nid] = level
                node = self.nodes[nid]
                # insert nodes at next level into queue
                for eid in node.eids[1-direction]:
                    nid_queue.append((self.edges[eid].nids[1-direction], level + 1))
        return True


def _subgraph_to_matrix(graph: OpGraph, nid: int, opmap: Mapping, direction: int) -> np.ndarray:
    """
    Contract the (sub-)graph in the specified direction to obtain its matrix representation.
    """
    if nid == graph.nid_terminal[direction]:
        return np.identity(1)
    op_sum = 0
    node = graph.nodes[nid]
    assert node.eids[direction], f"encountered dangling node {nid} in direction {direction}"
    for eid in node.eids[direction]:
        edge = graph.edges[eid]
        op_sub = _subgraph_to_matrix(graph, edge.nids[direction], opmap, direction)
        op_loc = sum(c * opmap[i] for i, c in edge.opics)
        if direction == 0:
            op = np.kron(op_sub, op_loc)
        else:
            op = np.kron(op_loc, op_sub)
        # not using += here to allow for "up-casting" to complex entries
        op_sum = op_sum + op
    return op_sum


class OpHalfchain:
    """
    Operator half-chain, temporary data structure for building
    an operator graph from a list of operator chains.
    """
    def __init__(self, oids: Sequence[int], qnums: Sequence[int], nidl: int):
        """
        Create an operator half-chain.

        Args:
            oids: list of local op_i operator IDs
            qnums: interleaved bond quantum numbers, including a leading and trailing quantum number
            nidl: ID of left-connected node
        """
        if len(oids) + 1 != len(qnums):
            raise ValueError("incompatible lengths of operator and quantum number lists")
        self.oids  = tuple(oids)
        self.qnums = tuple(qnums)
        self.nidl  = nidl

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, OpHalfchain):
            if self.oids == other.oids and self.qnums == other.qnums and self.nidl == other.nidl:
                return True
        return False

    def __hash__(self):
        """
        Generate a hash value of the operator half-chain.
        """
        return hash((self.oids, self.qnums, self.nidl))


class UNode:
    """
    Store the information of a 'U' node, temporary data structure for building
    an operator graph from a list of operator chains.
    """
    def __init__(self, oid: int, qnum0: int, qnum1: int, nidl: int):
        """
        Create a 'U' node.

        Args:
            oid: local operator ID
            qnum0: left quantum number
            qnum1: right quantum number
            nidl: ID of left-connected node
        """
        self.oid   = oid
        self.qnum0 = qnum0
        self.qnum1 = qnum1
        self.nidl  = nidl

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, UNode):
            return (self.oid, self.qnum0, self.qnum1, self.nidl) == \
                   (other.oid, other.qnum0, other.qnum1, other.nidl)
        return False


def _site_partition_halfchains(chains: Sequence[OpHalfchain], coeffs: Sequence[float]):
    """
    Repartition half-chains after splitting off the local operators acting on the leftmost site.
    """
    ulist = []
    vlist = []
    v_set = set() # entries of 'vlist' stored in a set for faster lookup
    edges = []
    gamma = {}
    for chain, coeff in zip(chains, coeffs):
        # U_i node
        u = UNode(chain.oids[0], chain.qnums[0], chain.qnums[1], chain.nidl)
        if u not in ulist:
            ulist.append(u)
            i = len(ulist) - 1
        else:
            i = ulist.index(u)
        # V_j node
        v = OpHalfchain(chain.oids[1:], chain.qnums[1:], -1)
        if v not in v_set:
            v_set.add(v)
            vlist.append(v)
            j = len(vlist) - 1
        else:
            j = vlist.index(v)
        # corresponding edge and gamma coefficient
        edge = (i, j)
        if edge in edges:
            gamma[edge] += coeff
        else:
            edges.append(edge)
            gamma[edge] = coeff
    return ulist, vlist, edges, gamma
