/// \file ttno_graph.h
/// \brief Tree tensor network operator (TTNO) graph internal data structure.
///
/// Vertices and hyperedges only depend on undirected tree topology and enumeration of physical sites.

#pragma once

#include "qnumber.h"
#include "abstract_graph.h"
#include "local_op.h"
#include "op_chain.h"
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Operator graph vertex, corresponding to a virtual bond in a TTNO.
///
struct ttno_graph_vertex
{
	int* eids[2];      //!< indices of connected hyperedges; direction ordered by site index
	int num_edges[2];  //!< number of connected hyperedges
	qnumber qnum;      //!< quantum number
};


void ttno_graph_vertex_add_edge(const int direction, const int eid, struct ttno_graph_vertex* vertex);


//________________________________________________________________________________________________________________________
///
/// \brief TTNO graph hyperedge, representing a weighted sum of local operators which are indexed by their IDs.
///
struct ttno_graph_hyperedge
{
	int* vids;                   //!< indices of connected vertices, ordered by neighboring hyperedge site indices
	int order;                   //!< order (number of connected vertices)
	struct local_op_ref* opics;  //!< weighted sum of local operators
	int nopics;                  //!< number of local operators in the sum
};


//________________________________________________________________________________________________________________________
///
/// \brief TTNO graph internal data structure for generating TTNO representations.
///
struct ttno_graph
{
	struct abstract_graph topology;       //!< logical tree topology; nodes correspond to physical sites
	struct ttno_graph_hyperedge** edges;  //!< list of hyperedges for each physical site
	struct ttno_graph_vertex** verts;     //!< list of vertices for each virtual bond, indexed by corresponding hyperedge site indices (i, j) with i < j
	int* num_edges;                       //!< number of edges for each site
	int* num_verts;                       //!< number of vertices for each virtual bond, i.e., virtual bond dimensions, indexed by corresponding hyperedge site indices (i, j) with i < j
	int nsites;                           //!< number of sites
};

int ttno_graph_from_opchains(const struct op_chain* chains, const int nchains, const struct abstract_graph* topology, const int oid_identity, struct ttno_graph* ttno_graph);

void delete_ttno_graph(struct ttno_graph* graph);

bool ttno_graph_is_consistent(const struct ttno_graph* graph);


//________________________________________________________________________________________________________________________
//

// conversion to full matrix (intended for testing)

void ttno_graph_to_matrix(const struct ttno_graph* graph, const struct dense_tensor* opmap, struct dense_tensor* mat);
