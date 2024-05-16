/// \file mpo_graph.h
/// \brief MPO graph internal data structure for generating MPO representations.

#pragma once

#include "qnumber.h"
#include "op_chain.h"
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Operator graph node, corresponding to a virtual bond in an MPO.
///
struct mpo_graph_node
{
	qnumber qnum;      //!< quantum number
	int* eids[2];      //!< indices of left- and right-connected edges
	int num_edges[2];  //!< number of connected edges
};


void mpo_graph_node_add_edge(const int direction, const int eid, struct mpo_graph_node* node);


//________________________________________________________________________________________________________________________
///
/// \brief Local operator ID and corresponding coefficient.
///
struct local_op_ref
{
	int oid;        //!< operator ID
	double coeff;   //!< coefficient
};


//________________________________________________________________________________________________________________________
///
/// \brief MPO operator graph edge, representing a weighted sum of local operators which are indexed by their IDs.
///
struct mpo_graph_edge
{
	int nids[2];                 //!< indices of left- and right-connected nodes
	struct local_op_ref* opics;  //!< weighted sum of local operators
	int nopics;                  //!< number of local operators in the sum
};


void mpo_graph_edge_local_op(const struct mpo_graph_edge* edge, const struct dense_tensor* opmap, struct dense_tensor* op);


//________________________________________________________________________________________________________________________
///
/// \brief MPO graph internal data structure for generating MPO representations.
///
struct mpo_graph
{
	struct mpo_graph_node** nodes;  //!< list of nodes for each virtual bond
	struct mpo_graph_edge** edges;  //!< list of edges for each site
	int* num_nodes;                 //!< number of nodes for each virtual bond, i.e., virtual bond dimensions
	int* num_edges;                 //!< number of edges for each site
	int nsites;                     //!< number of sites
};

void mpo_graph_from_opchains(const struct op_chain* chains, const int nchains, const int nsites, const int oid_identity, struct mpo_graph* mpo_graph);

void delete_mpo_graph(struct mpo_graph* mpo_graph);

bool mpo_graph_is_consistent(const struct mpo_graph* mpo_graph);


//________________________________________________________________________________________________________________________
//

// conversion to full matrix (intended for testing)

void mpo_graph_to_matrix(const struct mpo_graph* mpo_graph, const struct dense_tensor* opmap, const enum numeric_type dtype, struct dense_tensor* a);
