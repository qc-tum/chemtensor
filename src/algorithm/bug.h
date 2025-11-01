/// \file bug.h
/// \brief Basis-Update and Galerkin (BUG) integration for tree tensor networks.

#pragma once

#include "block_sparse_tensor.h"
#include "ttns.h"
#include "ttno.h"


void bug_flow_update_basis(const struct ttno* hamiltonian, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds, const struct block_sparse_tensor* restrict env_parent, const struct block_sparse_tensor* restrict s0,
	const void* prefactor, const double dt, struct ttns* state,
	struct block_sparse_tensor* restrict avg_bonds_augmented, struct block_sparse_tensor* restrict augment_maps);

	
void bug_flow_update_connecting_tensor(const struct block_sparse_tensor* op, const struct block_sparse_tensor* restrict c0,
	const struct abstract_graph* topology, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds_augmented, const struct block_sparse_tensor* restrict env_parent,
	const void* prefactor, const double dt,
	struct block_sparse_tensor* restrict c1);


int bug_tree_time_step(const struct ttno* hamiltonian, const int i_root, const void* prefactor, const double dt, const double tol_compress, const ct_long max_vdim, struct ttns* state);
