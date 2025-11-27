/// \file tree_ops.h
/// \brief Higher-level tensor network operations on a tree topology.

#pragma once

#include "ttns.h"
#include "ttno.h"
#include "block_sparse_tensor.h"


void ttno_inner_product(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, void* ret);

void ttno_subtrees_inner_products(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, const int i_root, struct block_sparse_tensor* avg_bonds);

void local_ttno_inner_product(const struct block_sparse_tensor* restrict chi, const struct block_sparse_tensor* restrict op, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const int i_parent, struct block_sparse_tensor* restrict avg_bonds);

void apply_local_ttno_tensor(const struct block_sparse_tensor* restrict op, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const struct block_sparse_tensor* restrict envs,
	struct block_sparse_tensor* restrict ret);
