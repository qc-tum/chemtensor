/// \file chain_ops.h
/// \brief Higher-level tensor network operations on a chain topology.

#pragma once

#include "mps.h"
#include "mpo.h"


void create_dummy_operator_block_right(const struct block_sparse_tensor* a, const struct block_sparse_tensor* b,
	const struct block_sparse_tensor* w, struct block_sparse_tensor* r);

void create_dummy_operator_block_left(const struct block_sparse_tensor* a, const struct block_sparse_tensor* b,
	const struct block_sparse_tensor* w, struct block_sparse_tensor* l);

void contraction_operator_step_right(const struct block_sparse_tensor* a, const struct block_sparse_tensor* b,
	const struct block_sparse_tensor* w, const struct block_sparse_tensor* r, struct block_sparse_tensor* r_next);

void contraction_operator_step_left(const struct block_sparse_tensor* a, const struct block_sparse_tensor* b,
	const struct block_sparse_tensor* w, const struct block_sparse_tensor* l, struct block_sparse_tensor* l_next);


void compute_right_operator_blocks(const struct mps* psi, const struct mps* chi, const struct mpo* op, struct block_sparse_tensor* r_list);


void mpo_inner_product(const struct mps* chi, const struct mpo* op, const struct mps* psi, void* ret);

//________________________________________________________________________________________________________________________
//

void apply_local_hamiltonian(const struct block_sparse_tensor* a, const struct block_sparse_tensor* w,
	const struct block_sparse_tensor* l, const struct block_sparse_tensor* r, struct block_sparse_tensor* b);

void compute_local_hamiltonian_environment(const struct block_sparse_tensor* a, const struct block_sparse_tensor* b,
	const struct block_sparse_tensor* l, const struct block_sparse_tensor* r, struct block_sparse_tensor* dw);

//________________________________________________________________________________________________________________________
//

void apply_mpo(const struct mpo* op, const struct mps* psi, struct mps* op_psi);
