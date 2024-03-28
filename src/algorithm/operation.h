/// \file operation.h
/// \brief Higher-level tensor network operations.

#pragma once

#include "mps.h"
#include "mpo.h"


void mps_vdot(const struct mps* chi, const struct mps* psi, void* ret);

double mps_norm(const struct mps* psi);

//________________________________________________________________________________________________________________________
//

void create_dummy_operator_block_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict r);

void create_dummy_operator_block_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict l);

void contraction_operator_step_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next);

void contraction_operator_step_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict l, struct block_sparse_tensor* restrict l_next);


void compute_right_operator_blocks(const struct mps* restrict psi, const struct mps* restrict chi, const struct mpo* op, struct block_sparse_tensor* r_list);


void operator_inner_product(const struct mps* chi, const struct mpo* op, const struct mps* psi, void* ret);

//________________________________________________________________________________________________________________________
//

void apply_local_hamiltonian(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict w,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict b);
