/// \file chain_ops.c
/// \brief Higher-level tensor network operations on a chain topology.

#include <memory.h>
#include <assert.h>
#include "chain_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy right operator block.
///
/// The right operator blocks use the axes ordering convention
/// (virtual bond connected to 'a', virtual bond connected to 'w', virtual bond connected to conjugated 'b', outer virtual bond).
///
///        ╭─────────╮
///        │         │
///     ─>─2         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///     ─<─1    r    3─<─
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///     ─<─0         │
///        │         │
///        ╰─────────╯
///
void create_dummy_operator_block_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict r)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy trailing virtual bond dimensions
	assert(a->dim_logical[2] == 1);
	assert(b->dim_logical[2] == 1);
	assert(w->dim_logical[3] == 1);

	const enum numeric_type dtype = a->dtype;

	const ct_long dim[6] = { 1, 1, 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[6] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,  TENSOR_AXIS_OUT,
	};
	const qnumber* qnums[6] = {
		a->axis_qnums_logical[2], w->axis_qnums_logical[3], b->axis_qnums_logical[2],
		a->axis_qnums_logical[2], w->axis_qnums_logical[3], b->axis_qnums_logical[2],
	};
	struct block_sparse_tensor s;
	allocate_block_sparse_tensor(dtype, 6, dim, axis_dir, qnums, &s);
	assert(s.nelem == 1);
	assert(s.nblocks == 1);
	memcpy(s.data, numeric_one(dtype), sizeof_numeric_type(dtype));
	struct block_sparse_tensor t;
	block_sparse_tensor_flatten_axes(&s, 4, TENSOR_AXIS_IN, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_flatten_axes(&t, 3, TENSOR_AXIS_IN, r);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy reduced right operator block.
///
///        ╭─────────╮
///        │         │
///     ─<─1         │
///        │         │
///        │         │
///        │         │
///        │    r    2─<─
///        │         │
///        │         │
///        │         │
///     ─<─0         │
///        │         │
///        ╰─────────╯
///
static void create_dummy_operator_block_right_reduced(const struct block_sparse_tensor* restrict a,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict r)
{
	assert(a->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy trailing virtual bond dimensions
	assert(a->dim_logical[2] == 1);
	assert(w->dim_logical[3] == 1);

	const enum numeric_type dtype = a->dtype;

	const ct_long dim[4] = { 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[4] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,
	};
	const qnumber* qnums[6] = {
		a->axis_qnums_logical[2], w->axis_qnums_logical[3],
		a->axis_qnums_logical[2], w->axis_qnums_logical[3],
	};
	struct block_sparse_tensor t;
	allocate_block_sparse_tensor(dtype, 4, dim, axis_dir, qnums, &t);
	assert(t.nelem == 1);
	assert(t.nblocks == 1);
	memcpy(t.data, numeric_one(dtype), sizeof_numeric_type(dtype));
	block_sparse_tensor_flatten_axes(&t, 2, TENSOR_AXIS_IN, r);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy left operator block.
///
/// The left operator blocks use the axes ordering convention
/// (outer virtual bond, virtual bond connected to 'a', virtual bond connected to 'w', virtual bond connected to conjugated 'b').
///
///        ╭─────────╮
///        │         │
///        │         3─>─
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///     ─<─0    l    2─<─
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         │
///        │         1─<─
///        │         │
///        ╰─────────╯
///
void create_dummy_operator_block_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict l)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy leading virtual bond dimensions
	assert(a->dim_logical[0] == 1);
	assert(b->dim_logical[0] == 1);
	assert(w->dim_logical[0] == 1);

	const enum numeric_type dtype = a->dtype;

	const ct_long dim[6] = { 1, 1, 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[6] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,  TENSOR_AXIS_OUT,
	};
	const qnumber* qnums[6] = {
		a->axis_qnums_logical[0], w->axis_qnums_logical[0], b->axis_qnums_logical[0],
		a->axis_qnums_logical[0], w->axis_qnums_logical[0], b->axis_qnums_logical[0],
	};
	struct block_sparse_tensor s;
	allocate_block_sparse_tensor(dtype, 6, dim, axis_dir, qnums, &s);
	assert(s.nelem == 1);
	assert(s.nblocks == 1);
	memcpy(s.data, numeric_one(dtype), sizeof_numeric_type(dtype));
	struct block_sparse_tensor t;
	block_sparse_tensor_flatten_axes(&s, 0, TENSOR_AXIS_OUT, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_flatten_axes(&t, 0, TENSOR_AXIS_OUT, l);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy reduced left operator block.
///
///        ╭─────────╮
///        │         │
///        │         1─<─
///        │         │
///        │         │
///        │         │
///     ─<─0    l    │
///        │         │
///        │         │
///        │         │
///        │         2─<─
///        │         │
///        ╰─────────╯
///
/// Note that the reduced left block uses a reversed axis order.
///
static void create_dummy_reduced_operator_block_left(const struct block_sparse_tensor* restrict a,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict l)
{
	assert(a->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy leading virtual bond dimensions
	assert(a->dim_logical[0] == 1);
	assert(w->dim_logical[0] == 1);

	const enum numeric_type dtype = a->dtype;

	const ct_long dim[4] = { 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[4] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,
	};
	const qnumber* qnums[4] = {
		w->axis_qnums_logical[0], a->axis_qnums_logical[0],
		w->axis_qnums_logical[0], a->axis_qnums_logical[0],
	};
	struct block_sparse_tensor t;
	allocate_block_sparse_tensor(dtype, 4, dim, axis_dir, qnums, &t);
	assert(t.nelem == 1);
	assert(t.nblocks == 1);
	memcpy(t.data, numeric_one(dtype), sizeof_numeric_type(dtype));
	block_sparse_tensor_flatten_axes(&t, 0, TENSOR_AXIS_OUT, l);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from right to left, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///        ╭───────╮         ╭─────────╮
///        │       │         │         │
///     ─>─0   b*  2─>─   ─>─2         │
///        │       │         │         │
///        ╰───1───╯         │         │
///            │             │         │
///            ^             │         │
///            │             │         │
///        ╭───1───╮         │         │
///        │       │         │         │
///     ─<─0   w   3─<─   ─<─1    r    3─<─
///        │       │         │         │
///        ╰───2───╯         │         │
///            │             │         │
///            ^             │         │
///            │             │         │
///        ╭───1───╮         │         │
///        │       │         │         │
///     ─<─0   a   2─<─   ─<─0         │
///        │       │         │         │
///        ╰───────╯         ╰─────────╯
///
void contraction_operator_step_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axes(&s, 1, w, TENSOR_AXIS_RANGE_TRAILING, 2, &t);
	delete_block_sparse_tensor(&s);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor br = *b;  // copy internal data pointers
	br.axis_dir = ct_malloc(br.ndim * sizeof(enum tensor_axis_direction));
	memcpy(br.axis_dir, b->axis_dir, br.ndim * sizeof(enum tensor_axis_direction));
	block_sparse_tensor_reverse_axis_directions(&br);
	if (is_complex_numeric_type(br.dtype))
	{
		// TODO: fuse conjugation with dot product
		struct block_sparse_tensor bc;
		copy_block_sparse_tensor(&br, &bc);
		conjugate_block_sparse_tensor(&bc);
		block_sparse_tensor_multiply_axes(&t, 2, &bc, TENSOR_AXIS_RANGE_TRAILING, 2, r_next);
		delete_block_sparse_tensor(&bc);
	}
	else
	{
		block_sparse_tensor_multiply_axes(&t, 2, &br, TENSOR_AXIS_RANGE_TRAILING, 2, r_next);
	}
	ct_free(br.axis_dir);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from right to left, with a matrix product operator sandwiched in between,
/// for a right block without outer (dummy) virtual bond.
///
/// To-be contracted tensor network:
///
///        ╭───────╮         ╭─────────╮
///        │       │         │         │
///     ─>─0   b*  2─>─   ─>─2         │
///        │       │         │         │
///        ╰───1───╯         │         │
///            │             │         │
///            ^             │         │
///            │             │         │
///        ╭───1───╮         │         │
///        │       │         │         │
///     ─<─0   w   3─<─   ─<─1    r    │
///        │       │         │         │
///        ╰───2───╯         │         │
///            │             │         │
///            ^             │         │
///            │             │         │
///        ╭───1───╮         │         │
///        │       │         │         │
///     ─<─0   a   2─<─   ─<─0         │
///        │       │         │         │
///        ╰───────╯         ╰─────────╯
///
static void contraction_operator_step_right_reduced(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);
	assert(r->ndim == 3);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axes(&s, 1, w, TENSOR_AXIS_RANGE_TRAILING, 2, &t);
	delete_block_sparse_tensor(&s);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor br = *b;  // copy internal data pointers
	br.axis_dir = ct_malloc(br.ndim * sizeof(enum tensor_axis_direction));
	memcpy(br.axis_dir, b->axis_dir, br.ndim * sizeof(enum tensor_axis_direction));
	block_sparse_tensor_reverse_axis_directions(&br);
	if (is_complex_numeric_type(br.dtype))
	{
		// TODO: fuse conjugation with dot product
		struct block_sparse_tensor bc;
		copy_block_sparse_tensor(&br, &bc);
		conjugate_block_sparse_tensor(&bc);
		block_sparse_tensor_multiply_axes(&t, 2, &bc, TENSOR_AXIS_RANGE_TRAILING, 2, r_next);
		delete_block_sparse_tensor(&bc);
	}
	else
	{
		block_sparse_tensor_multiply_axes(&t, 2, &br, TENSOR_AXIS_RANGE_TRAILING, 2, r_next);
	}
	ct_free(br.axis_dir);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from left to right, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///        ╭─────────╮         ╭───────╮
///        │         │         │       │
///        │         3─>─   ─>─0   b*  2─>─
///        │         │         │       │
///        │         │         ╰───1───╯
///        │         │             │
///        │         │             ^
///        │         │             │
///        │         │         ╭───1───╮
///        │         │         │       │
///     ─<─0    l    2─<─   ─<─0   w   3─<─
///        │         │         │       │
///        │         │         ╰───2───╯
///        │         │             │
///        │         │             ^
///        │         │             │
///        │         │         ╭───1───╮
///        │         │         │       │
///        │         1─<─   ─<─0   a   2─<─
///        │         │         │       │
///        ╰─────────╯         ╰───────╯
///
void contraction_operator_step_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict l, struct block_sparse_tensor* restrict l_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);
	assert(l->ndim == 4);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor s;
	struct block_sparse_tensor br = *b;  // copy internal data pointers
	br.axis_dir = ct_malloc(br.ndim * sizeof(enum tensor_axis_direction));
	memcpy(br.axis_dir, b->axis_dir, br.ndim * sizeof(enum tensor_axis_direction));
	block_sparse_tensor_reverse_axis_directions(&br);
	if (is_complex_numeric_type(br.dtype))
	{
		// TODO: fuse conjugation with dot product
		struct block_sparse_tensor bc;
		copy_block_sparse_tensor(&br, &bc);
		conjugate_block_sparse_tensor(&bc);
		block_sparse_tensor_dot(l, TENSOR_AXIS_RANGE_TRAILING, &bc, TENSOR_AXIS_RANGE_LEADING, 1, &s);
		delete_block_sparse_tensor(&bc);
	}
	else
	{
		block_sparse_tensor_dot(l, TENSOR_AXIS_RANGE_TRAILING, &br, TENSOR_AXIS_RANGE_LEADING, 1, &s);
	}
	ct_free(br.axis_dir);

	// multiply with 'w' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axes(&s, 2, w, TENSOR_AXIS_RANGE_LEADING, 2, &t);
	delete_block_sparse_tensor(&s);

	// multiply with 'a' tensor
	block_sparse_tensor_multiply_axes(&t, 1, a, TENSOR_AXIS_RANGE_LEADING, 2, l_next);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from left to right for a reduced left block.
///
/// To-be contracted tensor network:
///
///                                ^
///                                │
///        ╭─────────╮         ╭───1───╮
///        │         │         │       │
///        │         1─<─   ─<─0   w   3─<─
///        │         │         │       │
///        │         │         ╰───2───╯
///        │         │             │
///     ─<─0    l    │             ^
///        │         │             │
///        │         │         ╭───1───╮
///        │         │         │       │
///        │         2─<─   ─<─0   a   2─<─
///        │         │         │       │
///        ╰─────────╯         ╰───────╯
///
static void contraction_operator_step_left_reduced(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict w,
	const struct block_sparse_tensor* restrict l, struct block_sparse_tensor* restrict l_next)
{
	// multiply with 'a' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_dot(l, TENSOR_AXIS_RANGE_TRAILING, a, TENSOR_AXIS_RANGE_LEADING, 1, &t);
	assert(t.ndim == 4);

	// multiply with 'w' tensor
	struct block_sparse_tensor wt;
	const int perm[4] = { 0, 2, 1, 3 };
	block_sparse_tensor_transpose(perm, w, &wt);
	block_sparse_tensor_multiply_axes(&t, 1, &wt, TENSOR_AXIS_RANGE_LEADING, 2, l_next);
	delete_block_sparse_tensor(&t);
	delete_block_sparse_tensor(&wt);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute all partial contractions from the right.
/// 'r_list' must point to an array of (uninitialized) tensors of length 'nsites' at input.
///
void compute_right_operator_blocks(const struct mps* restrict psi, const struct mps* restrict chi, const struct mpo* op, struct block_sparse_tensor* r_list)
{
	const int nsites = op->nsites;
	assert(psi->nsites == nsites);
	assert(chi->nsites == nsites);
	assert(nsites >= 1);

	// initialize rightmost tensor
	create_dummy_operator_block_right(&psi->a[nsites - 1], &chi->a[nsites - 1], &op->a[nsites - 1], &r_list[nsites - 1]);

	for (int i = nsites - 1; i > 0; i--)
	{
		contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r_list[i], &r_list[i - 1]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner product between an MPO and two MPS, `<chi | op | psi>`.
///
void mpo_inner_product(const struct mps* chi, const struct mpo* op, const struct mps* psi, void* ret)
{
	const int nsites = op->nsites;
	assert(chi->nsites == nsites);
	assert(psi->nsites == nsites);
	assert(nsites >= 1);
	// for now requiring dummy leading and trailing virtual bond dimensions
	assert(chi->a[         0].dim_logical[0] == 1);
	assert( op->a[         0].dim_logical[0] == 1);
	assert(psi->a[         0].dim_logical[0] == 1);
	assert(chi->a[nsites - 1].dim_logical[2] == 1);
	assert( op->a[nsites - 1].dim_logical[3] == 1);
	assert(psi->a[nsites - 1].dim_logical[2] == 1);

	const enum numeric_type dtype = psi->a[0].dtype;

	// initialize 'r'
	struct block_sparse_tensor r;
	create_dummy_operator_block_right(&psi->a[nsites - 1], &chi->a[nsites - 1], &op->a[nsites - 1], &r);

	for (int i = nsites - 1; i >= 0; i--)
	{
		struct block_sparse_tensor r_next;
		contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r, &r_next);
		delete_block_sparse_tensor(&r);
		r = r_next;  // copy internal data pointers
	}

	// flatten left virtual bonds
	{
		struct block_sparse_tensor t;
		block_sparse_tensor_flatten_axes(&r, 0, TENSOR_AXIS_OUT, &t);
		delete_block_sparse_tensor(&r);
		block_sparse_tensor_flatten_axes(&t, 0, TENSOR_AXIS_OUT, &r);
		delete_block_sparse_tensor(&t);
	}

	// 'r' should now be a 1 x 1 tensor
	assert(r.ndim == 2);
	assert(r.dim_logical[0] == 1 && r.dim_logical[1] == 1);
	assert(r.nelem == 1);
	assert(r.nblocks == 1);
	memcpy(ret, r.data, sizeof_numeric_type(dtype));
	delete_block_sparse_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a local Hamiltonian operator.
///
/// To-be contracted tensor network:
///
///              .....................................
///             '                                     '
///        ╭────:────╮                           ╭────:────╮
///        │    :    │                           │    :    │
///        │    :    3─>─  0               2  ─>─2    :    │
///        │    :    │                           │    :    │
///        │    :    │                           │    :    │
///        │    :    │             1             │    :    │
///        │    :    │             ^             │    :    │
///        │    :    │             │             │    :    │
///        │    '....│.........╭───1───╮.........│....'    │
///        │         │         │       │         │         │
///     ─<─0    l    2─<─   ─<─0   w   3─<─   ─<─1    r    3─<─
///        │         │         │       │         │         │
///        │         │         ╰───2───╯         │         │
///        │         │             │             │         │
///        │         │             ^             │         │
///        │         │             │             │         │
///        │         │         ╭───1───╮         │         │
///        │         │         │       │         │         │
///        │         1─<─   ─<─0   a   2─<─   ─<─0         │
///        │         │         │       │         │         │
///        ╰─────────╯         ╰───────╯         ╰─────────╯
///
/// The dotted outline marks the output tensor.
/// The outer virtual bonds are contracted as well.
///
void apply_local_hamiltonian(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict w,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict b)
{
	assert(a->ndim == 3);
	assert(w->ndim == 4);
	assert(l->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axes(&s, 1, w, TENSOR_AXIS_RANGE_TRAILING, 2, &t);
	delete_block_sparse_tensor(&s);
	assert(t.ndim == 5);

	// make the right outer virtual bond the leading dimension
	const int perm[5] = { 4, 0, 1, 2, 3 };
	block_sparse_tensor_transpose(perm, &t, &s);
	delete_block_sparse_tensor(&t);

	// multiply with 'l' tensor
	block_sparse_tensor_dot(l, TENSOR_AXIS_RANGE_LEADING, &s, TENSOR_AXIS_RANGE_LEADING, 3, b);
	delete_block_sparse_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a local Hamiltonian operator, using a reduced variant of the left and right blocks.
///
/// To-be contracted tensor network:
///
///              .....................................
///             '                                     '
///        ╭────:────╮                           ╭────:────╮
///        │    :    │                           │    :    │
///        │    :    0─>─  0               2  ─>─2    :    │
///        │    :    │                           │    :    │
///        │    :    │                           │    :    │
///        │    :    │             1             │    :    │
///        │    :    │             ^             │    :    │
///        │    :    │             │             │    :    │
///        │    '....│.........╭───1───╮.........│....'    │
///        │         │         │       │         │         │
///        │    l    1─<─   ─<─0   w   3─<─   ─<─1    r    │
///        │         │         │       │         │         │
///        │         │         ╰───2───╯         │         │
///        │         │             │             │         │
///        │         │             ^             │         │
///        │         │             │             │         │
///        │         │         ╭───1───╮         │         │
///        │         │         │       │         │         │
///        │         2─<─   ─<─0   a   2─<─   ─<─0         │
///        │         │         │       │         │         │
///        ╰─────────╯         ╰───────╯         ╰─────────╯
///
/// The dotted outline marks the output tensor.
///
static void apply_local_hamiltonian_reduced(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict w,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict b)
{
	assert(a->ndim == 3);
	assert(w->ndim == 4);
	assert(l->ndim == 3);
	assert(r->ndim == 3);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axes(&s, 1, w, TENSOR_AXIS_RANGE_TRAILING, 2, &t);
	delete_block_sparse_tensor(&s);
	assert(t.ndim == 4);

	// reverse the trailing axes of 'l' to match the ordering of 't'
	const int perm[3] = { 0, 2, 1 };
	block_sparse_tensor_transpose(perm, l, &s);

	// multiply with transposed 'l' tensor
	block_sparse_tensor_dot(&s, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, b);
	delete_block_sparse_tensor(&t);
	delete_block_sparse_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the network environment of a local Hamiltonian operator.
///
/// To-be contracted tensor network:
///
///        ╭─────────╮         ╭───────╮         ╭─────────╮
///        │         │         │       │         │         │
///        │         3─>─   ─>─0   b*  2─>─   ─>─2         │
///        │         │         │       │         │         │
///        │     ....│.........╰───1───╯.........│....     │
///        │    '    │             │             │    '    │
///        │    :    │             ^             │    :    │
///        │    :    │             1             │    :    │
///        │    :    │                           │    :    │
///        │    :    │                           │    :    │
///     ─<─0    l    2─<─  0               3  ─<─1    r    3─<─
///        │    :    │                           │    :    │
///        │    :    │                           │    :    │
///        │    :    │             2             │    :    │
///        │    :    │             ^             │    :    │
///        │    :    │             │             │    :    │
///        │    '....│.........╭───1───╮.........│....'    │
///        │         │         │       │         │         │
///        │         1─<─   ─<─0   a   2─<─   ─<─0         │
///        │         │         │       │         │         │
///        ╰─────────╯         ╰───────╯         ╰─────────╯
///
/// The dotted outline marks the output tensor.
/// The outer virtual bonds are contracted as well.
///
void compute_local_hamiltonian_environment(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict dw)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(l->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with conjugated 'b' tensor
	// re-order last two dimensions
	const int perm0[5] = { 0, 1, 2, 4, 3 };
	struct block_sparse_tensor t;
	block_sparse_tensor_transpose(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	struct block_sparse_tensor br = *b;  // copy internal data pointers
	br.axis_dir = ct_malloc(br.ndim * sizeof(enum tensor_axis_direction));
	memcpy(br.axis_dir, b->axis_dir, br.ndim * sizeof(enum tensor_axis_direction));
	block_sparse_tensor_reverse_axis_directions(&br);
	if (is_complex_numeric_type(br.dtype))
	{
		// TODO: fuse conjugation with dot product
		struct block_sparse_tensor bc;
		copy_block_sparse_tensor(&br, &bc);
		conjugate_block_sparse_tensor(&bc);
		block_sparse_tensor_dot(&bc, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_TRAILING, 1, &s);
		delete_block_sparse_tensor(&bc);
	}
	else
	{
		block_sparse_tensor_dot(&br, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_TRAILING, 1, &s);
	}
	ct_free(br.axis_dir);
	delete_block_sparse_tensor(&t);

	// multiply with 'l' tensor
	// re-order second and third dimension of 'l'
	const int perm1[4] = { 0, 2, 1, 3 };
	struct block_sparse_tensor k;
	block_sparse_tensor_transpose(perm1, l, &k);
	// re-order leading three dimensions of 's' (to make MPS virtual bond dimensions the leading dimensions)
	const int perm2[6] = { 2, 0, 1, 3, 4, 5 };
	block_sparse_tensor_transpose(perm2, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(&k, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&k);
	delete_block_sparse_tensor(&t);

	// trace out outer virtual bonds (assumed to be low-dimensional)
	block_sparse_tensor_cyclic_partial_trace(&s, 1, dw);
	delete_block_sparse_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply an operator represented as MPO to a state in MPS form,
/// based on a direct contraction of physical axes and grouping of virtual bonds.
///
void apply_mpo_direct(const struct mpo* op, const struct mps* psi, struct mps* op_psi)
{
	// quantum numbers on physical sites must match
	assert(psi->d == op->d);
	assert(qnumber_all_equal(psi->d, psi->qsite, op->qsite));
	assert(psi->nsites == op->nsites);
	assert(psi->nsites >= 1);

	allocate_empty_mps(psi->nsites, psi->d, psi->qsite, op_psi);

	for (int i = 0; i < psi->nsites; i++)
	{
		// move physical input axis of op->a[i] to the end
		const int perm_op[4] = { 0, 1, 3, 2 };
		struct block_sparse_tensor r;
		block_sparse_tensor_transpose(perm_op, &op->a[i], &r);

		// move physical axis of psi->a[i] to the beginning
		const int perm_psi[3] = { 1, 0, 2 };
		struct block_sparse_tensor s;
		block_sparse_tensor_transpose(perm_psi, &psi->a[i], &s);

		// contract local tensors
		struct block_sparse_tensor t;
		block_sparse_tensor_dot(&r, TENSOR_AXIS_RANGE_TRAILING, &s, TENSOR_AXIS_RANGE_LEADING, 1, &t);
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&s);

		// reorder axes
		const int perm_ax[5] = { 0, 3, 1, 2, 4 };
		block_sparse_tensor_transpose(perm_ax, &t, &r);
		delete_block_sparse_tensor(&t);

		// flatten left and right virtual bonds
		block_sparse_tensor_flatten_axes(&r, 0, TENSOR_AXIS_OUT, &s);
		delete_block_sparse_tensor(&r);
		block_sparse_tensor_flatten_axes(&s, 2, TENSOR_AXIS_IN, &op_psi->a[i]);
		delete_block_sparse_tensor(&s);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the right isometry 'q' of the RQ decomposition of an MPS-type tensor 'a',
/// interpreted as matrix with the left virtual bond as row and the physical and right virtual bond as column dimension.
///
/// The 'r' matrix of the decomposition is discarded.
///
static int mps_tensor_right_isometry(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict q)
{
	assert(a->ndim == 3);
	assert(a->axis_dir[1] == TENSOR_AXIS_OUT && a->axis_dir[2] == TENSOR_AXIS_IN);

	// combine physical and right virtual bond axis
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(a, 1, TENSOR_AXIS_IN, &a_mat);

	// perform RQ decomposition
	struct block_sparse_tensor r, q_mat;
	int ret = block_sparse_tensor_rq(&a_mat, QR_REDUCED, &r, &q_mat);
	delete_block_sparse_tensor(&a_mat);
	if (ret < 0) {
		return ret;
	}
	delete_block_sparse_tensor(&r);

	// split physical and right virtual bond axis again
	const ct_long dim_logical_right[2] = { a->dim_logical[1], a->dim_logical[2] };
	const enum tensor_axis_direction axis_dir_right[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
	const qnumber* qnums_logical_right[2] = { a->axis_qnums_logical[1], a->axis_qnums_logical[2] };
	block_sparse_tensor_split_axis(&q_mat, 1, dim_logical_right, axis_dir_right, qnums_logical_right, q);
	delete_block_sparse_tensor(&q_mat);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply an operator represented as MPO to a state in MPS form
/// using the Cholesky-based compression (CBC) method described in
/// "Efficient application of tensor network operators to tensor network states".
///
/// Reference:
///     Richard M. Milbradt, Shuo Sun, Christian B. Mendl, Johnnie Gray, Garnet K.-L. Chan
///     Efficient application of tensor network operators to tensor network states
///     arXiv:2601.19650
///
int apply_mpo_cbc(const struct mpo* op, const struct mps* psi, const double tol, const ct_long max_vdim, struct mps* op_psi)
{
	// quantum numbers on physical sites must match
	assert(psi->d == op->d);
	assert(qnumber_all_equal(psi->d, psi->qsite, op->qsite));
	assert(psi->nsites == op->nsites);
	assert(psi->nsites >= 1);

	const int nsites = psi->nsites;

	if (nsites == 1)
	{
		// special case
		apply_mpo_direct(op, psi, op_psi);
		return 0;
	}

	allocate_empty_mps(nsites, psi->d, psi->qsite, op_psi);

	// left blocks, with axis ordering (outer dummy bond, bond of 'op_psi', bond of 'op', bond of 'psi')
	struct block_sparse_tensor* lblocks = ct_malloc(nsites * sizeof(struct block_sparse_tensor));

	// dummy left block
	create_dummy_reduced_operator_block_left(&psi->a[0], &op->a[0], &lblocks[0]);

	// left-to-right sweep
	for (int i = 0; i < nsites; i++)
	{
		// contract left block with MPS and MPO tensors
		struct block_sparse_tensor t;
		{
			struct block_sparse_tensor s;
			contraction_operator_step_left_reduced(&psi->a[i], &op->a[i], &lblocks[i], &s);
			assert(s.ndim == 4);
			// flatten right virtual bonds of 'op * psi'
			block_sparse_tensor_flatten_axes(&s, 2, TENSOR_AXIS_IN, &t);
			delete_block_sparse_tensor(&s);
		}

		if (i < nsites - 1)
		{
			// flatten physical axis into the row dimension
			struct block_sparse_tensor t_mat;
			block_sparse_tensor_flatten_axes(&t, 0, TENSOR_AXIS_OUT, &t_mat);
			delete_block_sparse_tensor(&t);

			// split by singular value decomposition; only the right factor (including the singular values) is retained
			struct block_sparse_tensor u, sv;
			struct trunc_info info;
			int ret = split_block_sparse_matrix_svd(&t_mat, tol, true, max_vdim, false, SVD_DISTR_RIGHT, &u, &sv, &info);
			delete_block_sparse_tensor(&t_mat);
			if (ret < 0) {
				return ret;
			}
			delete_block_sparse_tensor(&u);

			// restore right virtual bonds of 'op' and 'psi' as separate axes
			const ct_long dim_logical_right[2] = { op->a[i].dim_logical[3], psi->a[i].dim_logical[2] };
			const enum tensor_axis_direction axis_dir_right[2] = { TENSOR_AXIS_IN, TENSOR_AXIS_IN };
			const qnumber* qnums_logical_right[2] = { op->a[i].axis_qnums_logical[3], psi->a[i].axis_qnums_logical[2] };
			block_sparse_tensor_split_axis(&sv, 1, dim_logical_right, axis_dir_right, qnums_logical_right, &lblocks[i + 1]);
			delete_block_sparse_tensor(&sv);
			assert(lblocks[i + 1].ndim == 3);
		}
		else
		{
			// rightmost site
			int ret = mps_tensor_right_isometry(&t, &op_psi->a[i]);
			delete_block_sparse_tensor(&t);
			if (ret < 0) {
				return ret;
			}
		}
	}

	// right-to-left sweep
	struct block_sparse_tensor rblock;
	create_dummy_operator_block_right_reduced(&psi->a[nsites - 1], &op->a[nsites - 1], &rblock);
	for (int i = nsites - 2; i >= 0; i--)
	{
		struct block_sparse_tensor rblock_next;
		contraction_operator_step_right_reduced(&psi->a[i + 1], &op_psi->a[i + 1], &op->a[i + 1], &rblock, &rblock_next);
		delete_block_sparse_tensor(&rblock);
		rblock = rblock_next;  // copy internal data pointers

		struct block_sparse_tensor b;
		apply_local_hamiltonian_reduced(&psi->a[i], &op->a[i], &lblocks[i], &rblock, &b);
		assert(b.ndim == 3);

		if (i > 0)
		{
			int ret = mps_tensor_right_isometry(&b, &op_psi->a[i]);
			delete_block_sparse_tensor(&b);
			if (ret < 0) {
				return ret;
			}
		}
		else
		{
			op_psi->a[i] = b;  // copy internal data pointers
		}
	}
	delete_block_sparse_tensor(&rblock);

	for (int i = 0; i < nsites; i++) {
		delete_block_sparse_tensor(&lblocks[i]);
	}
	ct_free(lblocks);

	assert(mps_is_consistent(op_psi));

	return 0;
}
