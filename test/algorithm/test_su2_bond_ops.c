#include <math.h>
#include "su2_bond_ops.h"
#include "bond_ops.h"
#include "aligned_memory.h"
#include "util.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


static void su2_to_block_sparse_tensor(const struct su2_tensor* s, struct block_sparse_tensor* t)
{
	struct dense_tensor s_dns;
	su2_to_dense_tensor(s, &s_dns);

	enum tensor_axis_direction* axis_dir = ct_malloc(s->ndim_logical * sizeof(enum tensor_axis_direction));
	qnumber** qnums = ct_malloc(s->ndim_logical * sizeof(qnumber*));
	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s_dns.dim[i] == su2_tensor_dim_logical_axis(s, i));
		axis_dir[i] = su2_tensor_logical_axis_direction(s, i);
		qnums[i] = ct_calloc(s_dns.dim[i], sizeof(qnumber));
	}
	
	allocate_block_sparse_tensor(s->dtype, s->ndim_logical, s_dns.dim, axis_dir, (const qnumber**)qnums, t);

	assert(t->blocks[0] != NULL);
	memcpy(t->blocks[0]->data, s_dns.data, dense_tensor_num_elements(&s_dns) * sizeof_numeric_type(s_dns.dtype));

	for (int i = 0; i < s->ndim_logical; i++) {
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&s_dns);
}


char* test_split_su2_matrix_svd()
{
	struct rng_state rng_state;
	seed_rng_state(102, &rng_state);

	const ct_long max_vdim = 3701;

	// whether to copy the fusion-splitting tree to the left or right tensor
	for (int copy_tree_left = 0; copy_tree_left < 2; copy_tree_left++)
	{
		// singular value truncation tolerance
		for (int t = 0; t < 2; t++)
		{
			const double tol = (t == 0 ? 0.1 : 5.3);

			// singular value distribution mode
			for (int d = 0; d < 2; d++)
			{
				const enum su2_singular_value_distr svd_distr = (d == 0 ? SU2_SVD_DISTR_LEFT : SU2_SVD_DISTR_RIGHT);

				// whether the singular values are renormalized before applying the truncation threshold
				for (int relative_thresh = 0; relative_thresh < 2; relative_thresh++)
				{
					// construct an SU(2) tensor 'a'
					struct su2_tensor a;
					{
						// construct the fuse and split tree
						//
						//      0
						//      │
						//      │
						//      ╱╲
						//     ╱  ╲
						//    1    2
						//
						struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
						struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
						struct su2_tree_node j0f = { .i_ax = 0, .c = { NULL, NULL } };
						struct su2_tree_node j0s = { .i_ax = 0, .c = { &j1,  &j2  } };

						struct su2_fuse_split_tree tree = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = 3 };

						if (!su2_fuse_split_tree_is_consistent(&tree)) {
							return "internal consistency check for the fuse and split tree failed";
						}

						// outer (logical and auxiliary) 'j' quantum numbers
						qnumber j0list[] = { 1, 3, 5 };
						qnumber j1list[] = { 1, 5 };
						qnumber j2list[] = { 0 };  // auxiliary
						const struct su2_irreducible_list outer_irreps[3] = {
							{ .jlist = j0list, .num = ARRLEN(j0list) },
							{ .jlist = j1list, .num = ARRLEN(j1list) },
							{ .jlist = j2list, .num = ARRLEN(j2list) },
						};

						// degeneracy dimensions, indexed by 'j' quantum numbers
						//                         j:  0  1  2  3  4  5
						const ct_long dim_degen0[] = { 0, 7, 0, 6, 0, 4 };
						const ct_long dim_degen1[] = { 0, 9, 0, 0, 0, 5 };
						const ct_long* dim_degen[] = {
							dim_degen0,
							dim_degen1,
						};

						allocate_su2_tensor(CT_DOUBLE_COMPLEX, 2, 1, &tree, outer_irreps, dim_degen, &a);

						// fill degeneracy tensors with random entries
						dcomplex alpha;
						numeric_from_double(0.1, a.dtype, &alpha);
						su2_tensor_fill_random_normal(&alpha, numeric_zero(a.dtype), &rng_state, &a);

						if (!su2_tensor_is_consistent(&a)) {
							return "internal consistency check for SU(2) tensor failed";
						}
						if (a.charge_sectors.nsec == 0) {
							return "expecting at least one charge sectors in SU(2) tensor";
						}
					}

					// perform splitting
					struct su2_tensor a0, a1;
					struct trunc_info info;
					if (split_su2_matrix_svd(&a, tol, (bool)relative_thresh, max_vdim, svd_distr, (bool)copy_tree_left, &a0, &a1, &info) < 0) {
						return "'split_su2_matrix_svd' failed internally";
					}

					if (!su2_tensor_is_consistent(&a0)) {
						return "internal consistency check for SU(2) tensor failed";
					}
					if (!su2_tensor_is_consistent(&a1)) {
						return "internal consistency check for SU(2) tensor failed";
					}

					if (copy_tree_left)
					{
						if (!su2_fuse_split_tree_equal(&a0.tree, &a.tree)) {
							return "fusion-splitting tree of left tensor from SU(2) symmetric matrix splitting does not agree with the original one";
						}
					}
					else
					{
						if (!su2_fuse_split_tree_equal(&a1.tree, &a.tree)) {
							return "fusion-splitting tree of right tensor from SU(2) symmetric matrix splitting does not agree with the original one";
						}
					}

					if (svd_distr == SU2_SVD_DISTR_RIGHT)
					{
						// 'a0' must be an isometry
						if (!su2_tensor_is_isometry(&a0, 1e-13, false)) {
							return "left tensor from SU(2) symmetric matrix splitting is not an isometry";
						}
					}
					else
					{
						// 'a1' must be an isometry
						if (!su2_tensor_is_isometry(&a1, 1e-13, true)) {
							return "right tensor from SU(2) symmetric matrix splitting is not an isometry";
						}
					}

					// reference calculation
					struct dense_tensor a_trunc_ref;
					struct trunc_info info_ref;
					{
						// convert to a block-sparse tensor (with a single dense block)
						struct block_sparse_tensor a_blk;
						su2_to_block_sparse_tensor(&a, &a_blk);

						// option to truncate based on the number of retained singular values;
						// otherwise, accumulated squares truncated based on a tolerance can fall between a multiplicity interval
						const ct_long max_vdim_eff = su2_tensor_dim_logical_axis(&a0, 1);

						struct block_sparse_tensor a0_blk, a1_blk;
						if (split_block_sparse_matrix_svd(&a_blk, tol < 1 ? 0. : tol, (bool)relative_thresh, tol < 1 ? max_vdim_eff : max_vdim, false, (const enum singular_value_distr)svd_distr, &a0_blk, &a1_blk, &info_ref) < 0) {
							return "splitting a block-sparse matrix (as reference calculation) failed internally";
						}

						struct block_sparse_tensor a_trunc_blk;
						block_sparse_tensor_dot(&a0_blk, TENSOR_AXIS_RANGE_TRAILING, &a1_blk, TENSOR_AXIS_RANGE_LEADING, 1, &a_trunc_blk);

						block_sparse_to_dense_tensor(&a_trunc_blk, &a_trunc_ref);

						delete_block_sparse_tensor(&a_trunc_blk);
						delete_block_sparse_tensor(&a0_blk);
						delete_block_sparse_tensor(&a1_blk);
						delete_block_sparse_tensor(&a_blk);
					}

					// compare norm of retained singular values
					if (fabs(info.norm_sigma - info_ref.norm_sigma) > 1e-13) {
						return "norm of the retained singular values after splitting an SU(2) symmetric matrix does not agree with reference";
					}

					// reassemble matrix after splitting, for comparison with reference
					struct su2_tensor a_trunc;
					if (copy_tree_left)
					{
						su2_tensor_add_auxiliary_axis(&a0, 0, false);
						const int i_ax_0[] = { 1, 2 };
						const int i_ax_1[] = { 0, 2 };
						su2_tensor_contract_simple(&a0, i_ax_0, &a1, i_ax_1, 2, &a_trunc);
					}
					else
					{
						const int i_ax_0[] = { 1 };
						const int i_ax_1[] = { 0 };
						su2_tensor_contract_simple(&a0, i_ax_0, &a1, i_ax_1, 1, &a_trunc);
					}

					// compare
					if (!dense_su2_tensor_allclose(&a_trunc_ref, &a_trunc, 1e-13)) {
						return "merged SU(2) symmetric matrix after truncation does not match reference";
					}

					delete_su2_tensor(&a0);
					delete_su2_tensor(&a1);
					delete_dense_tensor(&a_trunc_ref);
					delete_su2_tensor(&a_trunc);
					delete_su2_tensor(&a);
				}
			}
		}
	}

	return 0;
}
