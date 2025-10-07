#include <math.h>
#include "bug.h"
#include "tree_ops.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


//________________________________________________________________________________________________________________________
///
/// \brief Construct a Hamiltonian as TTNO operator graph based on local operator chains, which are shifted along a 1D lattice.
///
static void local_opchains_to_ttno_graph(const struct op_chain* lopchains, const int nlopchains, const int nsites_physical, const struct abstract_graph* topology, struct ttno_graph* graph)
{
	int nchains = 0;
	for (int j = 0; j < nlopchains; j++)
	{
		assert(lopchains[j].length <= nsites_physical);
		nchains += (nsites_physical - lopchains[j].length + 1);
	}
	struct op_chain* opchains = ct_malloc(nchains * sizeof(struct op_chain));
	int c = 0;
	for (int j = 0; j < nlopchains; j++)
	{
		for (int i = 0; i < nsites_physical - lopchains[j].length + 1; i++)
		{
			// add shifted opchain; shallow copy sufficient here
			memcpy(&opchains[c], &lopchains[j], sizeof(struct op_chain));
			opchains[c].istart = i;
			c++;
		}
	}
	assert(c == nchains);

	ttno_graph_from_opchains(opchains, nchains, nsites_physical, topology, graph);

	ct_free(opchains);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a TTNO assembly representation of the Bose-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
static void construct_bose_hubbard_1d_ttno_assembly_complex(const int nsites_physical, const long d, const double t, const double u, const double mu, const struct abstract_graph* topology, struct ttno_assembly* assembly)
{
	assert(nsites_physical >= 2);
	assert(d >= 1);

	// physical quantum numbers (particle number)
	assembly->d = d;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	for (long i = 0; i < d; i++) {
		assembly->qsite[i] = i;
	}

	assembly->dtype = CT_DOUBLE_COMPLEX;

	// operator map
	const int OID_Id = 0;  // identity
	const int OID_Bd = 1;  // b_{\dagger}
	const int OID_B  = 2;  // b
	const int OID_N  = 3;  // n
	const int OID_NI = 4;  // n (n - 1) / 2
	assembly->num_local_ops = 5;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < assembly->num_local_ops; i++) {
		const long dim[2] = { assembly->d, assembly->d };
		allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
	}
	// identity operator
	dense_tensor_set_identity(&assembly->opmap[OID_Id]);
	// bosonic creation operator
	dcomplex* b_dag = assembly->opmap[OID_Bd].data;
	for (long i = 0; i < d - 1; i++) {
		b_dag[(i + 1)*d + i] = sqrt(i + 1);
	}
	// bosonic annihilation operator
	dcomplex* b_ann = assembly->opmap[OID_B].data;
	for (long i = 0; i < d - 1; i++) {
		b_ann[i*d + (i + 1)] = sqrt(i + 1);
	}
	// bosonic number operator
	dcomplex* numop = assembly->opmap[OID_N].data;
	for (long i = 0; i < d; i++) {
		numop[i*d + i] = i;
	}
	// bosonic local interaction operator n (n - 1) / 2
	dcomplex* v_int = assembly->opmap[OID_NI].data;
	for (long i = 0; i < d; i++) {
		v_int[i*d + i] = i * (i - 1) / 2;
	}

	// coefficient map; first two entries must always be 0 and 1
	const dcomplex coeffmap[] = { 0, 1, -t, -mu, u };
	assembly->num_coeffs = ARRLEN(coeffmap);
	assembly->coeffmap = ct_malloc(sizeof(coeffmap));
	memcpy(assembly->coeffmap, coeffmap, sizeof(coeffmap));

	// local two-site and single-site terms
	int oids_c0[] = { OID_Bd, OID_B  };  qnumber qnums_c0[] = { 0,  1,  0 };
	int oids_c1[] = { OID_B,  OID_Bd };  qnumber qnums_c1[] = { 0, -1,  0 };
	int oids_c2[] = { OID_N  };          qnumber qnums_c2[] = { 0,  0 };
	int oids_c3[] = { OID_NI };          qnumber qnums_c3[] = { 0,  0 };
	struct op_chain lopchains[] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .cid = 2, .length = ARRLEN(oids_c0), .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .cid = 2, .length = ARRLEN(oids_c1), .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .cid = 3, .length = ARRLEN(oids_c2), .istart = 0 },
		{ .oids = oids_c3, .qnums = qnums_c3, .cid = 4, .length = ARRLEN(oids_c3), .istart = 0 },
	};

	// convert to a TTNO graph
	local_opchains_to_ttno_graph(lopchains, ARRLEN(lopchains), nsites_physical, topology, &assembly->graph);
}


static inline int read_quantum_numbers(const hid_t file, const char* varname, long* dim, qnumber** qnums)
{
	hsize_t qdims[1];
	if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
		return -1;
	}
	(*dim) = qdims[0];

	(*qnums) = ct_malloc((*dim) * sizeof(qnumber));
	if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, (*qnums)) < 0) {
		return -2;
	}

	return 0;
}


char* test_bug_flow_update_basis_leaf()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_bug_flow_update_basis_leaf.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bug_flow_update_basis_leaf failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of physical and branching lattice sites
	const int nsites_physical  = 3;
	const int nsites_branching = 0;

	// local physical dimension
	long d;

	qnumber* qsite;
	if (read_quantum_numbers(file, "qsite", &d, &qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	qnumber qnum_sector;
	if (read_hdf5_attribute(file, "qnum_sector", H5T_NATIVE_INT, &qnum_sector) < 0) {
		return "reading quantum number sector from disk failed";
	}

	long dim_bond_state;
	qnumber* qbond_state_base;
	if (read_quantum_numbers(file, "qbond_state", &dim_bond_state, &qbond_state_base) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}

	long dim_bond_op;
	qnumber* qbond_op_base;
	if (read_quantum_numbers(file, "qbond_op", &dim_bond_op, &qbond_op_base) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}

	dcomplex prefactor;
	if (read_hdf5_attribute(file, "prefactor", hdf5_dcomplex_id, &prefactor) < 0) {
		return "reading 'prefactor' from disk failed";
	}

	double dt;
	if (read_hdf5_attribute(file, "dt", H5T_NATIVE_DOUBLE, &dt) < 0) {
		return "reading time step 'dt' from disk failed";
	}

	// topology variants
	for (int m = 0; m < 3; m++)
	{
		// m == 0: current site index < neighboring (parent) site index, and auxiliary axis attached to current site tensor
		// m == 1: current site index < neighboring (parent) site index, without auxiliary axis
		// m == 2: current site index > neighboring (parent) site index, without auxiliary axis

		// tree topology:
		//
		// (m+2)%3
		//    │
		//    │
		// (m+1)%3
		//    │
		//    │
		//    m
		//
		int neigh[3][2] = {
			{ (m + 1) % 3, -1 },
			{  m, (m + 2) % 3 },
			{ (m + 1) % 3, -1 },
		};
		int* neighbor_map[3] = {
			neigh[(3 - m) % 3], neigh[(4 - m) % 3], neigh[(5 - m) % 3],
		};
		int num_neighbors[3] = { 1, 1, 1 };
		num_neighbors[(m + 1) % 3] = 2;
		struct abstract_graph topology = {
			.neighbor_map  = neighbor_map,
			.num_neighbors = num_neighbors,
			.num_nodes     = nsites_physical + nsites_branching,
		};
		assert(abstract_graph_is_connected_tree(&topology));

		qnumber* qbond_state = ct_malloc(dim_bond_state * sizeof(qnumber));
		qnumber* qbond_op    = ct_malloc(dim_bond_op * sizeof(qnumber));
		if (m == 0)
		{
			memcpy(qbond_state, qbond_state_base, dim_bond_state * sizeof(qnumber));
			memcpy(qbond_op,    qbond_op_base,    dim_bond_op    * sizeof(qnumber));
		}
		else if (m == 1)
		{
			// absorb quantum number sector in virtual bond
			for (long i = 0; i < dim_bond_state; i++) {
				qbond_state[i] = qbond_state_base[i] + qnum_sector;
			}
			memcpy(qbond_op, qbond_op_base, dim_bond_op * sizeof(qnumber));
		}
		else
		{
			// absorb quantum number sector in virtual bond, and flip sign due to reversed bond direction
			for (long i = 0; i < dim_bond_state; i++) {
				qbond_state[i] = -(qbond_state_base[i] + qnum_sector);
			}
			for (long i = 0; i < dim_bond_op; i++) {
				qbond_op[i] = -qbond_op_base[i];
			}
		}

		// local tensor of initial quantum state
		struct block_sparse_tensor a_state_0;
		{
			struct dense_tensor a_state_0_dns;

			if (m == 0)
			{
				// read dense tensor from disk
				const long dim[3] = { d, 1, dim_bond_state };
				allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &a_state_0_dns);
				if (read_hdf5_dataset(file, "a_state_0", hdf5_dcomplex_id, a_state_0_dns.data) < 0) {
					return "reading tensor entries from disk failed";
				}

				const qnumber qnum_sector_array[1] = { qnum_sector };
				const qnumber* qnums[3] = { qsite, qnum_sector_array, qbond_state };
				const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };

				// convert dense to block-sparse tensor
				dense_to_block_sparse_tensor(&a_state_0_dns, axis_dir, qnums, &a_state_0);
			}
			else if (m == 1)
			{
				// read dense tensor from disk
				const long dim[2] = { d, dim_bond_state };
				allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &a_state_0_dns);
				if (read_hdf5_dataset(file, "a_state_0", hdf5_dcomplex_id, a_state_0_dns.data) < 0) {
					return "reading tensor entries from disk failed";
				}

				const qnumber* qnums[2] = { qsite, qbond_state };
				const enum tensor_axis_direction axis_dir[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

				// convert dense to block-sparse tensor
				dense_to_block_sparse_tensor(&a_state_0_dns, axis_dir, qnums, &a_state_0);
			}
			else
			{
				// read dense tensor from disk
				struct dense_tensor a_state_0_tp_dns;
				const long dim[2] = { d, dim_bond_state };
				allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &a_state_0_tp_dns);
				if (read_hdf5_dataset(file, "a_state_0", hdf5_dcomplex_id, a_state_0_tp_dns.data) < 0) {
					return "reading tensor entries from disk failed";
				}
				const int perm[2] = { 1, 0 };
				transpose_dense_tensor(perm, &a_state_0_tp_dns, &a_state_0_dns);
				delete_dense_tensor(&a_state_0_tp_dns);

				const qnumber* qnums[2] = { qbond_state, qsite };
				const enum tensor_axis_direction axis_dir[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };

				// convert dense to block-sparse tensor
				dense_to_block_sparse_tensor(&a_state_0_dns, axis_dir, qnums, &a_state_0);
			}

			// verify that sparsity pattern is correctly described by quantum numbers
			if (!dense_block_sparse_tensor_allclose(&a_state_0_dns, &a_state_0, 0)) {
				return "quantum number sparsity pattern mismatch";
			}

			delete_dense_tensor(&a_state_0_dns);
		}
		struct block_sparse_tensor a_state_list[3] = { 0 };
		a_state_list[m] = a_state_0;

		// local tensor of operator
		struct block_sparse_tensor a_op;
		{
			struct dense_tensor a_op_dns;

			if (m <= 1)
			{
				// read dense tensor from disk
				const long dim[3] = { d, d, dim_bond_op };
				allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &a_op_dns);
				if (read_hdf5_dataset(file, "a_op", hdf5_dcomplex_id, a_op_dns.data) < 0) {
					return "reading tensor entries from disk failed";
				}

				const qnumber* qnums[3] = { qsite, qsite, qbond_op };
				const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };

				// convert dense to block-sparse tensor
				dense_to_block_sparse_tensor(&a_op_dns, axis_dir, qnums, &a_op);
			}
			else
			{
				// read dense tensor from disk
				struct dense_tensor a_op_tp_dns;
				const long dim[3] = { d, d, dim_bond_op };
				allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &a_op_tp_dns);
				if (read_hdf5_dataset(file, "a_op", hdf5_dcomplex_id, a_op_tp_dns.data) < 0) {
					return "reading tensor entries from disk failed";
				}

				const int perm[3] = { 2, 0, 1 };
				transpose_dense_tensor(perm, &a_op_tp_dns, &a_op_dns);
				delete_dense_tensor(&a_op_tp_dns);

				const qnumber* qnums[3] = { qbond_op, qsite, qsite };
				const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

				// convert dense to block-sparse tensor
				dense_to_block_sparse_tensor(&a_op_dns, axis_dir, qnums, &a_op);
			}

			// verify that sparsity pattern is correctly described by quantum numbers
			if (!dense_block_sparse_tensor_allclose(&a_op_dns, &a_op, 0)) {
				return "quantum number sparsity pattern mismatch";
			}

			delete_dense_tensor(&a_op_dns);
		}
		struct block_sparse_tensor a_op_list[3] = { 0 };
		a_op_list[m] = a_op;

		struct block_sparse_tensor env_parent;
		{
			// read dense tensor from disk
			struct dense_tensor env_parent_dns;
			const long dim[3] = { dim_bond_state, dim_bond_op, dim_bond_state };
			allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &env_parent_dns);
			if (read_hdf5_dataset(file, "env_parent", hdf5_dcomplex_id, env_parent_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			const qnumber* qnums[3] = { qbond_state, qbond_op, qbond_state };
			const enum tensor_axis_direction axis_dir[3] = {
				m <= 1 ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN,
				m <= 1 ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN,
				m <= 1 ? TENSOR_AXIS_IN  : TENSOR_AXIS_OUT,
			};

			// convert dense to block-sparse tensor
			dense_to_block_sparse_tensor(&env_parent_dns, axis_dir, qnums, &env_parent);

			// verify that sparsity pattern is correctly described by quantum numbers
			if (!dense_block_sparse_tensor_allclose(&env_parent_dns, &env_parent, 0)) {
				return "quantum number sparsity pattern mismatch";
			}

			delete_dense_tensor(&env_parent_dns);
		}

		struct block_sparse_tensor s0;
		{
			// read dense tensor from disk
			struct dense_tensor s0_dns;
			const long dim[2] = { dim_bond_state, dim_bond_state };
			allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &s0_dns);
			if (read_hdf5_dataset(file, "s0", hdf5_dcomplex_id, s0_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			const qnumber* qnums[2] = { qbond_state, qbond_state };
			const enum tensor_axis_direction axis_dir[2] = {
				m <= 1 ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN,
				m <= 1 ? TENSOR_AXIS_IN  : TENSOR_AXIS_OUT,
			};

			// convert dense to block-sparse tensor
			dense_to_block_sparse_tensor(&s0_dns, axis_dir, qnums, &s0);

			// verify that sparsity pattern is correctly described by quantum numbers
			if (!dense_block_sparse_tensor_allclose(&s0_dns, &s0, 0)) {
				return "quantum number sparsity pattern mismatch";
			}

			delete_dense_tensor(&s0_dns);
		}

		// (initial) quantum state, will be updated in-place
		struct ttns y = {
			.a                = a_state_list,
			.topology         = topology,
			.nsites_physical  = nsites_physical,
			.nsites_branching = nsites_branching,
		};

		// fictitious operator
		struct ttno op = {
			.a                = a_op_list,
			.topology         = topology,
			.qsite            = qsite,
			.d                = d,
			.nsites_physical  = nsites_physical,
			.nsites_branching = nsites_branching,
		};

		struct block_sparse_tensor avg_bonds_augmented[3] = { 0 };
		struct block_sparse_tensor augment_maps[3] = { 0 };
		bug_flow_update_basis(&op, m, (m + 1) % 3, NULL, &env_parent, &s0, &prefactor, dt, &y, avg_bonds_augmented, augment_maps);

		// local reference tensor of updated quantum state
		struct dense_tensor a_state_1_ref;
		{
			// read dense tensor from disk
			hsize_t dim_file[2];
			if (get_hdf5_dataset_dims(file, "a_state_1", dim_file) < 0) {
				return "obtaining dimensions of reference tensor failed";
			}
			long dim[3];
			dim[0] = dim_file[0];
			if (m == 0) {
				// include dummy auxiliary axis
				dim[1] = 1;
				dim[2] = dim_file[1];
			}
			else {
				dim[1] = dim_file[1];
			}
			allocate_dense_tensor(CT_DOUBLE_COMPLEX, m == 0 ? 3 : 2, dim, &a_state_1_ref);
			if (read_hdf5_dataset(file, "a_state_1", hdf5_dcomplex_id, a_state_1_ref.data) < 0) {
				return "reading tensor entries from disk failed";
			}
		}

		// contracting physical axes with reference tensor should result in an isometry (not necessarily square due to augmentation by identity blocks)
		{
			struct dense_tensor a_state_dns;
			block_sparse_to_dense_tensor(&y.a[m], &a_state_dns);

			conjugate_dense_tensor(&a_state_1_ref);

			struct dense_tensor u;
			dense_tensor_dot(&a_state_1_ref, TENSOR_AXIS_RANGE_LEADING,
				&a_state_dns, m <= 1 ? TENSOR_AXIS_RANGE_LEADING : TENSOR_AXIS_RANGE_TRAILING, m == 0 ? 2 : 1, &u);
			if (!dense_tensor_is_isometry(&u, 1e-14, true)) {
				return "expecting an isometric overlap matrix";
			}
			delete_dense_tensor(&u);

			delete_dense_tensor(&a_state_dns);
		}

		delete_dense_tensor(&a_state_1_ref);

		delete_block_sparse_tensor(&augment_maps[m]);
		delete_block_sparse_tensor(&avg_bonds_augmented[m]);
		delete_block_sparse_tensor(&s0);
		delete_block_sparse_tensor(&env_parent);
		delete_block_sparse_tensor(&a_op);
		delete_block_sparse_tensor(&y.a[m]);
		ct_free(qbond_op);
		ct_free(qbond_state);
	}

	ct_free(qbond_op_base);
	ct_free(qbond_state_base);
	ct_free(qsite);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_bug_flow_update_connecting_tensor()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_bug_flow_update_connecting_tensor.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bug_flow_update_connecting_tensor failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of physical and branching lattice sites
	const int nsites_physical  = 5;
	const int nsites_branching = 0;

	// tree topology:
	//
	//     4
	//     │
	//     │
	//     2
	//    ╱│╲
	//   ╱ │ ╲
	//  1  3  0
	//
	int neigh0[] = { 2 };
	int neigh1[] = { 2 };
	int neigh2[] = { 0, 1, 3, 4 };
	int neigh3[] = { 2 };
	int neigh4[] = { 2 };
	int* neighbor_map[5] = {
		neigh0, neigh1, neigh2, neigh3, neigh4,
	};
	int num_neighbors[5] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites_physical + nsites_branching,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	// local physical dimension
	long d;

	// physical and virtual bond quantum numbers

	qnumber* qsite;
	if (read_quantum_numbers(file, "qsite", &d, &qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	long dim_bond02_state;
	qnumber* qbond02_state;
	if (read_quantum_numbers(file, "qbond02_state", &dim_bond02_state, &qbond02_state) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond12_state;
	qnumber* qbond12_state;
	if (read_quantum_numbers(file, "qbond12_state", &dim_bond12_state, &qbond12_state) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond23_state;
	qnumber* qbond23_state;
	if (read_quantum_numbers(file, "qbond23_state", &dim_bond23_state, &qbond23_state) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond24_state;
	qnumber* qbond24_state;
	if (read_quantum_numbers(file, "qbond24_state", &dim_bond24_state, &qbond24_state) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}

	long dim_bond02_op;
	qnumber* qbond02_op;
	if (read_quantum_numbers(file, "qbond02_op", &dim_bond02_op, &qbond02_op) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond12_op;
	qnumber* qbond12_op;
	if (read_quantum_numbers(file, "qbond12_op", &dim_bond12_op, &qbond12_op) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond23_op;
	qnumber* qbond23_op;
	if (read_quantum_numbers(file, "qbond23_op", &dim_bond23_op, &qbond23_op) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}
	long dim_bond24_op;
	qnumber* qbond24_op;
	if (read_quantum_numbers(file, "qbond24_op", &dim_bond24_op, &qbond24_op) < 0) {
		return "reading virtual bond quantum numbers from disk failed";
	}

	dcomplex prefactor;
	if (read_hdf5_attribute(file, "prefactor", hdf5_dcomplex_id, &prefactor) < 0) {
		return "reading 'prefactor' from disk failed";
	}

	double dt;
	if (read_hdf5_attribute(file, "dt", H5T_NATIVE_DOUBLE, &dt) < 0) {
		return "reading time step 'dt' from disk failed";
	}

	// local TTNS tensor
	struct block_sparse_tensor c0;
	{
		// read dense tensor from disk
		struct dense_tensor c0_dns;
		const long dim[5] = { dim_bond02_state, dim_bond12_state, d, dim_bond23_state, dim_bond24_state };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, dim, &c0_dns);
		if (read_hdf5_dataset(file, "c0", hdf5_dcomplex_id, c0_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[5] = { qbond02_state, qbond12_state, qsite, qbond23_state, qbond24_state };
		const enum tensor_axis_direction axis_dir[5] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&c0_dns, axis_dir, qnums, &c0);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&c0_dns, &c0, 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&c0_dns);
	}

	// local TTNO tensor
	struct block_sparse_tensor a_op;
	{
		// read dense tensor from disk
		struct dense_tensor a_op_dns;
		const long dim[6] = { dim_bond02_op, dim_bond12_op, d, d, dim_bond23_op, dim_bond24_op };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 6, dim, &a_op_dns);
		if (read_hdf5_dataset(file, "a_op", hdf5_dcomplex_id, a_op_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[6] = { qbond02_op, qbond12_op, qsite, qsite, qbond23_op, qbond24_op };
		const enum tensor_axis_direction axis_dir[6] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN, TENSOR_AXIS_IN };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&a_op_dns, axis_dir, qnums, &a_op);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&a_op_dns, &a_op, 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&a_op_dns);
	}

	struct block_sparse_tensor avg_bonds[5] = { 0 };
	// bond (0, 2)
	{
		// read dense tensor from disk
		struct dense_tensor avg_bond_dns;
		const long dim[3] = { dim_bond02_state, dim_bond02_op, dim_bond02_state };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &avg_bond_dns);
		if (read_hdf5_dataset(file, "avg02", hdf5_dcomplex_id, avg_bond_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[3] = { qbond02_state, qbond02_op, qbond02_state };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_IN, TENSOR_AXIS_IN, TENSOR_AXIS_OUT };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&avg_bond_dns, axis_dir, qnums, &avg_bonds[0]);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&avg_bond_dns, &avg_bonds[0], 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&avg_bond_dns);
	}
	// bond (1, 2)
	{
		// read dense tensor from disk
		struct dense_tensor avg_bond_dns;
		const long dim[3] = { dim_bond12_state, dim_bond12_op, dim_bond12_state };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &avg_bond_dns);
		if (read_hdf5_dataset(file, "avg12", hdf5_dcomplex_id, avg_bond_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[3] = { qbond12_state, qbond12_op, qbond12_state };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_IN, TENSOR_AXIS_IN, TENSOR_AXIS_OUT };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&avg_bond_dns, axis_dir, qnums, &avg_bonds[1]);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&avg_bond_dns, &avg_bonds[1], 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&avg_bond_dns);
	}
	// bond (2, 3)
	{
		// read dense tensor from disk
		struct dense_tensor avg_bond_dns;
		const long dim[3] = { dim_bond23_state, dim_bond23_op, dim_bond23_state };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &avg_bond_dns);
		if (read_hdf5_dataset(file, "avg23", hdf5_dcomplex_id, avg_bond_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[3] = { qbond23_state, qbond23_op, qbond23_state };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&avg_bond_dns, axis_dir, qnums, &avg_bonds[3]);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&avg_bond_dns, &avg_bonds[3], 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&avg_bond_dns);
	}

	struct block_sparse_tensor env_parent;
	{
		// read dense tensor from disk
		struct dense_tensor env_parent_dns;
		const long dim[3] = { dim_bond24_state, dim_bond24_op, dim_bond24_state };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim, &env_parent_dns);
		if (read_hdf5_dataset(file, "env_parent", hdf5_dcomplex_id, env_parent_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		const qnumber* qnums[3] = { qbond24_state, qbond24_op, qbond24_state };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&env_parent_dns, axis_dir, qnums, &env_parent);

		// verify that sparsity pattern is correctly described by quantum numbers
		if (!dense_block_sparse_tensor_allclose(&env_parent_dns, &env_parent, 0)) {
			return "quantum number sparsity pattern mismatch";
		}

		delete_dense_tensor(&env_parent_dns);
	}

	// compute updated tensor
	struct block_sparse_tensor c1;
	bug_flow_update_connecting_tensor(&a_op, &c0, &topology, 2, 4, avg_bonds, &env_parent, &prefactor, dt, &c1);

	// local reference tensor of updated quantum state
	struct dense_tensor c1_ref;
	{
		// read dense tensor from disk
		hsize_t dim_file[5];
		if (get_hdf5_dataset_dims(file, "c1", dim_file) < 0) {
			return "obtaining dimensions of reference tensor failed";
		}
		const long dim[5] = { dim_file[0], dim_file[1], dim_file[2], dim_file[3], dim_file[4] };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, dim, &c1_ref);
		if (read_hdf5_dataset(file, "c1", hdf5_dcomplex_id, c1_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}
	}

	// compare
	if (!dense_block_sparse_tensor_allclose(&c1_ref, &c1, 1e-14)) {
		return "flow-updated connecting tensor does not match reference";
	}

	delete_dense_tensor(&c1_ref);
	delete_block_sparse_tensor(&c1);
	delete_block_sparse_tensor(&env_parent);
	delete_block_sparse_tensor(&avg_bonds[3]);
	delete_block_sparse_tensor(&avg_bonds[1]);
	delete_block_sparse_tensor(&avg_bonds[0]);
	delete_block_sparse_tensor(&a_op);
	delete_block_sparse_tensor(&c0);
	ct_free(qbond24_op);
	ct_free(qbond02_op);
	ct_free(qbond23_op);
	ct_free(qbond12_op);
	ct_free(qbond24_state);
	ct_free(qbond02_state);
	ct_free(qbond23_state);
	ct_free(qbond12_state);
	ct_free(qsite);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_bug_tree_time_step()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_bug_tree_time_step.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bug_tree_time_step failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of physical and branching lattice sites
	const int nsites_physical  = 8;
	const int nsites_branching = 1;
	const int nsites = nsites_physical + nsites_branching;
	// local physical dimension
	const long d = 2;

	// tree topology:
	//
	//          7
	//         ╱ ╲
	//        ╱   ╲
	//       ╱     ╲
	//      ╱       ╲
	//     3         8
	//    ╱ ╲       ╱│╲
	//   ╱   ╲     ╱ │ ╲
	//  0     2   4  5  6
	//        │
	//        │
	//        1
	//
	int neigh0[] = { 3 };
	int neigh1[] = { 2 };
	int neigh2[] = { 1, 3 };
	int neigh3[] = { 0, 2, 7 };
	int neigh4[] = { 8 };
	int neigh5[] = { 8 };
	int neigh6[] = { 8 };
	int neigh7[] = { 3, 8 };
	int neigh8[] = { 4, 5, 6, 7 };
	int* neighbor_map[9] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8,
	};
	int num_neighbors[9] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6), ARRLEN(neigh7), ARRLEN(neigh8),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites_physical + nsites_branching,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	const int i_root = 7;

	// construct Bose-Hubbard Hamiltonian as TTNO with complex entries
	struct ttno hamiltonian;
	{
		double th;
		if (read_hdf5_attribute(file, "th", H5T_NATIVE_DOUBLE, &th) < 0) {
			return "reading 'th' coefficient from disk failed";
		}
		double u;
		if (read_hdf5_attribute(file, "u", H5T_NATIVE_DOUBLE, &u) < 0) {
			return "reading 'u' coefficient from disk failed";
		}
		double mu;
		if (read_hdf5_attribute(file, "mu", H5T_NATIVE_DOUBLE, &mu) < 0) {
			return "reading 'mu' coefficient from disk failed";
		}

		struct ttno_assembly assembly;
		construct_bose_hubbard_1d_ttno_assembly_complex(nsites_physical, d, th, u, mu, &topology, &assembly);
		if (!ttno_graph_is_consistent(&assembly.graph)) {
			return "constructed TTNO graph is inconsistent";
		}

		ttno_from_assembly(&assembly, &hamiltonian);

		delete_ttno_assembly(&assembly);
	}

	// initial state as TTNS
	struct ttns state;
	{
		// local physical dimensions and quantum numbers
		const long d_list[9] = { d, d, d, d, d, d, d, d, 1 };
		const qnumber qsite_phys[2] = { 0, 1 };
		const qnumber qsite_branch[1]  = { 0 };
		const qnumber* qsite[9] = {
			qsite_phys, qsite_phys, qsite_phys, qsite_phys, qsite_phys, qsite_phys, qsite_phys, qsite_phys, qsite_branch,
		};

		qnumber qnum_sector;
		if (read_hdf5_attribute(file, "qnum_sector", H5T_NATIVE_INT, &qnum_sector) < 0) {
			return "reading quantum number sector from disk failed";
		}

		// virtual bond dimensions and quantum numbers
		long* dim_bonds  = ct_calloc(nsites*nsites, sizeof(long));
		qnumber** qbonds = ct_calloc(nsites*nsites, sizeof(qnumber*));
		for (int l = 0; l < nsites; l++)
		{
			for (int n = 0; n < topology.num_neighbors[l]; n++)
			{
				const int k = topology.neighbor_map[l][n];
				assert(k != l);
				if (k > l) {
					continue;
				}
				char varname[1024];
				sprintf(varname, "qbond%i%i", k, l);
				if (read_quantum_numbers(file, varname, &dim_bonds[k*nsites + l], &qbonds[k*nsites + l]) < 0) {
					return "reading virtual bond quantum numbers from disk failed";
				}
			}
		}

		allocate_ttns(CT_DOUBLE_COMPLEX, nsites_physical, &topology, d_list, qsite, qnum_sector, dim_bonds, (const qnumber**)qbonds, &state);

		if (!ttns_is_consistent(&state)) {
			return "internal TTNS consistency check failed";
		}
		if (ttns_quantum_number_sector(&state) != qnum_sector) {
			return "TTNS quantum number sector differs from expected value";
		}

		for (int l = 0; l < nsites; l++)
		{
			for (int n = 0; n < topology.num_neighbors[l]; n++)
			{
				const int k = topology.neighbor_map[l][n];
				assert(k != l);
				if (k > l) {
					continue;
				}
				ct_free(qbonds[k*nsites + l]);
			}
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		for (int l = 0; l < nsites; l++)
		{
			struct dense_tensor a_dns;
			allocate_dense_tensor(state.a[l].dtype, state.a[l].ndim, state.a[l].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "a%i_init", l);
			if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &state.a[l]);

			// verify that sparsity pattern is correctly described by quantum numbers
			if (!dense_block_sparse_tensor_allclose(&a_dns, &state.a[l], 0)) {
				return "quantum number sparsity pattern mismatch";
			}

			delete_dense_tensor(&a_dns);
		}
	}

	// integration parameters
	dcomplex prefactor;
	if (read_hdf5_attribute(file, "prefactor", hdf5_dcomplex_id, &prefactor) < 0) {
		return "reading integration prefactor from disk failed";
	}
	double dt;
	if (read_hdf5_attribute(file, "dt", H5T_NATIVE_DOUBLE, &dt) < 0) {
		return "reading time step from disk failed";
	}
	double rel_tol_compress;
	if (read_hdf5_attribute(file, "rel_tol", H5T_NATIVE_DOUBLE, &rel_tol_compress) < 0) {
		return "reading relative compression tolerance from disk failed";
	}
	const long max_vdim = 1024;
	int nsteps;
	if (read_hdf5_attribute(file, "nsteps", H5T_NATIVE_INT, &nsteps) < 0) {
		return "reading number of time steps from disk failed";
	}

	// perform BUG time integration
	for (int i = 0; i < nsteps; i++)
	{
		int ret = bug_tree_time_step(&hamiltonian, i_root, &prefactor, dt, rel_tol_compress, max_vdim, &state);
		if (ret < 0) {
			return "'bug_tree_time_step' failed internally";
		}
	}

	struct block_sparse_tensor vec;
	ttns_to_statevector(&state, &vec);

	// read reference vector from disk
	struct dense_tensor vec_ref;
	long dim_vec_ref[2] = { ipow(d, nsites_physical), 1 };  // include (dummy) auxiliary second dimension
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim_vec_ref, &vec_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "y", hdf5_dcomplex_id, vec_ref.data) < 0) {
		return "reading vector entries from disk failed";
	}

	// compare with reference
	if (!dense_block_sparse_tensor_allclose(&vec_ref, &vec, 1e-13)) {
		return "state vector after BUG integration step does not match reference";
	}

	delete_dense_tensor(&vec_ref);
	delete_block_sparse_tensor(&vec);
	delete_ttns(&state);
	delete_ttno(&hamiltonian);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
