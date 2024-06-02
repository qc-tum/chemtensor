/// \file hamiltonian.c
/// \brief Construction of common quantum Hamiltonians.

#include <math.h>
#include <assert.h>
#include "hamiltonian.h"
#include "mpo_graph.h"
#include "linked_list.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Construct a Hamiltonian as MPO based on local operator chains, which are shifted along a 1D lattice.
///
static void local_opchains_to_mpo(const enum numeric_type dtype, const long d, const qnumber* qsite, const int nsites,
	const struct op_chain* lopchains, const int nlopchains,
	const struct dense_tensor* opmap, const int oid_identity, struct mpo* mpo)
{
	int nchains = 0;
	for (int j = 0; j < nlopchains; j++)
	{
		assert(lopchains[j].length <= nsites);
		nchains += (nsites - lopchains[j].length + 1);
	}
	struct op_chain* opchains = aligned_alloc(MEM_DATA_ALIGN, nchains * sizeof(struct op_chain));
	int c = 0;
	for (int j = 0; j < nlopchains; j++)
	{
		for (int i = 0; i < nsites - lopchains[j].length + 1; i++)
		{
			// add shifted opchain; shallow copy sufficient here
			memcpy(&opchains[c], &lopchains[j], sizeof(struct op_chain));
			opchains[c].istart = i;
			c++;
		}
	}
	assert(c == nchains);

	struct mpo_graph graph;
	mpo_graph_from_opchains(opchains, nchains, nsites, oid_identity, &graph);

	mpo_from_graph(dtype, d, qsite, &graph, opmap, mpo);

	delete_mpo_graph(&graph);
	aligned_free(opchains);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contruct an MPO representation of the Ising Hamiltonian 'sum J Z Z + h Z + g X' on a one-dimensional lattice.
///
void construct_ising_1d_mpo(const int nsites, const double J, const double h, const double g, struct mpo* mpo)
{
	assert(nsites >= 2);

	// set physical quantum numbers to zero
	const qnumber qsite[2] = { 0, 0 };

	// operator map
	// 0: I
	// 1: Z
	// 2: h Z + g X
	struct dense_tensor opmap[3];
	for (int i = 0; i < 3; i++) {
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
	}
	const double sz[4] = { 1., 0., 0., -1. };  // Z
	const double fd[4] = { h,  g,  g,  -h  };  // external field term 'h Z + g X'
	dense_tensor_set_identity(&opmap[0]);
	memcpy(opmap[1].data, sz, sizeof(sz));
	memcpy(opmap[2].data, fd, sizeof(fd));

	// local two-site and single-site terms
	int oids_c0[2] = { 1, 1 };  qnumber qnums_c0[3] = { 0, 0, 0 };
	int oids_c1[1] = { 2 };     qnumber qnums_c1[2] = { 0, 0 };
	struct op_chain lopchains[2] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .coeff = J,  .length = 2, .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .coeff = 1., .length = 1, .istart = 0 },
	};

	// convert to MPO
	local_opchains_to_mpo(DOUBLE_REAL, 2, qsite, nsites, lopchains, sizeof(lopchains) / sizeof(struct op_chain), opmap, 0, mpo);

	// clean up
	for (int i = 0; i < 3; i++) {
		delete_dense_tensor(&opmap[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO representation of the XXZ Heisenberg Hamiltonian 'sum J (X X + Y Y + D Z Z) - h Z' on a one-dimensional lattice.
///
void construct_heisenberg_xxz_1d_mpo(const int nsites, const double J, const double D, const double h, struct mpo* mpo)
{
	assert(nsites >= 2);

	// physical quantum numbers (multiplied by 2)
	const qnumber qsite[2] = { 1, -1 };

	// spin operators
	const double sup[4] = { 0.,  1.,  0.,  0.  };  // S_up
	const double sdn[4] = { 0.,  0.,  1.,  0.  };  // S_down
	const double  sz[4] = { 0.5, 0.,  0., -0.5 };  // S_z

	// operator map
	// 0: I
	// 1: S_up
	// 2: S_down
	// 3: S_z
	struct dense_tensor opmap[4];
	for (int i = 0; i < 4; i++) {
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
	}
	dense_tensor_set_identity(&opmap[0]);
	memcpy(opmap[1].data, sup, sizeof(sup));
	memcpy(opmap[2].data, sdn, sizeof(sdn));
	memcpy(opmap[3].data, sz,  sizeof(sz));

	// local two-site and single-site terms
	int oids_c0[2] = { 1, 2 };  qnumber qnums_c0[3] = { 0,  2,  0 };
	int oids_c1[2] = { 2, 1 };  qnumber qnums_c1[3] = { 0, -2,  0 };
	int oids_c2[2] = { 3, 3 };  qnumber qnums_c2[3] = { 0,  0,  0 };
	int oids_c3[1] = { 3 };     qnumber qnums_c3[2] = { 0,  0 };
	struct op_chain lopchains[4] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .coeff = 0.5*J, .length = 2, .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .coeff = 0.5*J, .length = 2, .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .coeff = J*D,   .length = 2, .istart = 0 },
		{ .oids = oids_c3, .qnums = qnums_c3, .coeff = -h,    .length = 1, .istart = 0 },
	};

	// convert to MPO
	local_opchains_to_mpo(DOUBLE_REAL, 2, qsite, nsites, lopchains, sizeof(lopchains) / sizeof(struct op_chain), opmap, 0, mpo);

	// clean up
	for (int i = 0; i < 4; i++) {
		delete_dense_tensor(&opmap[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO representation of the Bose-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
void construct_bose_hubbard_1d_mpo(const int nsites, const long d, const double t, const double u, const double mu, struct mpo* mpo)
{
	assert(nsites >= 2);
	assert(d >= 1);

	// physical quantum numbers (particle number)
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	for (long i = 0; i < d; i++) {
		qsite[i] = i;
	}

	// operator map
	// 0: I
	// 1: b_{\dagger}
	// 2: b
	// 3: u n (n - 1)/2 - mu n
	struct dense_tensor opmap[4];
	for (int i = 0; i < 4; i++) {
		const long dim[2] = { d, d };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
	}
	// identity operator
	dense_tensor_set_identity(&opmap[0]);
	// bosonic creation operator
	double* b_dag = opmap[1].data;
	for (long i = 0; i < d - 1; i++) {
		b_dag[(i + 1)*d + i] = sqrt(i + 1);
	}
	// bosonic annihilation operator
	double* b_ann = opmap[2].data;
	for (long i = 0; i < d - 1; i++) {
		b_ann[i*d + (i + 1)] = sqrt(i + 1);
	}
	// interaction term
	double* v = opmap[3].data;
	for (long i = 0; i < d; i++) {
		v[i*d + i] = 0.5*u*i*(i - 1) - mu*i;
	}

	// local two-site and single-site terms
	int oids_c0[2] = { 1, 2 };  qnumber qnums_c0[3] = { 0,  1,  0 };
	int oids_c1[2] = { 2, 1 };  qnumber qnums_c1[3] = { 0, -1,  0 };
	int oids_c2[1] = { 3 };     qnumber qnums_c2[2] = { 0,  0 };
	struct op_chain lopchains[3] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .coeff =  1., .length = 1, .istart = 0 },
	};

	// convert to MPO
	local_opchains_to_mpo(DOUBLE_REAL, d, qsite, nsites, lopchains, sizeof(lopchains) / sizeof(struct op_chain), opmap, 0, mpo);

	// clean up
	for (int i = 0; i < 4; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	aligned_free(qsite);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO representation of the Fermi-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
/// States for each spin and site are '|0>' and '|1>'.
///
void construct_fermi_hubbard_1d_mpo(const int nsites, const double t, const double u, const double mu, struct mpo* mpo)
{
	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = { 0,  1,  1,  2 };
	const qnumber qs[4] = { 0, -1,  1,  0 };
	const qnumber qsite[4] = {
		(qn[0] << 16) + qs[0],
		(qn[1] << 16) + qs[1],
		(qn[2] << 16) + qs[2],
		(qn[3] << 16) + qs[3],
	};

	struct dense_tensor id;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &id);
		dense_tensor_set_identity(&id);
	}
	// creation and annihilation operators for a single spin and lattice site
	struct dense_tensor a_dag;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &a_dag);
		const double data[4] = { 0., 0., 1., 0. };
		memcpy(a_dag.data, data, sizeof(data));
	}
	struct dense_tensor a_ann;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &a_ann);
		const double data[4] = { 0., 1., 0., 0. };
		memcpy(a_ann.data, data, sizeof(data));
	}
	// number operator
	struct dense_tensor numop;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &numop);
		const double data[4] = { 0., 0., 0., 1. };
		memcpy(numop.data, data, sizeof(data));
	}
	// Pauli-Z matrix required for Jordan-Wigner transformation
	struct dense_tensor z;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &z);
		const double data[4] = { 1.,  0.,  0., -1. };
		memcpy(z.data, data, sizeof(data));
	}
	// local interaction and potential term
	struct dense_tensor vint;
	{
		const long dim[2] = { 4, 4 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &vint);
		const double diag[4] = { u/4, -u/4 - mu, -u/4 - mu, u/4 - 2*mu };
		double* data = vint.data;
		for (int i = 0; i < 4; i++) {
			data[i*4 + i] = diag[i];
		}
	}

	// operator map
	// 0:  I x I
	// 1: ad x I
	// 2:  a x I
	// 3: ad x Z
	// 4:  a x Z
	// 5:  I x ad
	// 6:  I x a
	// 7:  Z x ad
	// 8:  Z x a
	// 9:  u (n_up - 1/2) (n_dn - 1/2) - mu (n_up + n_dn)
	struct dense_tensor opmap[10];
	dense_tensor_kronecker_product(&id,    &id,    &opmap[0]);
	dense_tensor_kronecker_product(&a_dag, &id,    &opmap[1]);
	dense_tensor_kronecker_product(&a_ann, &id,    &opmap[2]);
	dense_tensor_kronecker_product(&a_dag, &z,     &opmap[3]);
	dense_tensor_kronecker_product(&a_ann, &z,     &opmap[4]);
	dense_tensor_kronecker_product(&id,    &a_dag, &opmap[5]);
	dense_tensor_kronecker_product(&id,    &a_ann, &opmap[6]);
	dense_tensor_kronecker_product(&z,     &a_dag, &opmap[7]);
	dense_tensor_kronecker_product(&z,     &a_ann, &opmap[8]);
	copy_dense_tensor(&vint, &opmap[9]);

	// cast to unsigned integer to avoid compiler warning when bit-shifting
	const unsigned int n1 = -1;

	// local two-site and single-site terms
	// spin-up kinetic hopping
	int oids_c0[2] = { 3, 2 };  qnumber qnums_c0[3] = { 0, ( 1 << 16) + 1, 0 };
	int oids_c1[2] = { 4, 1 };  qnumber qnums_c1[3] = { 0, (n1 << 16) - 1, 0 };
	// spin-down kinetic hopping
	int oids_c2[2] = { 5, 8 };  qnumber qnums_c2[3] = { 0, ( 1 << 16) - 1, 0 };
	int oids_c3[2] = { 6, 7 };  qnumber qnums_c3[3] = { 0, (n1 << 16) + 1, 0 };
	// interaction u (n_up-1/2) (n_dn-1/2) and number operator - mu (n_up + n_dn)
	int oids_c4[1] = { 9 };     qnumber qnums_c4[2] = { 0, 0 };
	struct op_chain lopchains[5] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c3, .qnums = qnums_c3, .coeff = -t,  .length = 2, .istart = 0 },
		{ .oids = oids_c4, .qnums = qnums_c4, .coeff =  1., .length = 1, .istart = 0 },
	};

	// convert to MPO
	local_opchains_to_mpo(DOUBLE_REAL, 4, qsite, nsites, lopchains, sizeof(lopchains) / sizeof(struct op_chain), opmap, 0, mpo);

	// clean up
	for (int i = 0; i < 10; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	delete_dense_tensor(&vint);
	delete_dense_tensor(&z);
	delete_dense_tensor(&numop);
	delete_dense_tensor(&a_ann);
	delete_dense_tensor(&a_dag);
	delete_dense_tensor(&id);
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two integers.
///
static inline int imin(const int a, const int b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two integers.
///
static inline int imax(const int a, const int b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate and initialize MPO graph vertex indices for each virtual bond (utility function).
///
static int* allocate_vertex_ids(const int nsites)
{
	int* vids = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(int));
	// initialize by invalid index
	for (int i = 0; i < nsites + 1; i++) {
		vids[i] = -1;
	}
	return vids;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate and initialize an MPO graph edge with a single local operator reference.
///
static struct mpo_graph_edge* construct_mpo_graph_edge(const int vid0, const int vid1, const int oid, const double coeff)
{
	struct mpo_graph_edge* edge = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct mpo_graph_edge));
	edge->vids[0] = vid0;
	edge->vids[1] = vid1;
	edge->opics = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct local_op_ref));
	edge->opics[0].oid = oid;
	edge->opics[0].coeff = coeff;
	edge->nopics = 1;

	return edge;
}

//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize the interaction coefficient tensor as
/// vint - transpose(vint, (1, 0, 2, 3)) - transpose(vint, (0, 1, 3, 2)) + transpose(vint, (1, 0, 3, 2))
///
static inline void symmetrize_interaction_coefficients(const struct dense_tensor* restrict vint, struct dense_tensor* restrict gint)
{
	assert(vint->ndim == 4);

	copy_dense_tensor(vint, gint);

	// - transpose(vint, (1, 0, 2, 3))
	{
		const int perm[4] = { 1, 0, 2, 3 };
		struct dense_tensor vint_tp;
		transpose_dense_tensor(perm, vint, &vint_tp);
		dense_tensor_scalar_multiply_add(numeric_neg_one(vint->dtype), &vint_tp, gint);
		delete_dense_tensor(&vint_tp);
	}

	// - transpose(vint, (0, 1, 3, 2))
	{
		const int perm[4] = { 0, 1, 3, 2 };
		struct dense_tensor vint_tp;
		transpose_dense_tensor(perm, vint, &vint_tp);
		dense_tensor_scalar_multiply_add(numeric_neg_one(vint->dtype), &vint_tp, gint);
		delete_dense_tensor(&vint_tp);
	}

	// + transpose(vint, (1, 0, 3, 2))
	{
		const int perm[4] = { 1, 0, 3, 2 };
		struct dense_tensor vint_tp;
		transpose_dense_tensor(perm, vint, &vint_tp);
		dense_tensor_scalar_multiply_add(numeric_one(vint->dtype), &vint_tp, gint);
		delete_dense_tensor(&vint_tp);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a molecular Hamiltonian as MPO,
/// using physicists' convention for the interaction term (note ordering of k and l):
/// \f[
/// H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
/// \f]
///
void construct_molecular_hamiltonian_mpo(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo* mpo)
{
	assert(tkin->dtype == DOUBLE_REAL);
	assert(vint->dtype == DOUBLE_REAL);

	// dimension consistency checks
	assert(tkin->ndim == 2);
	assert(vint->ndim == 4);
	assert(tkin->dim[0] == tkin->dim[1]);
	assert(vint->dim[0] == vint->dim[1] &&
	       vint->dim[0] == vint->dim[2] &&
	       vint->dim[0] == vint->dim[3]);
	assert(tkin->dim[0] == vint->dim[0]);

	// number of "sites" (orbitals)
	const int nsites = tkin->dim[0];
	assert(nsites >= 1);

	// local operators
	// creation and annihilation operators for a single spin and lattice site
	const double a_ann[4] = { 0.,  1.,  0.,  0. };
	const double a_dag[4] = { 0.,  0.,  1.,  0. };
	// number operator
	const double numop[4] = { 0.,  0.,  0.,  1. };
	// Pauli-Z matrix required for Jordan-Wigner transformation
	const double z[4]     = { 1.,  0.,  0., -1. };

	// operator map
	// 0: I
	// 1: a^{\dagger}
	// 2: a
	// 3: numop
	// 4: Z
	struct dense_tensor opmap[5];
	for (int i = 0; i < 5; i++) {
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
	}
	dense_tensor_set_identity(&opmap[0]);
	memcpy(opmap[1].data, a_dag, sizeof(a_dag));
	memcpy(opmap[2].data, a_ann, sizeof(a_ann));
	memcpy(opmap[3].data, numop, sizeof(numop));
	memcpy(opmap[4].data, z,     sizeof(z));

	const int OID_IDENT = 0;
	const int OID_A_DAG = 1;
	const int OID_A_ANN = 2;
	const int OID_NUMOP = 3;
	const int OID_Z     = 4;

	// interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
	// can anti-commute fermionic operators such that i < j and l < k
	struct dense_tensor gint;
	symmetrize_interaction_coefficients(vint, &gint);
	// global minus sign from Jordan-Wigner transformation, since a Z = -a
	const double neg05 = -0.5;
	scale_dense_tensor(&neg05, &gint);
	assert(gint.dtype == DOUBLE_REAL);

	struct mpo_graph mpo_graph = {
		.verts     = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(struct mpo_graph_vertex*)),
		.edges     = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(struct mpo_graph_edge*)),
		.num_verts = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(int)),
		.num_edges = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(int)),
		.nsites    = nsites,
	};

	for (int i = 0; i < nsites + 1; i++)
	{
		// virtual bond dimensions
		const int nl = i;
		const int nr = nsites - i;
		const int n = imin(nl, nr);
		// identity chains
		int chi1 = (1 < i && i < nsites - 1) ? 2 : 1;
		// a^{\dagger}_i a^{\dagger}_j (for i < j), a_i a_j (for i < j) and a^{\dagger}_i a_j chains, extending from boundary to center
		int chi2 = n * (n - 1) + n * n;
		// a^{\dagger}_i and a_i chains, reaching (almost) from one boundary to the other
		int chi3 = 2 * ((i < nsites - 1 ? nl : 0) + (i > 1 ? nr : 0));

		mpo_graph.num_verts[i] = chi1 + chi2 + chi3;
		mpo_graph.verts[i] = aligned_calloc(MEM_DATA_ALIGN, mpo_graph.num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	int* node_counter = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(int));

	// node IDs
	// identity chains from the left and right
	int* vids_identity_l = allocate_vertex_ids(nsites);
	int* vids_identity_r = allocate_vertex_ids(nsites);
	for (int i = 0; i < nsites - 1; i++) {
		vids_identity_l[i] = node_counter[i]++;
	}
	for (int i = 2; i < nsites + 1; i++) {
		vids_identity_r[i] = node_counter[i]++;
	}
	// a^{\dagger}_i operators connected to left terminal
	int** vids_a_dag_l = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(int*));
	for (int i = 0; i < nsites - 2; i++) {
		vids_a_dag_l[i] = allocate_vertex_ids(nsites);
		for (int j = i + 1; j < nsites - 1; j++) {
			const int vid = node_counter[j]++;
			vids_a_dag_l[i][j] = vid;
			mpo_graph.verts[j][vid].qnum = 1;
		}
	}
	// a_i operators connected to left terminal
	int** vids_a_ann_l = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(int*));
	for (int i = 0; i < nsites - 2; i++) {
		vids_a_ann_l[i] = allocate_vertex_ids(nsites);
		for (int j = i + 1; j < nsites - 1; j++) {
			const int vid = node_counter[j]++;
			vids_a_ann_l[i][j] = vid;
			mpo_graph.verts[j][vid].qnum = -1;
		}
	}
	// a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
	int** vids_a_dag_a_dag_l = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			vids_a_dag_a_dag_l[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = j + 1; k < nsites/2 + 1; k++) {
				const int vid = node_counter[k]++;
				vids_a_dag_a_dag_l[i*nsites + j][k] = vid;
				mpo_graph.verts[k][vid].qnum = 2;
			}
		}
	}
	// a_i a_j operators connected to left terminal
	int** vids_a_ann_a_ann_l = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			vids_a_ann_a_ann_l[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = j + 1; k < nsites/2 + 1; k++) {
				const int vid = node_counter[k]++;
				vids_a_ann_a_ann_l[i*nsites + j][k] = vid;
				mpo_graph.verts[k][vid].qnum = -2;
			}
		}
	}
	// a^{\dagger}_i a_j operators connected to left terminal
	int** vids_a_dag_a_ann_l = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			vids_a_dag_a_ann_l[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = imax(i, j) + 1; k < nsites/2 + 1; k++) {
				vids_a_dag_a_ann_l[i*nsites + j][k] = node_counter[k]++;
				// vertex quantum number is zero
			}
		}
	}
	// a^{\dagger}_i operators connected to right terminal
	int** vids_a_dag_r = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(int*));
	for (int i = 2; i < nsites; i++) {
		vids_a_dag_r[i] = allocate_vertex_ids(nsites);
		for (int j = 2; j < i + 1; j++) {
			const int vid = node_counter[j]++;
			vids_a_dag_r[i][j] = vid;
			mpo_graph.verts[j][vid].qnum = -1;
		}
	}
	// a_i operators connected to right terminal
	int** vids_a_ann_r = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(int*));
	for (int i = 2; i < nsites; i++) {
		vids_a_ann_r[i] = allocate_vertex_ids(nsites);
		for (int j = 2; j < i + 1; j++) {
			const int vid = node_counter[j]++;
			vids_a_ann_r[i][j] = vid;
			mpo_graph.verts[j][vid].qnum = 1;
		}
	}
	// a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
	int** vids_a_dag_a_dag_r = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			vids_a_dag_a_dag_r[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < i + 1; k++) {
				const int vid = node_counter[k]++;
				vids_a_dag_a_dag_r[i*nsites + j][k] = vid;
				mpo_graph.verts[k][vid].qnum = -2;
			}
		}
	}
	// a_i a_j operators connected to right terminal
	int** vids_a_ann_a_ann_r = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			vids_a_ann_a_ann_r[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < i + 1; k++) {
				const int vid = node_counter[k]++;
				vids_a_ann_a_ann_r[i*nsites + j][k] = vid;
				mpo_graph.verts[k][vid].qnum = 2;
			}
		}
	}
	// a^{\dagger}_i a_j operators connected to right terminal
	int** vids_a_dag_a_ann_r = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			vids_a_dag_a_ann_r[i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < imin(i, j) + 1; k++) {
				vids_a_dag_a_ann_r[i*nsites + j][k] = node_counter[k]++;
				// vertex quantum number is zero
			}
		}
	}
	// consistency check
	for (int i = 0; i < nsites + 1; i++) {
		assert(node_counter[i] == mpo_graph.num_verts[i]);
	}

	// edges
	// temporarily store edges in linked lists
	struct linked_list* edges = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(struct linked_list));
	// identities connected to left and right terminals
	for (int i = 0; i < nsites - 2; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_identity_l[i], vids_identity_l[i + 1], OID_IDENT, 1.));
	}
	for (int i = 2; i < nsites; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_identity_r[i], vids_identity_r[i + 1], OID_IDENT, 1.));
	}
	// a^{\dagger}_i operators connected to left terminal
	for (int i = 0; i < nsites - 2; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_identity_l[i], vids_a_dag_l[i][i + 1], OID_A_DAG, 1.));
		// Z operator from Jordan-Wigner transformation
		for (int j = i + 1; j < nsites - 2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_a_dag_l[i][j + 1], OID_Z, 1.));
		}
	}
	// a_i operators connected to left terminal
	for (int i = 0; i < nsites - 2; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_identity_l[i], vids_a_ann_l[i][i + 1], OID_A_ANN, 1.));
		// Z operator from Jordan-Wigner transformation
		for (int j = i + 1; j < nsites - 2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_ann_l[i][j], vids_a_ann_l[i][j + 1], OID_Z, 1.));
		}
	}
	// a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_a_dag_a_dag_l[i*nsites + j][j + 1], OID_A_DAG, 1.));
			// identities for transition to next site
			for (int k = j + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_dag_a_dag_l[i*nsites + j][k], vids_a_dag_a_dag_l[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
		}
	}
	// a_i a_j operators connected to left terminal
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_ann_l[i][j], vids_a_ann_a_ann_l[i*nsites + j][j + 1], OID_A_ANN, 1.));
			// identities for transition to next site
			for (int k = j + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_ann_a_ann_l[i*nsites + j][k], vids_a_ann_a_ann_l[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
		}
	}
	// a^{\dagger}_i a_j operators connected to left terminal
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			if (i < j) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_a_dag_a_ann_l[i*nsites + j][j + 1], OID_A_ANN, 1.));
			}
			else if (i == j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids_identity_l[i], vids_a_dag_a_ann_l[i*nsites + j][i + 1], OID_NUMOP, 1.));
			}
			else { // i > j
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids_a_ann_l[j][i], vids_a_dag_a_ann_l[i*nsites + j][i + 1], OID_A_DAG, 1.));
			}
			// identities for transition to next site
			for (int k = imax(i, j) + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + j][k], vids_a_dag_a_ann_l[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
		}
	}
	// a^{\dagger}_i operators connected to right terminal
	for (int i = 2; i < nsites; i++) {
		// Z operator from Jordan-Wigner transformation
		for (int j = 2; j < i; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_dag_r[i][j], vids_a_dag_r[i][j + 1], OID_Z, 1.));
		}
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_a_dag_r[i][i], vids_identity_r[i + 1], OID_A_DAG, 1.));
	}
	// a_i operators connected to right terminal
	for (int i = 2; i < nsites; i++) {
		// Z operator from Jordan-Wigner transformation
		for (int j = 2; j < i; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_ann_r[i][j], vids_a_ann_r[i][j + 1], OID_Z, 1.));
		}
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids_a_ann_r[i][i], vids_identity_r[i + 1], OID_A_ANN, 1.));
	}
	// a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < i; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_dag_a_dag_r[i*nsites + j][k], vids_a_dag_a_dag_r[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids_a_dag_a_dag_r[i*nsites + j][i], vids_a_dag_r[j][i + 1], OID_A_DAG, 1.));
		}
	}
	// a_i a_j operators connected to right terminal
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < i; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_ann_a_ann_r[i*nsites + j][k], vids_a_ann_a_ann_r[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids_a_ann_a_ann_r[i*nsites + j][i], vids_a_ann_r[j][i + 1], OID_A_ANN, 1.));
		}
	}
	// a^{\dagger}_i a_j operators connected to right terminal
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < imin(i, j); k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_a_dag_a_ann_r[i*nsites + j][k], vids_a_dag_a_ann_r[i*nsites + j][k + 1], OID_IDENT, 1.));
			}
			if (i < j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids_a_dag_a_ann_r[i*nsites + j][i], vids_a_ann_r[j][i + 1], OID_A_DAG, 1.));
			}
			else if (i == j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids_a_dag_a_ann_r[i*nsites + j][i], vids_identity_r[i + 1], OID_NUMOP, 1.));
			}
			else { // i > j
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids_a_dag_a_ann_r[i*nsites + j][j], vids_a_dag_r[i][j + 1], OID_A_ANN, 1.));
			}
		}
	}
	const double* tkin_data = tkin->data;
	// diagonal kinetic terms t_{i,i} n_i
	for (int i = 0; i < nsites/2; i++) {
		linked_list_append(&edges[i + 1],
			construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + i][i + 1], vids_identity_r[i + 2], OID_IDENT, tkin_data[i*nsites + i]));
	}
	for (int i = nsites/2 + 1; i < nsites; i++) {
		linked_list_append(&edges[i - 1],
			construct_mpo_graph_edge(vids_identity_l[i - 1], vids_a_dag_a_ann_r[i*nsites + i][i], OID_IDENT, tkin_data[i*nsites + i]));
	}
	linked_list_append(&edges[nsites/2],
		construct_mpo_graph_edge(vids_identity_l[nsites/2], vids_identity_r[nsites/2 + 1], OID_NUMOP, tkin_data[(nsites/2)*nsites + (nsites/2)]));
	// t_{i,j} a^{\dagger}_i a_j terms, for i < j
	for (int i = 0; i < nsites/2; i++) {
		for (int j = i + 1; j < nsites/2 + 1; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_identity_r[j + 1], OID_A_ANN, tkin_data[i*nsites + j]));
		}
	}
	for (int j = nsites/2 + 1; j < nsites; j++) {
		for (int i = nsites/2; i < j; i++) {
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids_identity_l[i], vids_a_ann_r[j][i + 1], OID_A_DAG, tkin_data[i*nsites + j]));
		}
	}
	for (int i = 0; i < nsites/2; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			linked_list_append(&edges[nsites/2],
				construct_mpo_graph_edge(vids_a_dag_l[i][nsites/2], vids_a_ann_r[j][nsites/2 + 1], OID_Z, tkin_data[i*nsites + j]));
		}
	}
	// t_{i,j} a^{\dagger}_i a_j terms, for i > j
	for (int j = 0; j < nsites/2; j++) {
		for (int i = j + 1; i < nsites/2 + 1; i++) {
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids_a_ann_l[j][i], vids_identity_r[i + 1], OID_A_DAG, tkin_data[i*nsites + j]));
		}
	}
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2; j < i; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_identity_l[j], vids_a_dag_r[i][j + 1], OID_A_ANN, tkin_data[i*nsites + j]));
		}
	}
	for (int j = 0; j < nsites/2; j++) {
		for (int i = nsites/2 + 1; i < nsites; i++) {
			linked_list_append(&edges[nsites/2],
				construct_mpo_graph_edge(vids_a_ann_l[j][nsites/2], vids_a_dag_r[i][nsites/2 + 1], OID_Z, tkin_data[i*nsites + j]));
		}
	}
	const double* gint_data = gint.data;
	// g_{i,j,k,j} a^{\dagger}_i n_j a_k terms, for i < j < k
	for (int i = 0; i < nsites - 2; i++) {
		for (int j = i + 1; j < nsites - 1; j++) {
			for (int k = j + 1; k < nsites; k++) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_a_ann_r[k][j + 1], OID_NUMOP, gint_data[((i*nsites + j)*nsites + k)*nsites + j]));
			}
		}
	}
	// g_{i,j,i,l} a_l n_i a^{\dagger}_j terms, for l < i < j
	for (int l = 0; l < nsites - 2; l++) {
		for (int i = l + 1; i < nsites - 1; i++) {
			for (int j = i + 1; j < nsites; j++) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids_a_ann_l[l][i], vids_a_dag_r[j][i + 1], OID_NUMOP, gint_data[((i*nsites + j)*nsites + i)*nsites + l]));
			}
		}
	}
	// g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for i < j < l < k
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			for (int l = j + 1; l < nsites/2 + 1; l++) {
				for (int k = l + 1; k < nsites; k++) {
					linked_list_append(&edges[l],
						construct_mpo_graph_edge(vids_a_dag_a_dag_l[i*nsites + j][l], vids_a_ann_r[k][l + 1], OID_A_ANN, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	for (int l = nsites/2 + 1; l < nsites - 1; l++) {
		for (int k = l + 1; k < nsites; k++) {
			for (int j = nsites/2; j < l; j++) {
				for (int i = 0; i < j; i++) {
					linked_list_append(&edges[j],
						construct_mpo_graph_edge(vids_a_dag_l[i][j], vids_a_ann_a_ann_r[l*nsites + k][j + 1], OID_A_DAG, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			for (int l = nsites/2 + 1; l < nsites - 1; l++) {
				for (int k = l + 1; k < nsites; k++) {
					linked_list_append(&edges[nsites/2],
						construct_mpo_graph_edge(vids_a_dag_a_dag_l[i*nsites + j][nsites/2], vids_a_ann_a_ann_r[l*nsites + k][nsites/2 + 1], OID_IDENT, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	// g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for l < k < i < j
	for (int l = 0; l < nsites/2 - 1; l++) {
		for (int k = l + 1; k < nsites/2; k++) {
			for (int i = k + 1; i < nsites/2 + 1; i++) {
				for (int j = i + 1; j < nsites; j++) {
					linked_list_append(&edges[i],
						construct_mpo_graph_edge(vids_a_ann_a_ann_l[l*nsites + k][i], vids_a_dag_r[j][i + 1], OID_A_DAG, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			for (int k = nsites/2; k < i; k++) {
				for (int l = 0; l < k; l++) {
					linked_list_append(&edges[k],
						construct_mpo_graph_edge(vids_a_ann_l[l][k], vids_a_dag_a_dag_r[i*nsites + j][k + 1], OID_A_ANN, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	for (int l = 0; l < nsites/2 - 1; l++) {
		for (int k = l + 1; k < nsites/2; k++) {
			for (int i = nsites/2 + 1; i < nsites - 1; i++) {
				for (int j = i + 1; j < nsites; j++) {
					linked_list_append(&edges[nsites/2],
						construct_mpo_graph_edge(vids_a_ann_a_ann_l[l*nsites + k][nsites/2], vids_a_dag_a_dag_r[i*nsites + j][nsites/2 + 1], OID_IDENT, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}
	// g_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k terms, for i, l < j, k
	for (int i = 0; i < nsites/2; i++) {
		for (int l = 0; l < nsites/2; l++) {
			for (int j = imax(i, l) + 1; j < nsites; j++) {
				for (int k = imax(i, l) + 1; k < nsites; k++) {
					if (imin(j, k) > nsites/2) {
						continue;
					}
					if (j < k) {
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + l][j], vids_a_ann_r[k][j + 1], OID_A_DAG, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
					else if (j == k) {
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + l][j], vids_identity_r[j + 1], OID_NUMOP, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
					else { // j > k
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + l][k], vids_a_dag_r[j][k + 1], OID_A_ANN, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
				}
			}
		}
	}
	for (int j = nsites/2 + 1; j < nsites; j++) {
		for (int k = nsites/2 + 1; k < nsites; k++) {
			for (int i = 0; i < imin(j, k); i++) {
				for (int l = 0; l < imin(j, k); l++) {
					if (imax(i, l) < nsites/2) {
						continue;
					}
					if (i < l) {
						linked_list_append(&edges[l],
							construct_mpo_graph_edge(vids_a_dag_l[i][l], vids_a_dag_a_ann_r[j*nsites + k][l + 1], OID_A_ANN, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
					else if (i == l) {
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids_identity_l[i], vids_a_dag_a_ann_r[j*nsites + k][i + 1], OID_NUMOP, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
					else { // i > l
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids_a_ann_l[l][i], vids_a_dag_a_ann_r[j*nsites + k][i + 1], OID_A_DAG, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
					}
				}
			}
		}
	}
	for (int i = 0; i < nsites/2; i++) {
		for (int l = 0; l < nsites/2; l++) {
			for (int j = nsites/2 + 1; j < nsites; j++) {
				for (int k = nsites/2 + 1; k < nsites; k++) {
					linked_list_append(&edges[nsites/2],
						construct_mpo_graph_edge(vids_a_dag_a_ann_l[i*nsites + l][nsites/2], vids_a_dag_a_ann_r[j*nsites + k][nsites/2 + 1], OID_IDENT, gint_data[((i*nsites + j)*nsites + k)*nsites + l]));
				}
			}
		}
	}

	// transfer edges into mpo_graph structure and connect vertices
	for (int i = 0; i < nsites; i++)
	{
		mpo_graph.num_edges[i] = edges[i].size;
		mpo_graph.edges[i] = aligned_alloc(MEM_DATA_ALIGN, edges[i].size * sizeof(struct mpo_graph_edge));
		struct linked_list_node* edge_ref = edges[i].head;
		long eid = 0;
		while (edge_ref != NULL)
		{
			const struct mpo_graph_edge* edge = edge_ref->data;
			memcpy(&mpo_graph.edges[i][eid], edge, sizeof(struct mpo_graph_edge));

			// create references from graph vertices to edge
			assert(0 <= edge->vids[0] && edge->vids[0] < mpo_graph.num_verts[i]);
			assert(0 <= edge->vids[1] && edge->vids[1] < mpo_graph.num_verts[i + 1]);
			mpo_graph_vertex_add_edge(1, eid, &mpo_graph.verts[i    ][edge->vids[0]]);
			mpo_graph_vertex_add_edge(0, eid, &mpo_graph.verts[i + 1][edge->vids[1]]);

			edge_ref = edge_ref->next;
			eid++;
		}
		// note: opics pointers of edges have been retained in transfer
		delete_linked_list(&edges[i], aligned_free);
	}
	aligned_free(edges);

	assert(mpo_graph_is_consistent(&mpo_graph));

	// convert graph to MPO
	const qnumber qsite[2] = { 0, 1 };
	mpo_from_graph(DOUBLE_REAL, 2, qsite, &mpo_graph, opmap, mpo);

	// clean up
	aligned_free(vids_identity_l);
	aligned_free(vids_identity_r);
	for (int i = 0; i < nsites - 2; i++) {
		aligned_free(vids_a_dag_l[i]);
		aligned_free(vids_a_ann_l[i]);
	}
	aligned_free(vids_a_dag_l);
	aligned_free(vids_a_ann_l);
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			aligned_free(vids_a_dag_a_dag_l[i*nsites + j]);
			aligned_free(vids_a_ann_a_ann_l[i*nsites + j]);
		}
	}
	aligned_free(vids_a_dag_a_dag_l);
	aligned_free(vids_a_ann_a_ann_l);
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			aligned_free(vids_a_dag_a_ann_l[i*nsites + j]);
		}
	}
	aligned_free(vids_a_dag_a_ann_l);
	for (int i = 2; i < nsites; i++) {
		aligned_free(vids_a_dag_r[i]);
		aligned_free(vids_a_ann_r[i]);
	}
	aligned_free(vids_a_dag_r);
	aligned_free(vids_a_ann_r);
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			aligned_free(vids_a_dag_a_dag_r[i*nsites + j]);
			aligned_free(vids_a_ann_a_ann_r[i*nsites + j]);
		}
	}
	aligned_free(vids_a_dag_a_dag_r);
	aligned_free(vids_a_ann_a_ann_r);
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			aligned_free(vids_a_dag_a_ann_r[i*nsites + j]);
		}
	}
	aligned_free(vids_a_dag_a_ann_r);
	aligned_free(node_counter);
	delete_mpo_graph(&mpo_graph);
	delete_dense_tensor(&gint);
	for (int i = 0; i < 5; i++) {
		delete_dense_tensor(&opmap[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for sorting index - quantum number tuples.
///
struct index_qnumber_tuple
{
	int index;      //!< index
	qnumber qnum;   //!< quantum number
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_index_qnumber_tuple(const void* a, const void* b)
{
	const struct index_qnumber_tuple* x = (const struct index_qnumber_tuple*)a;
	const struct index_qnumber_tuple* y = (const struct index_qnumber_tuple*)b;

	if (x->index < y->index) {
		return -1;
	}
	if (x->index > y->index) {
		return 1;
	}
	// x->index == y->index
	// sort quantum numbers in descending order
	if (x->qnum > y->qnum) {
		return -1;
	}
	if (x->qnum < y->qnum) {
		return 1;
	}
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a molecular Hamiltonian as MPO,
/// using physicists' convention for the interaction term (note ordering of k and l):
/// \f[
/// H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
/// \f]
///
/// This version optimizes the virtual bond dimensions via the automatic construction starting from operator chains.
/// Can handle zero entries in 'tkin' and 'vint', but construction takes considerably longer for larger number of orbitals.
///
void construct_molecular_hamiltonian_mpo_opt(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo* mpo)
{
	assert(tkin->dtype == DOUBLE_REAL);
	assert(vint->dtype == DOUBLE_REAL);

	// dimension consistency checks
	assert(tkin->ndim == 2);
	assert(vint->ndim == 4);
	assert(tkin->dim[0] == tkin->dim[1]);
	assert(vint->dim[0] == vint->dim[1] &&
	       vint->dim[0] == vint->dim[2] &&
	       vint->dim[0] == vint->dim[3]);
	assert(tkin->dim[0] == vint->dim[0]);

	// number of "sites" (orbitals)
	const int nsites = tkin->dim[0];
	assert(nsites >= 1);

	// local operators
	// creation and annihilation operators for a single spin and lattice site
	const double a_ann[4] = { 0.,  1.,  0.,  0. };
	const double a_dag[4] = { 0.,  0.,  1.,  0. };
	// number operator
	const double numop[4] = { 0.,  0.,  0.,  1. };
	// Pauli-Z matrix required for Jordan-Wigner transformation
	const double z[4]     = { 1.,  0.,  0., -1. };

	// operator map
	// 0: I
	// 1: a^{\dagger}
	// 2: a
	// 3: numop
	// 4: Z
	struct dense_tensor opmap[5];
	for (int i = 0; i < 5; i++) {
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
	}
	dense_tensor_set_identity(&opmap[0]);
	memcpy(opmap[1].data, a_dag, sizeof(a_dag));
	memcpy(opmap[2].data, a_ann, sizeof(a_ann));
	memcpy(opmap[3].data, numop, sizeof(numop));
	memcpy(opmap[4].data, z,     sizeof(z));

	const int OID_IDENT = 0;
	const int OID_A_DAG = 1;
	const int OID_A_ANN = 2;
	const int OID_NUMOP = 3;
	const int OID_Z     = 4;

	// interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
	// can anti-commute fermionic operators such that i < j and l < k
	struct dense_tensor gint;
	symmetrize_interaction_coefficients(vint, &gint);
	// global minus sign from Jordan-Wigner transformation, since a Z = -a
	const double neg05 = -0.5;
	scale_dense_tensor(&neg05, &gint);
	assert(gint.dtype == DOUBLE_REAL);

	const int nchains = nsites * nsites + nsites * (nsites - 1) * nsites * (nsites - 1) / 4;
	struct op_chain* opchains = aligned_alloc(MEM_DATA_ALIGN, nchains * sizeof(struct op_chain));
	int oc = 0;
	// kinetic hopping terms t_{i,j} a^{\dagger}_i a_j
	const double* tkin_data = tkin->data;
	for (int i = 0; i < nsites; i++)
	{
		// case i < j
		for (int j = i + 1; j < nsites; j++)
		{
			allocate_op_chain(j - i + 1, &opchains[oc]);
			opchains[oc].oids[0] = OID_A_DAG;
			for (int n = 1; n < j - i; n++) {
				opchains[oc].oids[n] = OID_Z;
			}
			opchains[oc].oids[j - i] = OID_A_ANN;
			opchains[oc].qnums[0] = 0;
			for (int n = 1; n < j - i + 1; n++) {
				opchains[oc].qnums[n] = 1;
			}
			opchains[oc].qnums[j - i + 1] = 0;
			opchains[oc].coeff  = tkin_data[i*nsites + j];
			opchains[oc].istart = i;
			oc++;
		}
		// diagonal hopping term
		{
			allocate_op_chain(1, &opchains[oc]);
			opchains[oc].oids[0]  = OID_NUMOP;
			opchains[oc].qnums[0] = 0;
			opchains[oc].qnums[1] = 0;
			opchains[oc].coeff    = tkin_data[i*nsites + i];
			opchains[oc].istart   = i;
			oc++;
		}
		// case i > j
		for (int j = 0; j < i; j++)
		{
			allocate_op_chain(i - j + 1, &opchains[oc]);
			opchains[oc].oids[0] = OID_A_ANN;
			for (int n = 1; n < i - j; n++) {
				opchains[oc].oids[n] = OID_Z;
			}
			opchains[oc].oids[i - j] = OID_A_DAG;
			opchains[oc].qnums[0] = 0;
			for (int n = 1; n < i - j + 1; n++) {
				opchains[oc].qnums[n] = -1;
			}
			opchains[oc].qnums[i - j + 1] = 0;
			opchains[oc].coeff  = tkin_data[i*nsites + j];
			opchains[oc].istart = j;
			oc++;
		}
	}
	// interaction terms
	const double* gint_data = gint.data;
	for (int i = 0; i < nsites; i++)
	{
		for (int j = i + 1; j < nsites; j++)  // i < j
		{
			for (int k = 0; k < nsites; k++)
			{
				for (int l = 0; l < k; l++)  // l < k
				{
					struct index_qnumber_tuple tuples[4] = {
						{ .index = i, .qnum =  1 },
						{ .index = j, .qnum =  1 },
						{ .index = l, .qnum = -1 },
						{ .index = k, .qnum = -1 },
					};
					// sort by site index
					qsort(tuples, 4, sizeof(struct index_qnumber_tuple), compare_index_qnumber_tuple);

					const int a  = tuples[0].index;
					const int ba = tuples[1].index - a;
					const int ca = tuples[2].index - a;
					const int da = tuples[3].index - a;
					const qnumber p = tuples[0].qnum;
					const qnumber q = tuples[1].qnum;
					const qnumber r = tuples[2].qnum;
					const qnumber s = tuples[3].qnum;

					allocate_op_chain(da + 1, &opchains[oc]);

					if (ba == 0)  // a == b
					{
						assert(ca > 0);
						if (ca == da)
						{
							// two number operators
							// operator IDs
							opchains[oc].oids[0] = OID_NUMOP;
							for (int n = 1; n < da; n++) {
								opchains[oc].oids[n] = OID_IDENT;
							}
							opchains[oc].oids[da] = OID_NUMOP;
							// all quantum numbers are zero
							for (int n = 0; n < da + 2; n++) {
								opchains[oc].qnums[n] = 0;
							}
						}
						else
						{
							// number operator at the beginning
							// operator IDs
							opchains[oc].oids[0] = OID_NUMOP;
							for (int n = 1; n < ca; n++) {
								opchains[oc].oids[n] = OID_IDENT;
							}
							opchains[oc].oids[ca] = (r == 1 ? OID_A_DAG : OID_A_ANN);
							for (int n = ca + 1; n < da; n++) {
								opchains[oc].oids[n] = OID_Z;
							}
							opchains[oc].oids[da] = (s == 1 ? OID_A_DAG : OID_A_ANN);
							// quantum numbers
							for (int n = 0; n < ca + 1; n++) {
								opchains[oc].qnums[n] = 0;
							}
							for (int n = ca + 1; n < da + 1; n++) {
								opchains[oc].qnums[n] = r;
							}
							opchains[oc].qnums[da + 1] = 0;
						}
					}
					else if (ba == ca)
					{
						// number operator in the middle
						// operator IDs
						opchains[oc].oids[0] = (p == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = 1; n < ba; n++) {
							opchains[oc].oids[n] = OID_Z;
						}
						opchains[oc].oids[ba] = OID_NUMOP;
						for (int n = ba + 1; n < da; n++) {
							opchains[oc].oids[n] = OID_Z;
						}
						opchains[oc].oids[da] = (s == 1 ? OID_A_DAG : OID_A_ANN);
						// quantum numbers
						opchains[oc].qnums[0] = 0;
						for (int n = 1; n < da + 1; n++) {
							opchains[oc].qnums[n] = p;
						}
						opchains[oc].qnums[da + 1] = 0;
					}
					else if (ca == da)
					{
						// number operator at the end
						// operator IDs
						opchains[oc].oids[0] = (p == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = 1; n < ba; n++) {
						 	opchains[oc].oids[n] = OID_Z;
						}
						opchains[oc].oids[ba] = (q == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = ba + 1; n < ca; n++) {
						 	opchains[oc].oids[n] = OID_IDENT;
						}
						opchains[oc].oids[ca] = OID_NUMOP;
						// quantum numbers
						opchains[oc].qnums[0] = 0;
						for (int n = 1; n < ba + 1; n++) {
							opchains[oc].qnums[n] = p;
						}
						for (int n = ba + 1; n < ca + 2; n++) {
							opchains[oc].qnums[n] = 0;
						}
					}
					else
					{
						// generic case: i, j, k, l pairwise different
						// operator IDs
						opchains[oc].oids[0] = (p == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = 1; n < ba; n++) {
							opchains[oc].oids[n] = OID_Z;
						}
						opchains[oc].oids[ba] = (q == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = ba + 1; n < ca; n++) {
							opchains[oc].oids[n] = OID_IDENT;
						}
						opchains[oc].oids[ca] = (r == 1 ? OID_A_DAG : OID_A_ANN);
						for (int n = ca + 1; n < da; n++) {
							opchains[oc].oids[n] = OID_Z;
						}
						opchains[oc].oids[da] = (s == 1 ? OID_A_DAG : OID_A_ANN);
						// quantum numbers
						opchains[oc].qnums[0] = 0;
						for (int n = 1; n < ba + 1; n++) {
							opchains[oc].qnums[n] = p;
						}
						for (int n = ba + 1; n < ca + 1; n++) {
							opchains[oc].qnums[n] = p + q;
						}
						for (int n = ca + 1; n < da + 1; n++) {
							opchains[oc].qnums[n] = -s;
						}
						opchains[oc].qnums[da + 1] = 0;
					}

					opchains[oc].coeff  = gint_data[((i*nsites + j)*nsites + k)*nsites + l];
					opchains[oc].istart = a;
					oc++;
				}
			}
		}
	}
	assert(oc == nchains);

	struct mpo_graph graph;
	mpo_graph_from_opchains(opchains, nchains, nsites, OID_IDENT, &graph);

	const qnumber qsite[2] = { 0, 1 };
	mpo_from_graph(DOUBLE_REAL, 2, qsite, &graph, opmap, mpo);

	// clean up
	delete_mpo_graph(&graph);
	for (int i = 0; i < nchains; i++) {
		delete_op_chain(&opchains[i]);
	}
	aligned_free(opchains);
	delete_dense_tensor(&gint);
	for (int i = 0; i < 5; i++) {
		delete_dense_tensor(&opmap[i]);
	}
}
