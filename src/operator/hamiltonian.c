/// \file hamiltonian.c
/// \brief Construction of common quantum Hamiltonians.

#include <math.h>
#include <assert.h>
#include "hamiltonian.h"
#include "mpo_graph.h"
#include "linked_list.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


//________________________________________________________________________________________________________________________
///
/// \brief Construct a Hamiltonian as MPO operator graph based on local operator chains, which are shifted along a 1D lattice.
///
static void local_opchains_to_mpo_graph(const int nsites, const struct op_chain* lopchains, const int nlopchains, struct mpo_graph* graph)
{
	int nchains = 0;
	for (int j = 0; j < nlopchains; j++)
	{
		assert(lopchains[j].length <= nsites);
		nchains += (nsites - lopchains[j].length + 1);
	}
	struct op_chain* opchains = ct_malloc(nchains * sizeof(struct op_chain));
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

	mpo_graph_from_opchains(opchains, nchains, nsites, graph);

	ct_free(opchains);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contruct an MPO assembly representation of the Ising Hamiltonian 'sum J Z Z + h Z + g X' on a one-dimensional lattice.
///
void construct_ising_1d_mpo_assembly(const int nsites, const double J, const double h, const double g, struct mpo_assembly* assembly)
{
	assert(nsites >= 2);

	// set physical quantum numbers to zero
	assembly->d = 2;
	assembly->qsite = ct_calloc(assembly->d, sizeof(qnumber));

	assembly->dtype = CT_DOUBLE_REAL;

	// operator map
	const int OID_I = 0;  // identity
	const int OID_Z = 1;  // Pauli-Z
	const int OID_X = 2;  // Pauli-X
	assembly->num_local_ops = 3;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < assembly->num_local_ops; i++) {
		const long dim[2] = { assembly->d, assembly->d };
		allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
	}
	const double sz[4] = { 1., 0., 0., -1. };  // Z
	const double sx[4] = { 0., 1., 1.,  0. };  // X
	dense_tensor_set_identity(&assembly->opmap[OID_I]);
	memcpy(assembly->opmap[OID_Z].data, sz, sizeof(sz));
	memcpy(assembly->opmap[OID_X].data, sx, sizeof(sx));

	// coefficient map; first two entries must always be 0 and 1
	const double coeffmap[] = { 0, 1, J, h, g };
	assembly->num_coeffs = ARRLEN(coeffmap);
	assembly->coeffmap = ct_malloc(sizeof(coeffmap));
	memcpy(assembly->coeffmap, coeffmap, sizeof(coeffmap));

	// local two-site and single-site terms
	int oids_c0[] = { OID_Z, OID_Z };  qnumber qnums_c0[] = { 0, 0, 0 };
	int oids_c1[] = { OID_Z };         qnumber qnums_c1[] = { 0, 0 };
	int oids_c2[] = { OID_X };         qnumber qnums_c2[] = { 0, 0 };
	struct op_chain lopchains[] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .cid = 2, .length = ARRLEN(oids_c0), .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .cid = 3, .length = ARRLEN(oids_c1), .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .cid = 4, .length = ARRLEN(oids_c2), .istart = 0 },
	};

	// convert to an MPO graph
	local_opchains_to_mpo_graph(nsites, lopchains, ARRLEN(lopchains), &assembly->graph);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO assembly representation of the XXZ Heisenberg Hamiltonian 'sum J (X X + Y Y + D Z Z) - h Z' on a one-dimensional lattice.
///
void construct_heisenberg_xxz_1d_mpo_assembly(const int nsites, const double J, const double D, const double h, struct mpo_assembly* assembly)
{
	assert(nsites >= 2);

	// physical quantum numbers (multiplied by 2)
	assembly->d = 2;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	assembly->qsite[0] =  1;
	assembly->qsite[1] = -1;

	assembly->dtype = CT_DOUBLE_REAL;

	// spin operators
	const double sup[4] = { 0.,  1.,  0.,  0.  };  // S_up
	const double sdn[4] = { 0.,  0.,  1.,  0.  };  // S_down
	const double  sz[4] = { 0.5, 0.,  0., -0.5 };  // S_z

	// operator map
	const int OID_Id = 0;  // I
	const int OID_Su = 1;  // S_up
	const int OID_Sd = 2;  // S_down
	const int OID_Sz = 3;  // S_z
	assembly->num_local_ops = 4;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < assembly->num_local_ops; i++) {
		const long dim[2] = { assembly->d, assembly->d };
		allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
	}
	dense_tensor_set_identity(&assembly->opmap[OID_Id]);
	memcpy(assembly->opmap[OID_Su].data, sup, sizeof(sup));
	memcpy(assembly->opmap[OID_Sd].data, sdn, sizeof(sdn));
	memcpy(assembly->opmap[OID_Sz].data, sz,  sizeof(sz));

	// coefficient map; first two entries must always be 0 and 1
	const double coeffmap[] = { 0, 1, 0.5*J, J*D, -h };
	assembly->num_coeffs = ARRLEN(coeffmap);
	assembly->coeffmap = ct_malloc(sizeof(coeffmap));
	memcpy(assembly->coeffmap, coeffmap, sizeof(coeffmap));

	// local two-site and single-site terms
	int oids_c0[] = { OID_Su, OID_Sd };  qnumber qnums_c0[] = { 0,  2,  0 };
	int oids_c1[] = { OID_Sd, OID_Su };  qnumber qnums_c1[] = { 0, -2,  0 };
	int oids_c2[] = { OID_Sz, OID_Sz };  qnumber qnums_c2[] = { 0,  0,  0 };
	int oids_c3[] = { OID_Sz };          qnumber qnums_c3[] = { 0,  0 };
	struct op_chain lopchains[] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .cid = 2, .length = ARRLEN(oids_c0), .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .cid = 2, .length = ARRLEN(oids_c1), .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .cid = 3, .length = ARRLEN(oids_c2), .istart = 0 },
		{ .oids = oids_c3, .qnums = qnums_c3, .cid = 4, .length = ARRLEN(oids_c3), .istart = 0 },
	};

	// convert to an MPO graph
	local_opchains_to_mpo_graph(nsites, lopchains, ARRLEN(lopchains), &assembly->graph);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO assembly representation of the Bose-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
void construct_bose_hubbard_1d_mpo_assembly(const int nsites, const long d, const double t, const double u, const double mu, struct mpo_assembly* assembly)
{
	assert(nsites >= 2);
	assert(d >= 1);

	// physical quantum numbers (particle number)
	assembly->d = d;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	for (long i = 0; i < d; i++) {
		assembly->qsite[i] = i;
	}

	assembly->dtype = CT_DOUBLE_REAL;

	// operator map
	const int OID_Id = 0;  // identity
	const int OID_Bd = 1;  // b_{\dagger}
	const int OID_B  = 2;  // b
	const int OID_N  = 3;  // n
	const int OID_NI = 4;  // n (n - 1) / 2
	assembly->num_local_ops = 5;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < assembly->num_local_ops; i++) {
		const long dim[2] = { d, d };
		allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
	}
	// identity operator
	dense_tensor_set_identity(&assembly->opmap[OID_Id]);
	// bosonic creation operator
	double* b_dag = assembly->opmap[OID_Bd].data;
	for (long i = 0; i < d - 1; i++) {
		b_dag[(i + 1)*d + i] = sqrt(i + 1);
	}
	// bosonic annihilation operator
	double* b_ann = assembly->opmap[OID_B].data;
	for (long i = 0; i < d - 1; i++) {
		b_ann[i*d + (i + 1)] = sqrt(i + 1);
	}
	// bosonic number operator
	double* numop = assembly->opmap[OID_N].data;
	for (long i = 0; i < d; i++) {
		numop[i*d + i] = i;
	}
	// bosonic local interaction operator n (n - 1) / 2
	double* v_int = assembly->opmap[OID_NI].data;
	for (long i = 0; i < d; i++) {
		v_int[i*d + i] = i * (i - 1) / 2;
	}

	// coefficient map; first two entries must always be 0 and 1
	const double coeffmap[] = { 0, 1, -t, -mu, u };
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

	// convert to an MPO graph
	local_opchains_to_mpo_graph(nsites, lopchains, ARRLEN(lopchains), &assembly->graph);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO assembly representation of the Fermi-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
/// States for each spin and site are '|0>' and '|1>'.
///
void construct_fermi_hubbard_1d_mpo_assembly(const int nsites, const double t, const double u, const double mu, struct mpo_assembly* assembly)
{
	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = { 0,  1,  1,  2 };
	const qnumber qs[4] = { 0, -1,  1,  0 };
	assembly->d = 4;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	for (long i = 0; i < assembly->d; i++) {
		assembly->qsite[i] = encode_quantum_number_pair(qn[i], qs[i]);
	}

	assembly->dtype = CT_DOUBLE_REAL;

	struct dense_tensor id;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &id);
		dense_tensor_set_identity(&id);
	}
	// creation and annihilation operators for a single spin and lattice site
	struct dense_tensor a_dag;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &a_dag);
		const double data[4] = { 0., 0., 1., 0. };
		memcpy(a_dag.data, data, sizeof(data));
	}
	struct dense_tensor a_ann;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &a_ann);
		const double data[4] = { 0., 1., 0., 0. };
		memcpy(a_ann.data, data, sizeof(data));
	}
	// number operator
	struct dense_tensor numop;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &numop);
		const double data[4] = { 0., 0., 0., 1. };
		memcpy(numop.data, data, sizeof(data));
	}
	// Pauli-Z matrix required for Jordan-Wigner transformation
	struct dense_tensor z;
	{
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &z);
		const double data[4] = { 1.,  0.,  0., -1. };
		memcpy(z.data, data, sizeof(data));
	}
	// total number operator n_up + n_dn
	struct dense_tensor n_tot;
	{
		const long dim[2] = { 4, 4 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &n_tot);
		const double diag[4] = { 0, 1, 1, 2 };
		double* data = n_tot.data;
		for (int i = 0; i < 4; i++) {
			data[i*4 + i] = diag[i];
		}
	}
	// local interaction term (n_up - 1/2) (n_dn - 1/2)
	struct dense_tensor n_int;
	{
		const long dim[2] = { 4, 4 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &n_int);
		const double diag[4] = {  0.25, -0.25, -0.25,  0.25 };
		double* data = n_int.data;
		for (int i = 0; i < 4; i++) {
			data[i*4 + i] = diag[i];
		}
	}

	// operator map
	const int OID_Id =  0;  //  I x I
	const int OID_CI =  1;  // ad x I
	const int OID_AI =  2;  //  a x I
	const int OID_CZ =  3;  // ad x Z
	const int OID_AZ =  4;  //  a x Z
	const int OID_IC =  5;  //  I x ad
	const int OID_IA =  6;  //  I x a
	const int OID_ZC =  7;  //  Z x ad
	const int OID_ZA =  8;  //  Z x a
	const int OID_Nt =  9;  //  n_up + n_dn
	const int OID_NI = 10;  // (n_up - 1/2) (n_dn - 1/2)
	assembly->num_local_ops = 11;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	dense_tensor_kronecker_product(&id,    &id,    &assembly->opmap[OID_Id]);
	dense_tensor_kronecker_product(&a_dag, &id,    &assembly->opmap[OID_CI]);
	dense_tensor_kronecker_product(&a_ann, &id,    &assembly->opmap[OID_AI]);
	dense_tensor_kronecker_product(&a_dag, &z,     &assembly->opmap[OID_CZ]);
	dense_tensor_kronecker_product(&a_ann, &z,     &assembly->opmap[OID_AZ]);
	dense_tensor_kronecker_product(&id,    &a_dag, &assembly->opmap[OID_IC]);
	dense_tensor_kronecker_product(&id,    &a_ann, &assembly->opmap[OID_IA]);
	dense_tensor_kronecker_product(&z,     &a_dag, &assembly->opmap[OID_ZC]);
	dense_tensor_kronecker_product(&z,     &a_ann, &assembly->opmap[OID_ZA]);
	copy_dense_tensor(&n_tot, &assembly->opmap[OID_Nt]);
	copy_dense_tensor(&n_int, &assembly->opmap[OID_NI]);

	// coefficient map; first two entries must always be 0 and 1
	const double coeffmap[] = { 0, 1, -t, -mu, u };
	assembly->num_coeffs = ARRLEN(coeffmap);
	assembly->coeffmap = ct_malloc(sizeof(coeffmap));
	memcpy(assembly->coeffmap, coeffmap, sizeof(coeffmap));

	// local two-site and single-site terms
	// spin-up kinetic hopping
	int oids_c0[] = { OID_CZ, OID_AI };  qnumber qnums_c0[] = { 0, encode_quantum_number_pair( 1,  1), 0 };
	int oids_c1[] = { OID_AZ, OID_CI };  qnumber qnums_c1[] = { 0, encode_quantum_number_pair(-1, -1), 0 };
	// spin-down kinetic hopping
	int oids_c2[] = { OID_IC, OID_ZA };  qnumber qnums_c2[] = { 0, encode_quantum_number_pair( 1, -1), 0 };
	int oids_c3[] = { OID_IA, OID_ZC };  qnumber qnums_c3[] = { 0, encode_quantum_number_pair(-1,  1), 0 };
	// number operator - mu (n_up + n_dn) and interaction u (n_up-1/2) (n_dn-1/2)
	int oids_c4[] = { OID_Nt };          qnumber qnums_c4[] = { 0, 0 };
	int oids_c5[] = { OID_NI };          qnumber qnums_c5[] = { 0, 0 };
	struct op_chain lopchains[] = {
		{ .oids = oids_c0, .qnums = qnums_c0, .cid = 2, .length = ARRLEN(oids_c0), .istart = 0 },
		{ .oids = oids_c1, .qnums = qnums_c1, .cid = 2, .length = ARRLEN(oids_c1), .istart = 0 },
		{ .oids = oids_c2, .qnums = qnums_c2, .cid = 2, .length = ARRLEN(oids_c2), .istart = 0 },
		{ .oids = oids_c3, .qnums = qnums_c3, .cid = 2, .length = ARRLEN(oids_c3), .istart = 0 },
		{ .oids = oids_c4, .qnums = qnums_c4, .cid = 3, .length = ARRLEN(oids_c4), .istart = 0 },
		{ .oids = oids_c5, .qnums = qnums_c5, .cid = 4, .length = ARRLEN(oids_c5), .istart = 0 },
	};

	// convert to an MPO graph
	local_opchains_to_mpo_graph(nsites, lopchains, ARRLEN(lopchains), &assembly->graph);

	// clean up
	delete_dense_tensor(&n_int);
	delete_dense_tensor(&n_tot);
	delete_dense_tensor(&z);
	delete_dense_tensor(&numop);
	delete_dense_tensor(&a_ann);
	delete_dense_tensor(&a_dag);
	delete_dense_tensor(&id);
}


//________________________________________________________________________________________________________________________
///
/// \brief Local operator IDs for a molecular Hamiltonian.
///
enum molecular_oid
{
	MOLECULAR_OID_I   = 0,  //!< identity
	MOLECULAR_OID_C   = 1,  //!< \f$a^{\dagger}\f$
	MOLECULAR_OID_A   = 2,  //!< \f$a\f$
	MOLECULAR_OID_N   = 3,  //!< \f$n\f$
	MOLECULAR_OID_Z   = 4,  //!< \f$Z\f$
	NUM_MOLECULAR_OID = 5,  //!< number of local operators
};


//________________________________________________________________________________________________________________________
///
/// \brief Create the local operator map for a molecular Hamiltonian.
///
static void create_molecular_hamiltonian_operator_map(struct dense_tensor* opmap)
{
	// local operators
	// creation and annihilation operators for a single spin and lattice site
	const double a_ann[4] = { 0.,  1.,  0.,  0. };
	const double a_dag[4] = { 0.,  0.,  1.,  0. };
	// number operator
	const double numop[4] = { 0.,  0.,  0.,  1. };
	// Pauli-Z matrix required for Jordan-Wigner transformation
	const double z[4]     = { 1.,  0.,  0., -1. };

	for (int i = 0; i < NUM_MOLECULAR_OID; i++) {
		const long dim[2] = { 2, 2 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &opmap[i]);
	}

	dense_tensor_set_identity(&opmap[MOLECULAR_OID_I]);
	memcpy(opmap[MOLECULAR_OID_C].data, a_dag, sizeof(a_dag));
	memcpy(opmap[MOLECULAR_OID_A].data, a_ann, sizeof(a_ann));
	memcpy(opmap[MOLECULAR_OID_N].data, numop, sizeof(numop));
	memcpy(opmap[MOLECULAR_OID_Z].data, z,     sizeof(z));
}


//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize the interaction coefficient tensor as
/// vint - transpose(vint, (1, 0, 2, 3)) - transpose(vint, (0, 1, 3, 2)) + transpose(vint, (1, 0, 3, 2))
///
static inline void symmetrize_molecular_interaction_coefficients(const struct dense_tensor* restrict vint, struct dense_tensor* restrict gint)
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
/// \brief Allocate and initialize MPO graph vertex indices for each virtual bond (utility function).
///
static int* allocate_vertex_ids(const int nsites)
{
	int* vids = ct_malloc((nsites + 1) * sizeof(int));
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
static struct mpo_graph_edge* construct_mpo_graph_edge(const int vid0, const int vid1, const int oid, const int cid)
{
	struct mpo_graph_edge* edge = ct_malloc(sizeof(struct mpo_graph_edge));
	edge->vids[0] = vid0;
	edge->vids[1] = vid1;
	edge->opics = ct_malloc(sizeof(struct local_op_ref));
	edge->opics[0].oid = oid;
	edge->opics[0].cid = cid;
	edge->nopics = 1;

	return edge;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate and initialize an MPO graph edge with a local operator reference and several coefficient indices.
///
static struct mpo_graph_edge* construct_mpo_graph_edge_multicids(const int vid0, const int vid1, const int oid, const int* cids, const int numcids)
{
	assert(numcids >= 1);

	struct mpo_graph_edge* edge = ct_malloc(sizeof(struct mpo_graph_edge));
	edge->vids[0] = vid0;
	edge->vids[1] = vid1;
	edge->opics = ct_malloc(numcids * sizeof(struct local_op_ref));
	for (int i = 0; i < numcids; i++) {
		edge->opics[i].oid = oid;
		edge->opics[i].cid = cids[i];
	}
	edge->nopics = numcids;

	return edge;
}


//________________________________________________________________________________________________________________________
///
/// \brief Vertex connection of creation and annihilation chains for a molecular Hamiltonian.
///
enum molecular_vertex_connection_direction
{
	MOLECULAR_CONN_LEFT  = 0,  //!< connected to the left boundary
	MOLECULAR_CONN_RIGHT = 1,  //!< connected to the right boundary
};


//________________________________________________________________________________________________________________________
///
/// \brief Vertex IDs in an operator graph used for molecular Hamiltonian construction.
///
/// The two array dimensions specify the connection to the left and right boundary, respectively.
///
struct molecular_mpo_graph_vids
{
	int*  identity[2];     //!< identity chains
	int** a_dag[2];        //!< \f$a^{\dagger}_i\f$
	int** a_ann[2];        //!< \f$a_i\f$
	int** a_dag_a_dag[2];  //!< \f$a^{\dagger}_i a^{\dagger}_j\f$
	int** a_ann_a_ann[2];  //!< \f$a_i a_j\f$
	int** a_dag_a_ann[2];  //!< \f$a^{\dagger}_i a_j\f$
	int   nsites;          //!< number of sites (orbitals)
};


//________________________________________________________________________________________________________________________
///
/// \brief Create the vertices in an operator graph used for molecular Hamiltonian construction.
///
static void create_molecular_mpo_graph_vertices(const int nsites, struct mpo_graph* graph, struct molecular_mpo_graph_vids* vids)
{
	graph->verts     = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	graph->num_verts = ct_calloc(nsites + 1, sizeof(int));
	graph->nsites    = nsites;

	for (int i = 0; i < nsites + 1; i++)
	{
		// number of vertices, i.e., virtual bond dimensions
		const int nl = i;
		const int nr = nsites - i;
		const int n = imin(nl, nr);
		// identity chains
		int chi1 = (1 <= i && i <= nsites - 1) ? 2 : 1;
		// a^{\dagger}_i and a_i chains, reaching (almost) from one boundary to the other
		int chi2 = 2 * ((i < nsites - 1 ? nl : 0) + (i > 1 ? nr : 0));
		// a^{\dagger}_i a^{\dagger}_j (for i < j), a_i a_j (for i > j) and a^{\dagger}_i a_j chains, extending from boundary to center
		int chi3 = n * (n - 1) + n * n;

		graph->num_verts[i] = chi1 + chi2 + chi3;
		graph->verts[i] = ct_calloc(graph->num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	vids->nsites = nsites;

	int* vertex_counter = ct_calloc(nsites + 1, sizeof(int));

	// vertex IDs

	// identity chains from the left and right
	vids->identity[MOLECULAR_CONN_LEFT]  = allocate_vertex_ids(nsites);
	vids->identity[MOLECULAR_CONN_RIGHT] = allocate_vertex_ids(nsites);
	for (int i = 0; i < nsites; i++) {
		vids->identity[MOLECULAR_CONN_LEFT][i] = vertex_counter[i]++;
	}
	for (int i = 1; i < nsites + 1; i++) {
		vids->identity[MOLECULAR_CONN_RIGHT][i] = vertex_counter[i]++;
	}

	// a^{\dagger}_i operators connected to left boundary
	vids->a_dag[MOLECULAR_CONN_LEFT] = ct_calloc(nsites, sizeof(int*));
	for (int i = 0; i < nsites - 2; i++) {
		vids->a_dag[MOLECULAR_CONN_LEFT][i] = allocate_vertex_ids(nsites);
		for (int j = i + 1; j < nsites - 1; j++) {
			const int vid = vertex_counter[j]++;
			vids->a_dag[MOLECULAR_CONN_LEFT][i][j] = vid;
			graph->verts[j][vid].qnum = 1;
		}
	}

	// a_i operators connected to left boundary
	vids->a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(nsites, sizeof(int*));
	for (int i = 0; i < nsites - 2; i++) {
		vids->a_ann[MOLECULAR_CONN_LEFT][i] = allocate_vertex_ids(nsites);
		for (int j = i + 1; j < nsites - 1; j++) {
			const int vid = vertex_counter[j]++;
			vids->a_ann[MOLECULAR_CONN_LEFT][i][j] = vid;
			graph->verts[j][vid].qnum = -1;
		}
	}

	// a^{\dagger}_i a^{\dagger}_j operators connected to left boundary, for i < j
	vids->a_dag_a_dag[MOLECULAR_CONN_LEFT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = j + 1; k < nsites/2 + 1; k++) {
				const int vid = vertex_counter[k]++;
				vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j][k] = vid;
				graph->verts[k][vid].qnum = 2;
			}
		}
	}

	// a_i a_j operators connected to left boundary, for i > j
	vids->a_ann_a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < i; j++) {
			vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = i + 1; k < nsites/2 + 1; k++) {
				const int vid = vertex_counter[k]++;
				vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k] = vid;
				graph->verts[k][vid].qnum = -2;
			}
		}
	}

	// a^{\dagger}_i a_j operators connected to left boundary
	vids->a_dag_a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = imax(i, j) + 1; k < nsites/2 + 1; k++) {
				vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k] = vertex_counter[k]++;
				// vertex quantum number is zero
			}
		}
	}

	// a^{\dagger}_i operators connected to right boundary
	vids->a_dag[MOLECULAR_CONN_RIGHT] = ct_calloc(nsites, sizeof(int*));
	for (int i = 2; i < nsites; i++) {
		vids->a_dag[MOLECULAR_CONN_RIGHT][i] = allocate_vertex_ids(nsites);
		for (int j = 2; j < i + 1; j++) {
			const int vid = vertex_counter[j]++;
			vids->a_dag[MOLECULAR_CONN_RIGHT][i][j] = vid;
			graph->verts[j][vid].qnum = -1;
		}
	}

	// a_i operators connected to right boundary
	vids->a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(nsites, sizeof(int*));
	for (int i = 2; i < nsites; i++) {
		vids->a_ann[MOLECULAR_CONN_RIGHT][i] = allocate_vertex_ids(nsites);
		for (int j = 2; j < i + 1; j++) {
			const int vid = vertex_counter[j]++;
			vids->a_ann[MOLECULAR_CONN_RIGHT][i][j] = vid;
			graph->verts[j][vid].qnum = 1;
		}
	}

	// a^{\dagger}_i a^{\dagger}_j operators connected to right boundary, for i < j
	vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < i + 1; k++) {
				const int vid = vertex_counter[k]++;
				vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j][k] = vid;
				graph->verts[k][vid].qnum = -2;
			}
		}
	}

	// a_i a_j operators connected to right boundary, for i > j
	vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < i; j++) {
			vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < j + 1; k++) {
				const int vid = vertex_counter[k]++;
				vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k] = vid;
				graph->verts[k][vid].qnum = 2;
			}
		}
	}

	// a^{\dagger}_i a_j operators connected to right boundary
	vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j] = allocate_vertex_ids(nsites);
			for (int k = nsites/2 + 1; k < imin(i, j) + 1; k++) {
				vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k] = vertex_counter[k]++;
				// vertex quantum number is zero
			}
		}
	}

	// consistency check
	for (int i = 0; i < nsites + 1; i++) {
		assert(vertex_counter[i] == graph->num_verts[i]);
	}

	ct_free(vertex_counter);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the vertices in an operator graph used for molecular Hamiltonian construction.
///
static void delete_molecular_mpo_graph_vertices(struct molecular_mpo_graph_vids* vids)
{
	const int nsites = vids->nsites;

	ct_free(vids->identity[MOLECULAR_CONN_LEFT]);
	ct_free(vids->identity[MOLECULAR_CONN_RIGHT]);

	for (int i = 0; i < nsites - 2; i++) {
		ct_free(vids->a_dag[MOLECULAR_CONN_LEFT][i]);
		ct_free(vids->a_ann[MOLECULAR_CONN_LEFT][i]);
	}
	ct_free(vids->a_dag[MOLECULAR_CONN_LEFT]);
	ct_free(vids->a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j]);
		}
	}
	ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < i; j++) {
			ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j]);
		}
	}
	ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j]);
		}
	}
	ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 2; i < nsites; i++) {
		ct_free(vids->a_dag[MOLECULAR_CONN_RIGHT][i]);
		ct_free(vids->a_ann[MOLECULAR_CONN_RIGHT][i]);
	}
	ct_free(vids->a_dag[MOLECULAR_CONN_RIGHT]);
	ct_free(vids->a_ann[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j]);
		}
	}
	ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < i; j++) {
			ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j]);
		}
	}
	ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j]);
		}
	}
	ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Connect the vertices of the creation and annihilation operator strings
/// in the operator graph describing a molecular Hamiltonian.
///
static void molecular_mpo_graph_connect_operator_strings(const struct molecular_mpo_graph_vids* vids, struct linked_list* edges)
{
	const int nsites = vids->nsites;

	// identities connected to left and right boundaries
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->identity[MOLECULAR_CONN_LEFT][i + 1], MOLECULAR_OID_I, CID_ONE));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_RIGHT][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], MOLECULAR_OID_I, CID_ONE));
	}

	// a^{\dagger}_i operators connected to left boundary
	for (int i = 0; i < nsites - 2; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_dag[MOLECULAR_CONN_LEFT][i][i + 1], MOLECULAR_OID_C, CID_ONE));
		// Z operator from Jordan-Wigner transformation
		for (int j = i + 1; j < nsites - 2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][i][j], vids->a_dag[MOLECULAR_CONN_LEFT][i][j + 1], MOLECULAR_OID_Z, CID_ONE));
		}
	}

	// a_i operators connected to left boundary
	for (int i = 0; i < nsites - 2; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_ann[MOLECULAR_CONN_LEFT][i][i + 1], MOLECULAR_OID_A, CID_ONE));
		// Z operator from Jordan-Wigner transformation
		for (int j = i + 1; j < nsites - 2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][i][j], vids->a_ann[MOLECULAR_CONN_LEFT][i][j + 1], MOLECULAR_OID_Z, CID_ONE));
		}
	}

	// a^{\dagger}_i a^{\dagger}_j operators connected to left boundary
	for (int i = 0; i < nsites/2 - 1; i++) {
		for (int j = i + 1; j < nsites/2; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][i][j], vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j][j + 1], MOLECULAR_OID_C, CID_ONE));
			// identities for transition to next site
			for (int k = j + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j][k], vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
		}
	}

	// a_i a_j operators connected to left boundary
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < i; j++) {
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][j][i], vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][i + 1], MOLECULAR_OID_A, CID_ONE));
			// identities for transition to next site
			for (int k = i + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k], vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
		}
	}

	// a^{\dagger}_i a_j operators connected to left boundary
	for (int i = 0; i < nsites/2; i++) {
		for (int j = 0; j < nsites/2; j++) {
			if (i < j) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][i][j], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][j + 1], MOLECULAR_OID_A, CID_ONE));
			}
			else if (i == j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][i + 1], MOLECULAR_OID_N, CID_ONE));
			}
			else { // i > j
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][j][i], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][i + 1], MOLECULAR_OID_C, CID_ONE));
			}
			// identities for transition to next site
			for (int k = imax(i, j) + 1; k < nsites/2; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
		}
	}

	// a^{\dagger}_i operators connected to right boundary
	for (int i = 2; i < nsites; i++) {
		// Z operator from Jordan-Wigner transformation
		for (int j = 2; j < i; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_RIGHT][i][j], vids->a_dag[MOLECULAR_CONN_RIGHT][i][j + 1], MOLECULAR_OID_Z, CID_ONE));
		}
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_RIGHT][i][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], MOLECULAR_OID_C, CID_ONE));
	}

	// a_i operators connected to right boundary
	for (int i = 2; i < nsites; i++) {
		// Z operator from Jordan-Wigner transformation
		for (int j = 2; j < i; j++) {
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_RIGHT][i][j], vids->a_ann[MOLECULAR_CONN_RIGHT][i][j + 1], MOLECULAR_OID_Z, CID_ONE));
		}
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_RIGHT][i][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], MOLECULAR_OID_A, CID_ONE));
	}

	// a^{\dagger}_i a^{\dagger}_j operators connected to right boundary
	for (int i = nsites/2 + 1; i < nsites - 1; i++) {
		for (int j = i + 1; j < nsites; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < i; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j][k], vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][i*nsites + j][i], vids->a_dag[MOLECULAR_CONN_RIGHT][j][i + 1], MOLECULAR_OID_C, CID_ONE));
		}
	}

	// a_i a_j operators connected to right boundary
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < i; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < j; k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k], vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][j], vids->a_ann[MOLECULAR_CONN_RIGHT][i][j + 1], MOLECULAR_OID_A, CID_ONE));
		}
	}

	// a^{\dagger}_i a_j operators connected to right boundary
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int j = nsites/2 + 1; j < nsites; j++) {
			// identities for transition to next site
			for (int k = nsites/2 + 1; k < imin(i, j); k++) {
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k], vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][k + 1], MOLECULAR_OID_I, CID_ONE));
			}
			if (i < j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][i], vids->a_ann[MOLECULAR_CONN_RIGHT][j][i + 1], MOLECULAR_OID_C, CID_ONE));
			}
			else if (i == j) {
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], MOLECULAR_OID_N, CID_ONE));
			}
			else { // i > j
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][i*nsites + j][j], vids->a_dag[MOLECULAR_CONN_RIGHT][i][j + 1], MOLECULAR_OID_A, CID_ONE));
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Local operator description for a molecular Hamiltonian.
///
struct molecular_local_op_ref
{
	int isite;  //!< site index
	int oid;    //!< operator index
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_molecular_local_op_ref(const void* a, const void* b)
{
	const struct molecular_local_op_ref* x = (const struct molecular_local_op_ref*)a;
	const struct molecular_local_op_ref* y = (const struct molecular_local_op_ref*)b;

	if (x->isite < y->isite) {
		return -1;
	}
	if (x->isite > y->isite) {
		return 1;
	}
	// x->isite == y->isite
	if (x->oid < y->oid) {
		return -1;
	}
	if (x->oid > y->oid) {
		return 1;
	}
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the list of vertex IDs (for each bond index) corresponding to a single creation or annihilation operator.
///
static inline const int* molecular_mpo_graph_vids_get_single(const struct molecular_mpo_graph_vids* vids, const struct molecular_local_op_ref* op, const int direction)
{
	if (op->oid == MOLECULAR_OID_C) {
		return vids->a_dag[direction][op->isite];
	}
	if (op->oid == MOLECULAR_OID_A) {
		return vids->a_ann[direction][op->isite];
	}
	// operator vertices do not exist
	assert(false);
	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the list of vertex IDs (for each bond index) corresponding to a pair of creation and annihilation operators.
///
static inline const int* molecular_mpo_graph_vids_get_pair(const struct molecular_mpo_graph_vids* vids, const struct molecular_local_op_ref* op0, const struct molecular_local_op_ref* op1, const int direction)
{
	const int i = op0->isite;
	const int j = op1->isite;

	if (op0->oid == MOLECULAR_OID_C)
	{
		if (op1->oid == MOLECULAR_OID_C) {
			return vids->a_dag_a_dag[direction][i < j ? i*vids->nsites + j : j*vids->nsites + i];
		}
		if (op1->oid == MOLECULAR_OID_A) {
			return vids->a_dag_a_ann[direction][i*vids->nsites + j];
		}
	}
	if (op0->oid == MOLECULAR_OID_A)
	{
		if (op1->oid == MOLECULAR_OID_C) {
			return vids->a_dag_a_ann[direction][j*vids->nsites + i];
		}
		if (op1->oid == MOLECULAR_OID_A) {
			// reverse order
			return vids->a_ann_a_ann[direction][i < j ? j*vids->nsites + i : i*vids->nsites + j];
		}
	}
	// operator vertices do not exist
	assert(false);
	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Add an operator term (operator string of creation and annihilation operators)
/// to the operator graph describing a molecular Hamiltonian.
///
static void molecular_mpo_graph_add_term(const struct molecular_mpo_graph_vids* vids, const struct molecular_local_op_ref* oplist, const int numops, const int cid, struct linked_list* edges)
{
	const int nsites = vids->nsites;

	// sort by site (orbital) index
	struct molecular_local_op_ref* oplist_sorted = ct_malloc(numops * sizeof(struct molecular_local_op_ref));
	memcpy(oplist_sorted, oplist, numops * sizeof(struct molecular_local_op_ref));
	qsort(oplist_sorted, numops, sizeof(struct molecular_local_op_ref), compare_molecular_local_op_ref);

	if (numops == 2)
	{
		const int i = oplist_sorted[0].isite;
		const int j = oplist_sorted[1].isite;
		assert(0 <= i && i < nsites);
		assert(0 <= j && j < nsites);

		if (i == j)
		{
			// expecting number operator
			assert(oplist_sorted[0].oid == MOLECULAR_OID_C && oplist_sorted[1].oid == MOLECULAR_OID_A);
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], MOLECULAR_OID_N, cid));
		}
		else
		{
			assert(i < j);
			if (j <= nsites/2)
			{
				const int* vids_l = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids_l[j], vids->identity[MOLECULAR_CONN_RIGHT][j + 1], oplist_sorted[1].oid, cid));
			}
			else if (i >= nsites/2)
			{
				const int* vids_r = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[1], MOLECULAR_CONN_RIGHT);
				linked_list_append(&edges[i],
					construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids_r[i + 1], oplist_sorted[0].oid, cid));
			}
			else
			{
				const int* vids_l = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				const int* vids_r = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[1], MOLECULAR_CONN_RIGHT);
				linked_list_append(&edges[nsites/2],
					construct_mpo_graph_edge(vids_l[nsites/2], vids_r[nsites/2 + 1], MOLECULAR_OID_Z, cid));
			}
		}
	}
	else if (numops == 4)
	{
		const int i = oplist_sorted[0].isite;
		const int j = oplist_sorted[1].isite;
		const int k = oplist_sorted[2].isite;
		const int l = oplist_sorted[3].isite;
		assert(0 <= i && i < nsites);
		assert(0 <= j && j < nsites);
		assert(0 <= k && k < nsites);
		assert(0 <= l && l < nsites);

		if (j == k)
		{
			// expecting number operator
			assert(oplist_sorted[1].oid == MOLECULAR_OID_C && oplist_sorted[2].oid == MOLECULAR_OID_A);
			const int* vids_l = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
			const int* vids_r = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			linked_list_append(&edges[j],
				construct_mpo_graph_edge(vids_l[j], vids_r[j + 1], MOLECULAR_OID_N, cid));
		}
		else if (k <= nsites/2)
		{
			const int* vids_l = molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[0], &oplist_sorted[1], MOLECULAR_CONN_LEFT);
			if (k == l)
			{
				// expecting number operator
				assert(oplist_sorted[2].oid == MOLECULAR_OID_C && oplist_sorted[3].oid == MOLECULAR_OID_A);
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_l[k], vids->identity[MOLECULAR_CONN_RIGHT][k + 1], MOLECULAR_OID_N, cid));
			}
			else
			{
				const int* vids_r = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
				linked_list_append(&edges[k],
					construct_mpo_graph_edge(vids_l[k], vids_r[k + 1], oplist_sorted[2].oid, cid));
			}
		}
		else if (j >= nsites/2)
		{
			const int* vids_r = molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[2], &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			if (i == j)
			{
				// expecting number operator
				assert(oplist_sorted[0].oid == MOLECULAR_OID_C && oplist_sorted[1].oid == MOLECULAR_OID_A);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][j], vids_r[j + 1], MOLECULAR_OID_N, cid));
			}
			else
			{
				const int* vids_l = molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids_l[j], vids_r[j + 1], oplist_sorted[1].oid, cid));
			}
		}
		else
		{
			const int* vids_l = molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[0], &oplist_sorted[1], MOLECULAR_CONN_LEFT);
			const int* vids_r = molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[2], &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			linked_list_append(&edges[nsites/2],
				construct_mpo_graph_edge(vids_l[nsites/2], vids_r[nsites/2 + 1], MOLECULAR_OID_I, cid));
		}
	}
	else
	{
		// not implemented
		assert(false);
	}

	ct_free(oplist_sorted);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a molecular Hamiltonian as MPO assembly,
/// using physicists' convention for the interaction term (note ordering of k and l):
/// \f[
/// H = \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
/// \f]
///
/// If 'optimize == true', optimize the virtual bond dimensions via the automatic construction starting from operator chains.
/// Can handle zero entries in 'tkin' and 'vint', but construction takes considerably longer for larger number of orbitals.
///
void construct_molecular_hamiltonian_mpo_assembly(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, const bool optimize, struct mpo_assembly* assembly)
{
	assert(tkin->dtype == CT_DOUBLE_REAL);
	assert(vint->dtype == CT_DOUBLE_REAL);

	assembly->dtype = CT_DOUBLE_REAL;

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
	const int nsites2 = nsites * nsites;
	const int nsites_choose_two = nsites * (nsites - 1) / 2;

	// physical quantum numbers (particle number)
	assembly->d = 2;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	assembly->qsite[0] = 0;
	assembly->qsite[1] = 1;

	// operator map
	assembly->num_local_ops = NUM_MOLECULAR_OID;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	create_molecular_hamiltonian_operator_map(assembly->opmap);

	// interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k:
	// can anti-commute fermionic operators such that i < j and k < l
	struct dense_tensor gint;
	symmetrize_molecular_interaction_coefficients(vint, &gint);
	// prefactor 1/2
	const double one_half = 0.5;
	scale_dense_tensor(&one_half, &gint);

	// coefficient map
	double* coeffmap = ct_malloc((2 + nsites2 + nsites_choose_two * nsites_choose_two) * sizeof(double));
	// first two entries must always be 0 and 1
	coeffmap[0] = 0.;
	coeffmap[1] = 1.;
	int* tkin_cids = ct_calloc(nsites2, sizeof(int));
	int* gint_cids = ct_calloc(nsites2*nsites2, sizeof(int));
	int c = 2;
	const double* tkin_data = tkin->data;
	for (int i = 0; i < nsites; i++) {
		for (int j = 0; j < nsites; j++) {
			const int idx = i*nsites + j;
			// if optimize == false, retain a universal mapping between 'tkin' and 'coeffmap', independent of zero entries
			if (optimize && (tkin_data[idx] == 0)) {
				// filter out zero coefficients
				tkin_cids[idx] = CID_ZERO;
			}
			else {
				coeffmap[c] = tkin_data[idx];
				tkin_cids[idx] = c;
				c++;
			}
		}
	}
	const double* gint_data = gint.data;
	for (int i = 0; i < nsites; i++) {
		for (int j = i + 1; j < nsites; j++) {  // i < j
			for (int k = 0; k < nsites; k++) {
				for (int l = k + 1; l < nsites; l++) {  // k < l
					const int idx = ((i*nsites + j)*nsites + k)*nsites + l;
					// if optimize == false, retain a universal mapping between 'gint' and 'coeffmap', independent of zero entries
					if (optimize && (gint_data[idx] == 0)) {
						// filter out zero coefficients
						gint_cids[idx] = CID_ZERO;
					}
					else {
						coeffmap[c] = gint_data[idx];
						gint_cids[idx] = c;
						c++;
					}
				}
			}
		}
	}
	assert(c <= 2 + nsites2 + nsites_choose_two * nsites_choose_two);
	assembly->num_coeffs = c;
	assembly->coeffmap = coeffmap;

	if (optimize)
	{
		const int nchains = nsites2 + nsites_choose_two * nsites_choose_two;
		struct op_chain* opchains = ct_malloc(nchains * sizeof(struct op_chain));
		int oc = 0;
		// kinetic hopping terms t_{i,j} a^{\dagger}_i a_j
		for (int i = 0; i < nsites; i++)
		{
			// case i < j
			for (int j = i + 1; j < nsites; j++)
			{
				allocate_op_chain(j - i + 1, &opchains[oc]);
				opchains[oc].oids[0] = MOLECULAR_OID_C;
				for (int n = 1; n < j - i; n++) {
					opchains[oc].oids[n] = MOLECULAR_OID_Z;
				}
				opchains[oc].oids[j - i] = MOLECULAR_OID_A;
				opchains[oc].qnums[0] = 0;
				for (int n = 1; n < j - i + 1; n++) {
					opchains[oc].qnums[n] = 1;
				}
				opchains[oc].qnums[j - i + 1] = 0;
				opchains[oc].cid    = tkin_cids[i*nsites + j];
				opchains[oc].istart = i;
				oc++;
			}
			// diagonal hopping term
			{
				allocate_op_chain(1, &opchains[oc]);
				opchains[oc].oids[0]  = MOLECULAR_OID_N;
				opchains[oc].qnums[0] = 0;
				opchains[oc].qnums[1] = 0;
				opchains[oc].cid      = tkin_cids[i*nsites + i];
				opchains[oc].istart   = i;
				oc++;
			}
			// case i > j
			for (int j = 0; j < i; j++)
			{
				allocate_op_chain(i - j + 1, &opchains[oc]);
				opchains[oc].oids[0] = MOLECULAR_OID_A;
				for (int n = 1; n < i - j; n++) {
					opchains[oc].oids[n] = MOLECULAR_OID_Z;
				}
				opchains[oc].oids[i - j] = MOLECULAR_OID_C;
				opchains[oc].qnums[0] = 0;
				for (int n = 1; n < i - j + 1; n++) {
					opchains[oc].qnums[n] = -1;
				}
				opchains[oc].qnums[i - j + 1] = 0;
				opchains[oc].cid    = tkin_cids[i*nsites + j];
				opchains[oc].istart = j;
				oc++;
			}
		}
		// interaction terms
		for (int i = 0; i < nsites; i++)
		{
			for (int j = i + 1; j < nsites; j++)  // i < j
			{
				for (int k = 0; k < nsites; k++)
				{
					for (int l = k + 1; l < nsites; l++)  // k < l
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
								opchains[oc].oids[0] = MOLECULAR_OID_N;
								for (int n = 1; n < da; n++) {
									opchains[oc].oids[n] = MOLECULAR_OID_I;
								}
								opchains[oc].oids[da] = MOLECULAR_OID_N;
								// all quantum numbers are zero
								for (int n = 0; n < da + 2; n++) {
									opchains[oc].qnums[n] = 0;
								}
							}
							else
							{
								// number operator at the beginning
								// operator IDs
								opchains[oc].oids[0] = MOLECULAR_OID_N;
								for (int n = 1; n < ca; n++) {
									opchains[oc].oids[n] = MOLECULAR_OID_I;
								}
								opchains[oc].oids[ca] = (r == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
								for (int n = ca + 1; n < da; n++) {
									opchains[oc].oids[n] = MOLECULAR_OID_Z;
								}
								opchains[oc].oids[da] = (s == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
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
							opchains[oc].oids[0] = (p == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = 1; n < ba; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_Z;
							}
							opchains[oc].oids[ba] = MOLECULAR_OID_N;
							for (int n = ba + 1; n < da; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_Z;
							}
							opchains[oc].oids[da] = (s == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
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
							opchains[oc].oids[0] = (p == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = 1; n < ba; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_Z;
							}
							opchains[oc].oids[ba] = (q == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = ba + 1; n < ca; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_I;
							}
							opchains[oc].oids[ca] = MOLECULAR_OID_N;
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
							opchains[oc].oids[0] = (p == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = 1; n < ba; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_Z;
							}
							opchains[oc].oids[ba] = (q == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = ba + 1; n < ca; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_I;
							}
							opchains[oc].oids[ca] = (r == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
							for (int n = ca + 1; n < da; n++) {
								opchains[oc].oids[n] = MOLECULAR_OID_Z;
							}
							opchains[oc].oids[da] = (s == 1 ? MOLECULAR_OID_C : MOLECULAR_OID_A);
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

						opchains[oc].cid    = gint_cids[((i*nsites + j)*nsites + k)*nsites + l];
						opchains[oc].istart = a;
						oc++;
					}
				}
			}
		}
		assert(oc == nchains);

		mpo_graph_from_opchains(opchains, nchains, nsites, &assembly->graph);

		// clean up
		for (int i = 0; i < nchains; i++) {
			delete_op_chain(&opchains[i]);
		}
		ct_free(opchains);
	}
	else
	{
		// explicit construction (typically faster, but does not optimize cases
		// of zero coefficients, and is slightly sub-optimal close to boundary)

		struct molecular_mpo_graph_vids vids;
		create_molecular_mpo_graph_vertices(nsites, &assembly->graph, &vids);

		// temporarily store edges in linked lists
		struct linked_list* edges = ct_calloc(nsites, sizeof(struct linked_list));
		molecular_mpo_graph_connect_operator_strings(&vids, edges);

		// kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
		for (int i = 0; i < nsites; i++) {
			for (int j = 0; j < nsites; j++) {
				struct molecular_local_op_ref oplist[2] = {
					{ .isite = i, .oid = MOLECULAR_OID_C },
					{ .isite = j, .oid = MOLECULAR_OID_A }, };
				molecular_mpo_graph_add_term(&vids, oplist, ARRLEN(oplist), tkin_cids[i*nsites + j], edges);
			}
		}

		// interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
		for (int i = 0; i < nsites; i++) {
			for (int j = i + 1; j < nsites; j++) {  // i < j
				for (int k = 0; k < nsites; k++) {
					for (int l = k + 1; l < nsites; l++) {  // k < l
						struct molecular_local_op_ref oplist[4] = {
							{ .isite = i, .oid = MOLECULAR_OID_C },
							{ .isite = j, .oid = MOLECULAR_OID_C },
							{ .isite = l, .oid = MOLECULAR_OID_A },
							{ .isite = k, .oid = MOLECULAR_OID_A }, };
						molecular_mpo_graph_add_term(&vids, oplist, ARRLEN(oplist), gint_cids[((i*nsites + j)*nsites + k)*nsites + l], edges);
					}
				}
			}
		}

		// transfer edges into mpo_graph structure and connect vertices
		assembly->graph.edges     = ct_calloc(nsites, sizeof(struct mpo_graph_edge*));
		assembly->graph.num_edges = ct_calloc(nsites, sizeof(int));
		for (int i = 0; i < nsites; i++)
		{
			assembly->graph.num_edges[i] = edges[i].size;
			assembly->graph.edges[i] = ct_malloc(edges[i].size * sizeof(struct mpo_graph_edge));
			struct linked_list_node* edge_ref = edges[i].head;
			int eid = 0;
			while (edge_ref != NULL)
			{
				const struct mpo_graph_edge* edge = edge_ref->data;
				memcpy(&assembly->graph.edges[i][eid], edge, sizeof(struct mpo_graph_edge));

				// create references from graph vertices to edge
				assert(0 <= edge->vids[0] && edge->vids[0] < assembly->graph.num_verts[i]);
				assert(0 <= edge->vids[1] && edge->vids[1] < assembly->graph.num_verts[i + 1]);
				mpo_graph_vertex_add_edge(1, eid, &assembly->graph.verts[i    ][edge->vids[0]]);
				mpo_graph_vertex_add_edge(0, eid, &assembly->graph.verts[i + 1][edge->vids[1]]);

				edge_ref = edge_ref->next;
				eid++;
			}
			// note: opics pointers of edges have been retained in transfer
			delete_linked_list(&edges[i], ct_free);
		}
		ct_free(edges);

		assert(mpo_graph_is_consistent(&assembly->graph));

		delete_molecular_mpo_graph_vertices(&vids);
	}

	ct_free(gint_cids);
	ct_free(tkin_cids);
	delete_dense_tensor(&gint);
}


//________________________________________________________________________________________________________________________
///
/// \brief Local operator IDs for a molecular Hamiltonian using a spin orbital basis.
///
enum spin_molecular_oid
{
	SPIN_MOLECULAR_OID_Id  =  0,  //!< identity
	SPIN_MOLECULAR_OID_IC  =  1,  //!< \f$a^{\dagger}_{\downarrow}\f$
	SPIN_MOLECULAR_OID_IA  =  2,  //!< \f$a_{\downarrow}\f$
	SPIN_MOLECULAR_OID_IN  =  3,  //!< \f$n_{\downarrow}\f$
	SPIN_MOLECULAR_OID_CI  =  4,  //!< \f$a^{\dagger}_{\uparrow}\f$
	SPIN_MOLECULAR_OID_CC  =  5,  //!< \f$a^{\dagger}_{\uparrow} a^{\dagger}_{\downarrow}\f$
	SPIN_MOLECULAR_OID_CA  =  6,  //!< \f$a^{\dagger}_{\uparrow} a_{\downarrow}\f$
	SPIN_MOLECULAR_OID_CN  =  7,  //!< \f$a^{\dagger}_{\uparrow} n_{\downarrow}\f$
	SPIN_MOLECULAR_OID_CZ  =  8,  //!< \f$a^{\dagger}_{\uparrow} Z_{\downarrow}\f$
	SPIN_MOLECULAR_OID_AI  =  9,  //!< \f$a_{\uparrow}\f$
	SPIN_MOLECULAR_OID_AC  = 10,  //!< \f$a_{\uparrow} a^{\dagger}_{\downarrow}\f$
	SPIN_MOLECULAR_OID_AA  = 11,  //!< \f$a_{\uparrow} a_{\downarrow}\f$
	SPIN_MOLECULAR_OID_AN  = 12,  //!< \f$a_{\uparrow} n_{\downarrow}\f$
	SPIN_MOLECULAR_OID_AZ  = 13,  //!< \f$a_{\uparrow} Z_{\downarrow}\f$
	SPIN_MOLECULAR_OID_NI  = 14,  //!< \f$n_{\uparrow}\f$
	SPIN_MOLECULAR_OID_NC  = 15,  //!< \f$n_{\uparrow} a^{\dagger}_{\downarrow}\f$
	SPIN_MOLECULAR_OID_NA  = 16,  //!< \f$n_{\uparrow} a_{\downarrow}\f$
	SPIN_MOLECULAR_OID_NN  = 17,  //!< \f$n_{\uparrow} n_{\downarrow}\f$
	SPIN_MOLECULAR_OID_NZ  = 18,  //!< \f$n_{\uparrow} Z_{\downarrow}\f$
	SPIN_MOLECULAR_OID_ZC  = 19,  //!< \f$Z_{\uparrow} a^{\dagger}_{\downarrow}\f$
	SPIN_MOLECULAR_OID_ZA  = 20,  //!< \f$Z_{\uparrow} a_{\downarrow}\f$
	SPIN_MOLECULAR_OID_ZN  = 21,  //!< \f$Z_{\uparrow} n_{\downarrow}\f$
	SPIN_MOLECULAR_OID_ZZ  = 22,  //!< \f$Z_{\uparrow} Z_{\downarrow}\f$
	NUM_SPIN_MOLECULAR_OID = 23,  //!< number of local operators
};


/// \brief Map a molecular operator ID pair (for spin-up and spin-down) to the corresponding spin molecular operator ID.
static const enum spin_molecular_oid molecular_oid_single_pair_map[5][5] = {
	//        I                      C                      A                      N                      Z
	{ SPIN_MOLECULAR_OID_Id, SPIN_MOLECULAR_OID_IC, SPIN_MOLECULAR_OID_IA, SPIN_MOLECULAR_OID_IN, -1                    },  // I
	{ SPIN_MOLECULAR_OID_CI, SPIN_MOLECULAR_OID_CC, SPIN_MOLECULAR_OID_CA, SPIN_MOLECULAR_OID_CN, SPIN_MOLECULAR_OID_CZ },  // C
	{ SPIN_MOLECULAR_OID_AI, SPIN_MOLECULAR_OID_AC, SPIN_MOLECULAR_OID_AA, SPIN_MOLECULAR_OID_AN, SPIN_MOLECULAR_OID_AZ },  // A
	{ SPIN_MOLECULAR_OID_NI, SPIN_MOLECULAR_OID_NC, SPIN_MOLECULAR_OID_NA, SPIN_MOLECULAR_OID_NN, SPIN_MOLECULAR_OID_NZ },  // N
	{ -1,                    SPIN_MOLECULAR_OID_ZC, SPIN_MOLECULAR_OID_ZA, SPIN_MOLECULAR_OID_ZN, SPIN_MOLECULAR_OID_ZZ },  // Z
};


//________________________________________________________________________________________________________________________
///
/// \brief Create the local operator map for a molecular Hamiltonian, assuming a spin orbital basis.
///
static void create_spin_molecular_hamiltonian_operator_map(struct dense_tensor* opmap)
{
	struct dense_tensor opmap_single[NUM_MOLECULAR_OID];
	create_molecular_hamiltonian_operator_map(opmap_single);

	for (int i = 0; i < NUM_MOLECULAR_OID; i++)
	{
		for (int j = 0; j < NUM_MOLECULAR_OID; j++)
		{
			const int oid = molecular_oid_single_pair_map[i][j];
			if (oid == -1) {
				continue;
			}
			dense_tensor_kronecker_product(&opmap_single[i], &opmap_single[j], &opmap[oid]);
		}
	}

	for (int i = 0; i < NUM_MOLECULAR_OID; i++) {
		delete_dense_tensor(&opmap_single[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize the interaction coefficient tensor as
/// gint0 = vint + transpose(vint, (1, 0, 3, 2)) and gint1 = -(transpose(vint, (1, 0, 2, 3)) + transpose(vint, (0, 1, 3, 2)))
///
static inline void symmetrize_spin_molecular_interaction_coefficients(const struct dense_tensor* restrict vint, struct dense_tensor* restrict gint0, struct dense_tensor* restrict gint1)
{
	assert(vint->ndim == 4);

	copy_dense_tensor(vint, gint0);

	// + transpose(vint, (1, 0, 3, 2))
	{
		const int perm[4] = { 1, 0, 3, 2 };
		struct dense_tensor vint_tp;
		transpose_dense_tensor(perm, vint, &vint_tp);
		dense_tensor_scalar_multiply_add(numeric_one(vint->dtype), &vint_tp, gint0);
		delete_dense_tensor(&vint_tp);
	}

	// transpose(vint, (1, 0, 2, 3))
	{
		const int perm[4] = { 1, 0, 2, 3 };
		transpose_dense_tensor(perm, vint, gint1);
	}

	// + transpose(vint, (0, 1, 3, 2))
	{
		const int perm[4] = { 0, 1, 3, 2 };
		struct dense_tensor vint_tp;
		transpose_dense_tensor(perm, vint, &vint_tp);
		dense_tensor_scalar_multiply_add(numeric_one(vint->dtype), &vint_tp, gint1);
		delete_dense_tensor(&vint_tp);
	}

	// gint1 -> -gint1
	scale_dense_tensor(numeric_neg_one(gint1->dtype), gint1);
}


//________________________________________________________________________________________________________________________
///
/// \brief Vertex IDs in an operator graph used for molecular Hamiltonian construction, assuming a spin orbital basis.
///
/// The two array dimensions specify the connection to the left and right boundary, respectively.
///
struct spin_molecular_mpo_graph_vids
{
	int*  identity[2];     //!< identity chains
	int** a_dag[2];        //!< \f$a^{\dagger}_{i,\sigma}\f$
	int** a_ann[2];        //!< \f$a_{i,\sigma}\f$
	int** a_dag_a_dag[2];  //!< \f$a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau}\f$
	int** a_ann_a_ann[2];  //!< \f$a_{i,\sigma} a_{j,\tau}\f$
	int** a_dag_a_ann[2];  //!< \f$a^{\dagger}_{i,\sigma} a_{j,\tau}\f$
	int   nsites;          //!< number of sites (spatial orbitals)
};


//________________________________________________________________________________________________________________________
///
/// \brief Create the vertices in an operator graph used for molecular Hamiltonian construction, assuming a spin orbital basis.
///
static void create_spin_molecular_mpo_graph_vertices(const int nsites, struct mpo_graph* graph, struct spin_molecular_mpo_graph_vids* vids)
{
	graph->verts     = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	graph->num_verts = ct_calloc(nsites + 1, sizeof(int));
	graph->nsites    = nsites;

	for (int i = 0; i < nsites + 1; i++)
	{
		// number of vertices, i.e., virtual bond dimensions
		const int nl = i;
		const int nr = nsites - i;
		const int n = imin(nl, nr);
		// identity chains
		int chi1 = (1 <= i && i <= nsites - 1) ? 2 : 1;
		// a^{\dagger}_{i,\sigma} and a_{i,\sigma} chains, reaching (almost) from one boundary to the other
		int chi2 = 4 * ((i < nsites ? nl : 0) + (i > 0 ? nr : 0));
		// a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} (for (i, \sigma) < (j, \tau)), a_{i,\sigma} a_{j,\tau} (for (i, \sigma) > (j, \tau)) and a^{\dagger}_{i,\sigma} a_{j,\tau} chains, extending from boundary to center
		int chi3 = 2 * n * (2 * n - 1) + 4 * n * n;

		graph->num_verts[i] = chi1 + chi2 + chi3;
		graph->verts[i] = ct_calloc(graph->num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	vids->nsites = nsites;

	int* vertex_counter = ct_calloc(nsites + 1, sizeof(int));

	// vertex IDs

	const qnumber qnum_spin[2] = { 1, -1 };

	// identity chains from the left and right
	vids->identity[MOLECULAR_CONN_LEFT]  = allocate_vertex_ids(nsites);
	vids->identity[MOLECULAR_CONN_RIGHT] = allocate_vertex_ids(nsites);
	for (int i = 0; i < nsites; i++) {
		vids->identity[MOLECULAR_CONN_LEFT][i] = vertex_counter[i]++;
	}
	for (int i = 1; i < nsites + 1; i++) {
		vids->identity[MOLECULAR_CONN_RIGHT][i] = vertex_counter[i]++;
	}

	// a^{\dagger}_{i,\sigma} operators connected to left boundary
	vids->a_dag[MOLECULAR_CONN_LEFT] = ct_calloc(2*nsites, sizeof(int*));
	for (int i = 0; i < nsites - 1; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma] = allocate_vertex_ids(nsites);
			for (int j = i + 1; j < nsites; j++) {
				const int vid = vertex_counter[j]++;
				vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][j] = vid;
				graph->verts[j][vid].qnum = encode_quantum_number_pair( 1,  qnum_spin[sigma]);
			}
		}
	}

	// a_{i,\sigma} operators connected to left boundary
	vids->a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(2*nsites, sizeof(int*));
	for (int i = 0; i < nsites - 1; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma] = allocate_vertex_ids(nsites);
			for (int j = i + 1; j < nsites; j++) {
				const int vid = vertex_counter[j]++;
				vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma][j] = vid;
				graph->verts[j][vid].qnum = encode_quantum_number_pair(-1, -qnum_spin[sigma]);
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left boundary, for (i,\sigma) < (j,\tau)
	vids->a_dag_a_dag[MOLECULAR_CONN_LEFT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx] = allocate_vertex_ids(nsites);
					for (int k = j + 1; k < nsites/2 + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair( 2,  qnum_spin[sigma] + qnum_spin[tau]);
					}
				}
			}
		}
	}

	// a_{i,\sigma} a_{j,\tau} operators connected to left boundary, for (i,\sigma) > (j,\tau)
	vids->a_ann_a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx] = allocate_vertex_ids(nsites);
					for (int k = i + 1; k < nsites/2 + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair(-2, -qnum_spin[sigma] - qnum_spin[tau]);
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left boundary
	vids->a_dag_a_ann[MOLECULAR_CONN_LEFT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx] = allocate_vertex_ids(nsites);
					for (int k = imax(i, j) + 1; k < nsites/2 + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair( 0,  qnum_spin[sigma] - qnum_spin[tau]);
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} operators connected to right boundary
	vids->a_dag[MOLECULAR_CONN_RIGHT] = ct_calloc(2*nsites, sizeof(int*));
	for (int i = 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma] = allocate_vertex_ids(nsites);
			for (int j = 1; j < i + 1; j++) {
				const int vid = vertex_counter[j]++;
				vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma][j] = vid;
				graph->verts[j][vid].qnum = encode_quantum_number_pair(-1, -qnum_spin[sigma]);
			}
		}
	}

	// a_{i,\sigma} operators connected to right boundary
	vids->a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(2*nsites, sizeof(int*));
	for (int i = 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma] = allocate_vertex_ids(nsites);
			for (int j = 1; j < i + 1; j++) {
				const int vid = vertex_counter[j]++;
				vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma][j] = vid;
				graph->verts[j][vid].qnum = encode_quantum_number_pair( 1,  qnum_spin[sigma]);
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right boundary, for (i,\sigma) < (j,\tau)
	vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx] = allocate_vertex_ids(nsites);
					for (int k = nsites/2 + 1; k < i + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair(-2, -qnum_spin[sigma] - qnum_spin[tau]);
					}
				}
			}
		}
	}

	// a_{i,\sigma} a_{j,\tau} operators connected to right boundary, for (i,\sigma) > (j,\tau)
	vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx] = allocate_vertex_ids(nsites);
					for (int k = nsites/2 + 1; k < j + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair( 2,  qnum_spin[sigma] + qnum_spin[tau]);
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right boundary
	vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT] = ct_calloc(4*nsites*nsites, sizeof(int*));
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx] = allocate_vertex_ids(nsites);
					for (int k = nsites/2 + 1; k < imin(i, j) + 1; k++) {
						const int vid = vertex_counter[k]++;
						vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][k] = vid;
						graph->verts[k][vid].qnum = encode_quantum_number_pair( 0, -qnum_spin[sigma] + qnum_spin[tau]);
					}
				}
			}
		}
	}

	// consistency check
	for (int i = 0; i < nsites + 1; i++) {
		assert(vertex_counter[i] == graph->num_verts[i]);
	}

	ct_free(vertex_counter);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the vertices in an operator graph used for molecular Hamiltonian construction, assuming a spin orbital basis.
///
static void delete_spin_molecular_mpo_graph_vertices(struct spin_molecular_mpo_graph_vids* vids)
{
	const int nsites = vids->nsites;

	ct_free(vids->identity[MOLECULAR_CONN_LEFT]);
	ct_free(vids->identity[MOLECULAR_CONN_RIGHT]);

	for (int i = 0; i < nsites - 1; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			ct_free(vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma]);
			ct_free(vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma]);
		}
	}
	ct_free(vids->a_dag[MOLECULAR_CONN_LEFT]);
	ct_free(vids->a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT]);

	for (int i = 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			ct_free(vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma]);
			ct_free(vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma]);
		}
	}
	ct_free(vids->a_dag[MOLECULAR_CONN_RIGHT]);
	ct_free(vids->a_ann[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT]);

	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx]);
				}
			}
		}
	}
	ct_free(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Connect the vertices of the creation and annihilation operator strings
/// in the operator graph describing a molecular Hamiltonian, assuming a spin orbital basis.
///
static void spin_molecular_mpo_graph_connect_operator_strings(const struct spin_molecular_mpo_graph_vids* vids, struct linked_list* edges)
{
	const int nsites = vids->nsites;

	// identities connected to left and right boundaries
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->identity[MOLECULAR_CONN_LEFT][i + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i],
			construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_RIGHT][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
	}

	// a^{\dagger}_{i,\sigma} operators connected to left boundary
	for (int i = 0; i < nsites - 1; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_CZ : SPIN_MOLECULAR_OID_IC, CID_ONE));
			// Z operator from Jordan-Wigner transformation
			for (int j = i + 1; j < nsites - 1; j++) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][j], vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][j + 1], SPIN_MOLECULAR_OID_ZZ, CID_ONE));
			}
		}
	}

	// a_{i,\sigma} operators connected to left boundary
	for (int i = 0; i < nsites - 1; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_AZ : SPIN_MOLECULAR_OID_IA, CID_ONE));
			// Z operator from Jordan-Wigner transformation
			for (int j = i + 1; j < nsites - 1; j++) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma][j], vids->a_ann[MOLECULAR_CONN_LEFT][2*i + sigma][j + 1], SPIN_MOLECULAR_OID_ZZ, CID_ONE));
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left boundary, for (i,\sigma) < (j,\tau)
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					if (i < j) {
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][j], vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx][j + 1], tau == 0 ? SPIN_MOLECULAR_OID_CI : SPIN_MOLECULAR_OID_ZC, CID_ONE));
					}
					else {
						assert(sigma == 0 && tau == 1);
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][j], vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx][j + 1], SPIN_MOLECULAR_OID_CC, CID_ONE));
					}
					// identities for transition to next site
					for (int k = j + 1; k < nsites/2; k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx][k], vids->a_dag_a_dag[MOLECULAR_CONN_LEFT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
				}
			}
		}
	}

	// a_{i,\sigma} a_{j,\tau} operators connected to left boundary, for (i,\sigma) > (j,\tau)
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					if (i > j) {
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][2*j + tau][i], vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_AI : SPIN_MOLECULAR_OID_ZA, CID_ONE));
					}
					else {
						assert(sigma == 1 && tau == 0);
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx][i + 1], SPIN_MOLECULAR_OID_AA, CID_ONE));
					}
					// identities for transition to next site
					for (int k = i + 1; k < nsites/2; k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx][k], vids->a_ann_a_ann[MOLECULAR_CONN_LEFT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left boundary
	for (int i = 0; i < nsites/2; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = 0; j < nsites/2; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					if (i < j) {
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_LEFT][2*i + sigma][j], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][j + 1], tau == 0 ? SPIN_MOLECULAR_OID_AI : SPIN_MOLECULAR_OID_ZA, CID_ONE));
					}
					else if (i == j)
					{
						enum spin_molecular_oid oid;
						if (sigma < tau) {
							oid = SPIN_MOLECULAR_OID_CA;
						}
						else if (sigma == tau) {
							oid = (sigma == 0 ? SPIN_MOLECULAR_OID_NI : SPIN_MOLECULAR_OID_IN);
						}
						else {  // sigma > tau
							oid = SPIN_MOLECULAR_OID_AC;
						}
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->identity[MOLECULAR_CONN_LEFT][i], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][i + 1], oid, CID_ONE));
					}
					else { // i > j
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_LEFT][2*j + tau][i], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_CI : SPIN_MOLECULAR_OID_ZC, CID_ONE));
					}
					// identities for transition to next site
					for (int k = imax(i, j) + 1; k < nsites/2; k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][k], vids->a_dag_a_ann[MOLECULAR_CONN_LEFT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} operators connected to right boundary
	for (int i = 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			// Z operator from Jordan-Wigner transformation
			for (int j = 1; j < i; j++) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma][j], vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma][j + 1], SPIN_MOLECULAR_OID_ZZ, CID_ONE));
			}
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_CI : SPIN_MOLECULAR_OID_ZC, CID_ONE));
		}
	}

	// a_{i,\sigma} operators connected to right boundary
	for (int i = 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			// Z operator from Jordan-Wigner transformation
			for (int j = 1; j < i; j++) {
				linked_list_append(&edges[j],
					construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma][j], vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma][j + 1], SPIN_MOLECULAR_OID_ZZ, CID_ONE));
			}
			linked_list_append(&edges[i],
				construct_mpo_graph_edge(vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_AI : SPIN_MOLECULAR_OID_ZA, CID_ONE));
		}
	}

	// a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right boundary, for (i,\sigma) < (j,\tau)
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = i; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma >= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					// identities for transition to next site
					for (int k = nsites/2 + 1; k < i; k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx][k], vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
					if (i < j) {
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx][i], vids->a_dag[MOLECULAR_CONN_RIGHT][2*j + tau][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_CZ : SPIN_MOLECULAR_OID_IC, CID_ONE));
					}
					else {
						assert(sigma == 0 && tau == 1);
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_dag_a_dag[MOLECULAR_CONN_RIGHT][idx][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], SPIN_MOLECULAR_OID_CC, CID_ONE));
					}
				}
			}
		}
	}

	// a_{i,\sigma} a_{j,\tau} operators connected to right boundary, for (i,\sigma) > (j,\tau)
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j <= i; j++) {
				for (int tau = 0; tau < 2; tau++) {
					if (i == j && sigma <= tau) { continue; }
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					// identities for transition to next site
					for (int k = nsites/2 + 1; k < j; k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx][k], vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
					if (i > j) {
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx][j], vids->a_ann[MOLECULAR_CONN_RIGHT][2*i + sigma][j + 1], tau == 0 ? SPIN_MOLECULAR_OID_AZ : SPIN_MOLECULAR_OID_IA, CID_ONE));
					}
					else {
						assert(sigma == 1 && tau == 0);
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->a_ann_a_ann[MOLECULAR_CONN_RIGHT][idx][j], vids->identity[MOLECULAR_CONN_RIGHT][j + 1], SPIN_MOLECULAR_OID_AA, CID_ONE));
					}
				}
			}
		}
	}

	// a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right boundary
	for (int i = nsites/2 + 1; i < nsites; i++) {
		for (int sigma = 0; sigma < 2; sigma++) {
			for (int j = nsites/2 + 1; j < nsites; j++) {
				for (int tau = 0; tau < 2; tau++) {
					const int idx = (2*i + sigma)*2*nsites + (2*j + tau);
					// identities for transition to next site
					for (int k = nsites/2 + 1; k < imin(i, j); k++) {
						linked_list_append(&edges[k],
							construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][k], vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][k + 1], SPIN_MOLECULAR_OID_Id, CID_ONE));
					}
					if (i < j) {
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][i], vids->a_ann[MOLECULAR_CONN_RIGHT][2*j + tau][i + 1], sigma == 0 ? SPIN_MOLECULAR_OID_CZ : SPIN_MOLECULAR_OID_IC, CID_ONE));
					}
					else if (i == j) {
						enum spin_molecular_oid oid;
						if (sigma < tau) {
							oid = SPIN_MOLECULAR_OID_CA;
						}
						else if (sigma == tau) {
							oid = (sigma == 0 ? SPIN_MOLECULAR_OID_NI : SPIN_MOLECULAR_OID_IN);
						}
						else { // sigma > tau
							oid = SPIN_MOLECULAR_OID_AC;
						}
						linked_list_append(&edges[i],
							construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], oid, CID_ONE));
					}
					else { // i > j
						linked_list_append(&edges[j],
							construct_mpo_graph_edge(vids->a_dag_a_ann[MOLECULAR_CONN_RIGHT][idx][j], vids->a_dag[MOLECULAR_CONN_RIGHT][2*i + sigma][j + 1], tau == 0 ? SPIN_MOLECULAR_OID_AZ : SPIN_MOLECULAR_OID_IA, CID_ONE));
					}
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Local operator description for a molecular Hamiltonian, assuming a spin orbital basis.
///
struct spin_molecular_local_op_ref
{
	int isite;  //!< site index
	int ispin;  //!< spin index (0: spin-up, 1: spin-down)
	int oid;    //!< operator index
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_spin_molecular_local_op_ref(const void* a, const void* b)
{
	const struct spin_molecular_local_op_ref* x = (const struct spin_molecular_local_op_ref*)a;
	const struct spin_molecular_local_op_ref* y = (const struct spin_molecular_local_op_ref*)b;

	if (x->isite < y->isite) {
		return -1;
	}
	if (x->isite > y->isite) {
		return 1;
	}
	// x->isite == y->isite
	if (x->ispin < y->ispin) {
		return -1;
	}
	if (x->ispin > y->ispin) {
		return 1;
	}
	// x->ispin == y->ispin
	if (x->oid < y->oid) {
		return -1;
	}
	if (x->oid > y->oid) {
		return 1;
	}
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Spin - operator ID tuples (temporary data structure).
///
struct spin_oid_tuple
{
	int ispin;  //!< spin index (0: spin-up, 1: spin-down)
	int oid;    //!< operator index
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_spin_oid_tuple(const void* a, const void* b)
{
	const struct spin_oid_tuple* x = (const struct spin_oid_tuple*)a;
	const struct spin_oid_tuple* y = (const struct spin_oid_tuple*)b;

	if (x->ispin < y->ispin) {
		return -1;
	}
	if (x->ispin > y->ispin) {
		return 1;
	}
	// x->ispin == y->ispin
	if (x->oid < y->oid) {
		return -1;
	}
	if (x->oid > y->oid) {
		return 1;
	}
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a list of local creation and annihilation operators of the form [(spin_a, oid_a), ...]
/// to the corresponding operator using a spin orbital basis.
///
static enum spin_molecular_oid to_spin_molecular_operator(const struct spin_oid_tuple* oplist, const int numops, const bool even_parity_left, const bool even_parity_right)
{
	assert(1 <= numops && numops <= 4);
	struct spin_oid_tuple oplist_sorted[4];
	memcpy(oplist_sorted, oplist, numops * sizeof(struct spin_oid_tuple));
	qsort(oplist_sorted, numops, sizeof(struct spin_oid_tuple), compare_spin_oid_tuple);

	if (numops == 1)
	{
		if (oplist_sorted[0].ispin == 0) {
			enum molecular_oid transfer = (even_parity_right ? MOLECULAR_OID_I : MOLECULAR_OID_Z);
			return molecular_oid_single_pair_map[oplist_sorted[0].oid][transfer];
		}
		else {
			assert(oplist_sorted[0].ispin == 1);
			enum molecular_oid transfer = (even_parity_left ? MOLECULAR_OID_I : MOLECULAR_OID_Z);
			return molecular_oid_single_pair_map[transfer][oplist_sorted[0].oid];
		}
	}
	if (numops == 2)
	{
		if (oplist_sorted[0].ispin == oplist_sorted[1].ispin) {
			assert(oplist_sorted[0].oid == MOLECULAR_OID_C && oplist_sorted[1].oid == MOLECULAR_OID_A);
			if (oplist_sorted[0].ispin == 0) {
				return (even_parity_right ? SPIN_MOLECULAR_OID_NI : SPIN_MOLECULAR_OID_NZ);
			}
			else {
				assert(oplist_sorted[0].ispin == 1);
				return (even_parity_left ? SPIN_MOLECULAR_OID_IN : SPIN_MOLECULAR_OID_ZN);
			}
		}
		assert(oplist_sorted[0].ispin == 0 && oplist_sorted[1].ispin == 1);
		return molecular_oid_single_pair_map[oplist_sorted[0].oid][oplist_sorted[1].oid];
	}
	if (numops == 3)
	{
		if (oplist_sorted[1].ispin == 0) {
			assert(oplist_sorted[0].ispin == 0 && oplist_sorted[2].ispin == 1);
			assert(oplist_sorted[0].oid == MOLECULAR_OID_C && oplist_sorted[1].oid == MOLECULAR_OID_A);
			return molecular_oid_single_pair_map[MOLECULAR_OID_N][oplist_sorted[2].oid];
		}
		else {
			assert(oplist_sorted[0].ispin == 0 && oplist_sorted[1].ispin == 1 && oplist_sorted[2].ispin == 1);
			assert(oplist_sorted[1].oid == MOLECULAR_OID_C && oplist_sorted[2].oid == MOLECULAR_OID_A);
			return molecular_oid_single_pair_map[oplist_sorted[0].oid][MOLECULAR_OID_N];
		}
	}
	if (numops == 4)
	{
		assert(oplist_sorted[0].ispin == 0 && oplist_sorted[1].ispin == 0 && oplist_sorted[2].ispin == 1 && oplist_sorted[3].ispin == 1);
		assert(oplist_sorted[0].oid == MOLECULAR_OID_C && oplist_sorted[1].oid == MOLECULAR_OID_A);
		assert(oplist_sorted[2].oid == MOLECULAR_OID_C && oplist_sorted[3].oid == MOLECULAR_OID_A);
		return SPIN_MOLECULAR_OID_NN;
	}
	// invalid number of operators
	assert(false);
	return SPIN_MOLECULAR_OID_Id;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the list of vertex IDs (for each bond index) corresponding to a single creation or annihilation operator, assuming a spin orbital basis.
///
static inline const int* spin_molecular_mpo_graph_vids_get_single(const struct spin_molecular_mpo_graph_vids* vids, const struct spin_molecular_local_op_ref* op, const int direction)
{
	if (op->oid == MOLECULAR_OID_C) {
		return vids->a_dag[direction][2*op->isite + op->ispin];
	}
	if (op->oid == MOLECULAR_OID_A) {
		return vids->a_ann[direction][2*op->isite + op->ispin];
	}
	// operator vertices do not exist
	assert(false);
	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve the list of vertex IDs (for each bond index) corresponding to a pair of creation and annihilation operators, assuming a spin orbital basis.
///
static inline const int* spin_molecular_mpo_graph_vids_get_pair(const struct spin_molecular_mpo_graph_vids* vids, const struct spin_molecular_local_op_ref* op0, const struct spin_molecular_local_op_ref* op1, const int direction)
{
	const int is = 2*op0->isite + op0->ispin;
	const int jt = 2*op1->isite + op1->ispin;

	if (op0->oid == MOLECULAR_OID_C)
	{
		if (op1->oid == MOLECULAR_OID_C) {
			const int idx = (is < jt ? is*2*vids->nsites + jt : jt*2*vids->nsites + is);
			return vids->a_dag_a_dag[direction][idx];
		}
		if (op1->oid == MOLECULAR_OID_A) {
			const int idx = is*2*vids->nsites + jt;
			return vids->a_dag_a_ann[direction][idx];
		}
	}
	if (op0->oid == MOLECULAR_OID_A)
	{
		if (op1->oid == MOLECULAR_OID_C) {
			const int idx = jt*2*vids->nsites + is;
			return vids->a_dag_a_ann[direction][idx];
		}
		if (op1->oid == MOLECULAR_OID_A) {
			// reverse order
			const int idx = (is < jt ? jt*2*vids->nsites + is : is*2*vids->nsites + jt);
			return vids->a_ann_a_ann[direction][idx];
		}
	}
	// operator vertices do not exist
	assert(false);
	return NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Add an operator term (operator string of creation and annihilation operators)
/// to the operator graph describing a molecular Hamiltonian, assuming a spin orbital basis.
///
static void spin_molecular_mpo_graph_add_term(const struct spin_molecular_mpo_graph_vids* vids, const struct spin_molecular_local_op_ref* oplist, const int numops, const int* cids, const int numcids, struct linked_list* edges)
{
	const int nsites = vids->nsites;

	// sort by site (orbital) and spin index
	struct spin_molecular_local_op_ref* oplist_sorted = ct_malloc(numops * sizeof(struct spin_molecular_local_op_ref));
	memcpy(oplist_sorted, oplist, numops * sizeof(struct spin_molecular_local_op_ref));
	qsort(oplist_sorted, numops, sizeof(struct spin_molecular_local_op_ref), compare_spin_molecular_local_op_ref);

	if (numops == 2)
	{
		const int i = oplist_sorted[0].isite;
		const int j = oplist_sorted[1].isite;
		assert(0 <= i && i < nsites);
		assert(0 <= j && j < nsites);

		if (i == j)
		{
			const struct spin_oid_tuple spin_oid_tuples[2] = {
				{ .ispin = oplist_sorted[0].ispin, .oid = oplist_sorted[0].oid },
				{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
			};
			const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, true);
			linked_list_append(&edges[i],
				construct_mpo_graph_edge_multicids(vids->identity[MOLECULAR_CONN_LEFT][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], spin_oid, cids, numcids));
		}
		else
		{
			assert(i < j);
			if (j <= nsites/2)
			{
				const int* vids_l = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				const struct spin_oid_tuple spin_oid_tuples[1] = {
					{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), false, true);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge_multicids(vids_l[j], vids->identity[MOLECULAR_CONN_RIGHT][j + 1], spin_oid, cids, numcids));
			}
			else if (i >= nsites/2)
			{
				const int* vids_r = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[1], MOLECULAR_CONN_RIGHT);
				const struct spin_oid_tuple spin_oid_tuples[1] = {
					{ .ispin = oplist_sorted[0].ispin, .oid = oplist_sorted[0].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, false);
				linked_list_append(&edges[i],
					construct_mpo_graph_edge_multicids(vids->identity[MOLECULAR_CONN_LEFT][i], vids_r[i + 1], spin_oid, cids, numcids));
			}
			else
			{
				const int* vids_l = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				const int* vids_r = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[1], MOLECULAR_CONN_RIGHT);
				linked_list_append(&edges[nsites/2],
					construct_mpo_graph_edge_multicids(vids_l[nsites/2], vids_r[nsites/2 + 1], SPIN_MOLECULAR_OID_ZZ, cids, numcids));
			}
		}
	}
	else if (numops == 4)
	{
		const int i = oplist_sorted[0].isite;
		const int j = oplist_sorted[1].isite;
		const int k = oplist_sorted[2].isite;
		const int l = oplist_sorted[3].isite;
		assert(0 <= i && i < nsites);
		assert(0 <= j && j < nsites);
		assert(0 <= k && k < nsites);
		assert(0 <= l && l < nsites);

		if ((i == j) && (j == k) && (k == l))
		{
			const struct spin_oid_tuple spin_oid_tuples[4] = {
				{ .ispin = oplist_sorted[0].ispin, .oid = oplist_sorted[0].oid },
				{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
				{ .ispin = oplist_sorted[3].ispin, .oid = oplist_sorted[3].oid },
			};
			const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, true);
			linked_list_append(&edges[i],
				construct_mpo_graph_edge_multicids(vids->identity[MOLECULAR_CONN_LEFT][i], vids->identity[MOLECULAR_CONN_RIGHT][i + 1], spin_oid, cids, numcids));
		}
		else if ((i == j) && (j == k))
		{
			const int* vids_r = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			const struct spin_oid_tuple spin_oid_tuples[3] = {
				{ .ispin = oplist_sorted[0].ispin, .oid = oplist_sorted[0].oid },
				{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
			};
			const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, false);
			linked_list_append(&edges[i],
				construct_mpo_graph_edge_multicids(vids->identity[MOLECULAR_CONN_LEFT][i], vids_r[i + 1], spin_oid, cids, numcids));
		}
		else if ((j == k) && (k == l))
		{
			const int* vids_l = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
			const struct spin_oid_tuple spin_oid_tuples[3] = {
				{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
				{ .ispin = oplist_sorted[3].ispin, .oid = oplist_sorted[3].oid },
			};
			const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), false, true);
			linked_list_append(&edges[j],
				construct_mpo_graph_edge_multicids(vids_l[j], vids->identity[MOLECULAR_CONN_RIGHT][j + 1], spin_oid, cids, numcids));

		}
		else if (j == k)
		{
			const int* vids_l = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
			const int* vids_r = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			const struct spin_oid_tuple spin_oid_tuples[2] = {
				{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
			};
			const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), false, false);
			linked_list_append(&edges[j],
				construct_mpo_graph_edge_multicids(vids_l[j], vids_r[j + 1], spin_oid, cids, numcids));
		}
		else if (k <= nsites/2)
		{
			const int* vids_l = spin_molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[0], &oplist_sorted[1], MOLECULAR_CONN_LEFT);
			if (k == l)
			{
				const struct spin_oid_tuple spin_oid_tuples[2] = {
					{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
					{ .ispin = oplist_sorted[3].ispin, .oid = oplist_sorted[3].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, true);
				linked_list_append(&edges[k],
					construct_mpo_graph_edge_multicids(vids_l[k], vids->identity[MOLECULAR_CONN_RIGHT][k + 1], spin_oid, cids, numcids));
			}
			else
			{
				const int* vids_r = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
				const struct spin_oid_tuple spin_oid_tuples[1] = {
					{ .ispin = oplist_sorted[2].ispin, .oid = oplist_sorted[2].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, false);
				linked_list_append(&edges[k],
					construct_mpo_graph_edge_multicids(vids_l[k], vids_r[k + 1], spin_oid, cids, numcids));
			}
		}
		else if (j >= nsites/2)
		{
			const int* vids_r = spin_molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[2], &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			if (i == j)
			{
				const struct spin_oid_tuple spin_oid_tuples[2] = {
					{ .ispin = oplist_sorted[0].ispin, .oid = oplist_sorted[0].oid },
					{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), true, true);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge_multicids(vids->identity[MOLECULAR_CONN_LEFT][j], vids_r[j + 1], spin_oid, cids, numcids));
			}
			else
			{
				const int* vids_l = spin_molecular_mpo_graph_vids_get_single(vids, &oplist_sorted[0], MOLECULAR_CONN_LEFT);
				const struct spin_oid_tuple spin_oid_tuples[1] = {
					{ .ispin = oplist_sorted[1].ispin, .oid = oplist_sorted[1].oid },
				};
				const enum spin_molecular_oid spin_oid = to_spin_molecular_operator(spin_oid_tuples, ARRLEN(spin_oid_tuples), false, true);
				linked_list_append(&edges[j],
					construct_mpo_graph_edge_multicids(vids_l[j], vids_r[j + 1], spin_oid, cids, numcids));
			}
		}
		else
		{
			const int* vids_l = spin_molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[0], &oplist_sorted[1], MOLECULAR_CONN_LEFT);
			const int* vids_r = spin_molecular_mpo_graph_vids_get_pair(vids, &oplist_sorted[2], &oplist_sorted[3], MOLECULAR_CONN_RIGHT);
			linked_list_append(&edges[nsites/2],
				construct_mpo_graph_edge_multicids(vids_l[nsites/2], vids_r[nsites/2 + 1], SPIN_MOLECULAR_OID_Id, cids, numcids));
		}
	}
	else
	{
		// not implemented
		assert(false);
	}

	ct_free(oplist_sorted);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a molecular Hamiltonian as MPO assembly, assuming a spin orbital basis and
/// using physicists' convention for the interaction term (note ordering of k and l):
/// \f[
/// H = \sum_{i,j} \sum_{\sigma \in \{\uparrow, \downarrow\}} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma} + \frac{1}{2} \sum_{i,j,k,\ell} \sum_{\sigma, \tau \in \{\uparrow, \downarrow\}} v_{i,j,k,\ell} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{\ell,\tau} a_{k,\sigma}
/// \f]
///
/// If 'optimize == true', optimize the virtual bond dimensions via the automatic construction starting from operator chains.
/// Can handle zero entries in 'tkin' and 'vint', but construction takes considerably longer for larger number of orbitals.
///
void construct_spin_molecular_hamiltonian_mpo_assembly(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, const bool optimize, struct mpo_assembly* assembly)
{
	assert(tkin->dtype == CT_DOUBLE_REAL);
	assert(vint->dtype == CT_DOUBLE_REAL);

	assembly->dtype = CT_DOUBLE_REAL;

	// dimension consistency checks
	assert(tkin->ndim == 2);
	assert(vint->ndim == 4);
	assert(tkin->dim[0] == tkin->dim[1]);
	assert(vint->dim[0] == vint->dim[1] &&
	       vint->dim[0] == vint->dim[2] &&
	       vint->dim[0] == vint->dim[3]);
	assert(tkin->dim[0] == vint->dim[0]);

	// number of "sites" (spatial orbitals)
	const int nsites = tkin->dim[0];
	assert(nsites >= 1);
	const int nsites2 = nsites * nsites;
	const int nsites1_choose_two = (nsites + 1) * nsites / 2;

	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = { 0,  1,  1,  2 };
	const qnumber qs[4] = { 0, -1,  1,  0 };
	assembly->d = 4;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	for (long i = 0; i < assembly->d; i++) {
		assembly->qsite[i] = encode_quantum_number_pair(qn[i], qs[i]);
	}

	// operator map
	assembly->num_local_ops = NUM_SPIN_MOLECULAR_OID;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	create_spin_molecular_hamiltonian_operator_map(assembly->opmap);

	// interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau,\mu,\nu} v_{i,j,k,l} \delta_{\sigma,\mu} \delta_{\tau,\nu} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\nu} a_{k,\mu}:
	// can anti-commute fermionic operators such that (i,\sigma) < (j,\tau) and (k,\mu) < (l,\nu)
	struct dense_tensor gint0, gint1;
	symmetrize_spin_molecular_interaction_coefficients(vint, &gint0, &gint1);
	// prefactor 1/2
	const double one_half = 0.5;
	scale_dense_tensor(&one_half, &gint0);
	scale_dense_tensor(&one_half, &gint1);

	// coefficient map
	double* coeffmap = ct_malloc((2 + nsites2 + 2 * nsites1_choose_two * nsites1_choose_two) * sizeof(double));
	// first two entries must always be 0 and 1
	coeffmap[0] = 0.;
	coeffmap[1] = 1.;
	int* tkin_cids  = ct_calloc(nsites2, sizeof(int));
	int* gint0_cids = ct_calloc(nsites2*nsites2, sizeof(int));
	int* gint1_cids = ct_calloc(nsites2*nsites2, sizeof(int));
	int c = 2;
	const double* tkin_data = tkin->data;
	for (int i = 0; i < nsites; i++) {
		for (int j = 0; j < nsites; j++) {
			const int idx = i*nsites + j;
			// if optimize == false, retain a universal mapping between 'tkin' and 'coeffmap', independent of zero entries
			if (optimize && (tkin_data[idx] == 0)) {
				// filter out zero coefficients
				tkin_cids[idx] = CID_ZERO;
			}
			else {
				coeffmap[c] = tkin_data[idx];
				tkin_cids[idx] = c;
				c++;
			}
		}
	}
	const double* gint0_data = gint0.data;
	for (int i = 0; i < nsites; i++) {
		for (int j = i; j < nsites; j++) {  // i <= j
			for (int k = 0; k < nsites; k++) {
				for (int l = k; l < nsites; l++) {  // k <= l
					const int idx = ((i*nsites + j)*nsites + k)*nsites + l;
					// if optimize == false, retain a universal mapping between 'gint' and 'coeffmap', independent of zero entries
					if (optimize && (gint0_data[idx] == 0)) {
						// filter out zero coefficients
						gint0_cids[idx] = CID_ZERO;
					}
					else {
						coeffmap[c] = gint0_data[idx];
						gint0_cids[idx] = c;
						c++;
					}
				}
			}
		}
	}
	const double* gint1_data = gint1.data;
	for (int i = 0; i < nsites; i++) {
		for (int j = i; j < nsites; j++) {  // i <= j
			for (int k = 0; k < nsites; k++) {
				for (int l = k; l < nsites; l++) {  // k <= l
					const int idx = ((i*nsites + j)*nsites + k)*nsites + l;
					// if optimize == false, retain a universal mapping between 'gint' and 'coeffmap', independent of zero entries
					if (optimize && (gint1_data[idx] == 0)) {
						// filter out zero coefficients
						gint1_cids[idx] = CID_ZERO;
					}
					else {
						coeffmap[c] = gint1_data[idx];
						gint1_cids[idx] = c;
						c++;
					}
				}
			}
		}
	}
	assert(c <= 2 + nsites2 + 2 * nsites1_choose_two * nsites1_choose_two);
	assembly->num_coeffs = c;
	assembly->coeffmap = coeffmap;

	if (optimize)
	{
		// not implemented yet
		assert(false);
	}
	else
	{
		// explicit construction (typically faster, but does not optimize cases
		// of zero coefficients, and is slightly sub-optimal close to boundary)

		struct spin_molecular_mpo_graph_vids vids;
		create_spin_molecular_mpo_graph_vertices(nsites, &assembly->graph, &vids);

		// temporarily store edges in linked lists
		struct linked_list* edges = ct_calloc(nsites, sizeof(struct linked_list));
		spin_molecular_mpo_graph_connect_operator_strings(&vids, edges);

		// kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
		for (int i = 0; i < nsites; i++) {
			for (int j = 0; j < nsites; j++) {
				for (int sigma = 0; sigma < 2; sigma++) {
					struct spin_molecular_local_op_ref oplist[2] = {
						{ .isite = i, .ispin = sigma, .oid = MOLECULAR_OID_C },
						{ .isite = j, .ispin = sigma, .oid = MOLECULAR_OID_A }, };
					spin_molecular_mpo_graph_add_term(&vids, oplist, ARRLEN(oplist), &tkin_cids[i*nsites + j], 1, edges);
				}
			}
		}

		// interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l} a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
		for (int is = 0; is < 2*nsites; is++) {
			for (int jt = is + 1; jt < 2*nsites; jt++) {  // is < jt
				for (int km = 0; km < 2*nsites; km++) {
					for (int ln = km + 1; ln < 2*nsites; ln++) {  // km < ln
						const int site_idx[4] = { is / 2, jt / 2, km / 2, ln / 2 };
						const int spin_idx[4] = { is % 2, jt % 2, km % 2, ln % 2 };
						int num_cids = 0;
						int cids[2];
						const int idx = ((site_idx[0]*nsites + site_idx[1])*nsites + site_idx[2])*nsites + site_idx[3];
						if ((spin_idx[0] == spin_idx[2]) && (spin_idx[1] == spin_idx[3])) {
							cids[num_cids] = gint0_cids[idx];
							num_cids++;
						}
						if ((spin_idx[0] == spin_idx[3]) && (spin_idx[1] == spin_idx[2])) {
							cids[num_cids] = gint1_cids[idx];
							num_cids++;
						}
						if (num_cids == 0) {
							continue;
						}
						// note: operator ordering is a_{l,\nu} a_{k,\mu}
						struct spin_molecular_local_op_ref oplist[4] = {
							{ .isite = site_idx[0], .ispin = spin_idx[0], .oid = MOLECULAR_OID_C },
							{ .isite = site_idx[1], .ispin = spin_idx[1], .oid = MOLECULAR_OID_C },
							{ .isite = site_idx[3], .ispin = spin_idx[3], .oid = MOLECULAR_OID_A },
							{ .isite = site_idx[2], .ispin = spin_idx[2], .oid = MOLECULAR_OID_A }, };
						spin_molecular_mpo_graph_add_term(&vids, oplist, ARRLEN(oplist), cids, num_cids, edges);
					}
				}
			}
		}

		// transfer edges into mpo_graph structure and connect vertices
		assembly->graph.edges     = ct_calloc(nsites, sizeof(struct mpo_graph_edge*));
		assembly->graph.num_edges = ct_calloc(nsites, sizeof(int));
		for (int i = 0; i < nsites; i++)
		{
			assembly->graph.num_edges[i] = edges[i].size;
			assembly->graph.edges[i] = ct_malloc(edges[i].size * sizeof(struct mpo_graph_edge));
			struct linked_list_node* edge_ref = edges[i].head;
			int eid = 0;
			while (edge_ref != NULL)
			{
				const struct mpo_graph_edge* edge = edge_ref->data;
				memcpy(&assembly->graph.edges[i][eid], edge, sizeof(struct mpo_graph_edge));

				// create references from graph vertices to edge
				assert(0 <= edge->vids[0] && edge->vids[0] < assembly->graph.num_verts[i]);
				assert(0 <= edge->vids[1] && edge->vids[1] < assembly->graph.num_verts[i + 1]);
				mpo_graph_vertex_add_edge(1, eid, &assembly->graph.verts[i    ][edge->vids[0]]);
				mpo_graph_vertex_add_edge(0, eid, &assembly->graph.verts[i + 1][edge->vids[1]]);

				edge_ref = edge_ref->next;
				eid++;
			}
			// note: opics pointers of edges have been retained in transfer
			delete_linked_list(&edges[i], ct_free);
		}
		ct_free(edges);

		assert(mpo_graph_is_consistent(&assembly->graph));

		delete_spin_molecular_mpo_graph_vertices(&vids);
	}

	ct_free(gint1_cids);
	ct_free(gint0_cids);
	ct_free(tkin_cids);
	delete_dense_tensor(&gint1);
	delete_dense_tensor(&gint0);
}


//________________________________________________________________________________________________________________________
///
/// \brief Represent a product of sums of fermionic creation and annihilation operators of the following form as MPO:
/// \f[
/// H = \left(\sum_{i=1}^L \mathrm{coeffc}_i a^{\dagger}_i\right) \left(\sum_{j=1}^L \mathrm{coeffa}_j a_j\right)
/// \f]
///
void construct_quadratic_fermionic_mpo_assembly(const int nsites, const double* coeffc, const double* coeffa, struct mpo_assembly* assembly)
{
	assert(nsites >= 1);

	// physical quantum numbers (particle number)
	assembly->d = 2;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	assembly->qsite[0] = 0;
	assembly->qsite[1] = 1;

	assembly->dtype = CT_DOUBLE_REAL;

	// operator map
	assembly->num_local_ops = NUM_MOLECULAR_OID;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	create_molecular_hamiltonian_operator_map(assembly->opmap);

	// coefficient map
	assembly->num_coeffs = 2 + 3 * nsites;
	double* coeffmap = ct_malloc(assembly->num_coeffs * sizeof(double));
	// first two entries must always be 0 and 1
	coeffmap[0] = 0.;
	coeffmap[1] = 1.;
	int* coeffc_cids = ct_malloc(nsites * sizeof(int));
	int* coeffa_cids = ct_malloc(nsites * sizeof(int));
	int* coeffn_cids = ct_malloc(nsites * sizeof(int));
	int c = 2;
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffc[i];
		coeffc_cids[i] = c;
		c++;
	}
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffa[i];
		coeffa_cids[i] = c;
		c++;
	}
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffc[i] * coeffa[i];
		coeffn_cids[i] = c;
		c++;
	}
	assert(c == 2 + 3 * nsites);
	assembly->coeffmap = coeffmap;

	// construct operator graph

	assembly->graph.nsites = nsites;

	// vertices

	assembly->graph.verts     = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	assembly->graph.num_verts = ct_calloc(nsites + 1, sizeof(int));

	// identity chains from the left and right
	int* vids_identity_l = allocate_vertex_ids(nsites);
	int* vids_identity_r = allocate_vertex_ids(nsites);
	for (int i = 0; i < nsites; i++) {
		vids_identity_l[i] = assembly->graph.num_verts[i]++;
	}
	for (int i = 1; i < nsites + 1; i++) {
		vids_identity_r[i] = assembly->graph.num_verts[i]++;
	}
	// vertices connecting creation and annihilation operators
	int* vids_ca = allocate_vertex_ids(nsites);
	int* vids_ac = allocate_vertex_ids(nsites);
	for (int i = 1; i < nsites; i++)
	{
		vids_ca[i] = assembly->graph.num_verts[i]++;
		vids_ac[i] = assembly->graph.num_verts[i]++;
	}

	for (int i = 0; i < nsites + 1; i++)
	{
		assembly->graph.verts[i] = ct_calloc(assembly->graph.num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	// identity chains from the left and right: using default initialization with zeros
	// vertices connecting creation and annihilation operators
	for (int i = 1; i < nsites; i++)
	{
		assembly->graph.verts[i][vids_ca[i]].qnum =  1;  // connect creation with annihilation operator
		assembly->graph.verts[i][vids_ac[i]].qnum = -1;  // connect annihilation with creation operator
	}

	// edges

	// temporarily store edges in linked lists
	struct linked_list* edges = ct_calloc(nsites, sizeof(struct linked_list));

	// identities
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_identity_l[i + 1], MOLECULAR_OID_I, CID_ONE));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_r[i], vids_identity_r[i + 1], MOLECULAR_OID_I, CID_ONE));
	}
	// Z strings
	for (int i = 1; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ca[i], vids_ca[i + 1], MOLECULAR_OID_Z, CID_ONE));
	}
	for (int i = 1; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ac[i], vids_ac[i + 1], MOLECULAR_OID_Z, CID_ONE));
	}
	// number operators
	for (int i = 0; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_identity_r[i + 1], MOLECULAR_OID_N, coeffn_cids[i]));
	}
	// creation and annihilation operators
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_ca[i + 1], MOLECULAR_OID_C, coeffc_cids[i]));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ca[i], vids_identity_r[i + 1], MOLECULAR_OID_A, coeffa_cids[i]));
	}
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_ac[i + 1], MOLECULAR_OID_A, coeffa_cids[i]));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ac[i], vids_identity_r[i + 1], MOLECULAR_OID_C, coeffc_cids[i]));
	}

	ct_free(vids_ac);
	ct_free(vids_ca);
	ct_free(vids_identity_r);
	ct_free(vids_identity_l);

	ct_free(coeffn_cids);
	ct_free(coeffa_cids);
	ct_free(coeffc_cids);

	// transfer edges into mpo_graph structure and connect vertices
	assembly->graph.edges     = ct_calloc(nsites, sizeof(struct mpo_graph_edge*));
	assembly->graph.num_edges = ct_calloc(nsites, sizeof(int));
	for (int i = 0; i < nsites; i++)
	{
		assembly->graph.num_edges[i] = edges[i].size;
		assembly->graph.edges[i] = ct_malloc(edges[i].size * sizeof(struct mpo_graph_edge));
		struct linked_list_node* edge_ref = edges[i].head;
		int eid = 0;
		while (edge_ref != NULL)
		{
			const struct mpo_graph_edge* edge = edge_ref->data;
			memcpy(&assembly->graph.edges[i][eid], edge, sizeof(struct mpo_graph_edge));

			// create references from graph vertices to edge
			assert(0 <= edge->vids[0] && edge->vids[0] < assembly->graph.num_verts[i]);
			assert(0 <= edge->vids[1] && edge->vids[1] < assembly->graph.num_verts[i + 1]);
			mpo_graph_vertex_add_edge(1, eid, &assembly->graph.verts[i    ][edge->vids[0]]);
			mpo_graph_vertex_add_edge(0, eid, &assembly->graph.verts[i + 1][edge->vids[1]]);

			edge_ref = edge_ref->next;
			eid++;
		}
		// note: opics pointers of edges have been retained in transfer
		delete_linked_list(&edges[i], ct_free);
	}

	ct_free(edges);

	assert(mpo_graph_is_consistent(&assembly->graph));
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO assembly representation of a product of sums of fermionic creation and annihilation operators,
/// where sigma = 0 indicates spin-up and sigma = 1 indicates spin-down:
/// \f[
/// H = \left(\sum_{i=1}^L \mathrm{coeffc}_i a^{\dagger}_{i,\sigma}\right) \left(\sum_{j=1}^L \mathrm{coeffa}_j a_{j,\sigma}\right)
/// \f]
///
void construct_quadratic_spin_fermionic_mpo_assembly(const int nsites, const double* coeffc, const double* coeffa, const int sigma, struct mpo_assembly* assembly)
{
	assert(nsites >= 1);
	assert(sigma == 0 || sigma == 1);

	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = { 0,  1,  1,  2 };
	const qnumber qs[4] = { 0, -1,  1,  0 };
	assembly->d = 4;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	for (long i = 0; i < assembly->d; i++) {
		assembly->qsite[i] = encode_quantum_number_pair(qn[i], qs[i]);
	}

	assembly->dtype = CT_DOUBLE_REAL;

	// operator map
	const int OID_Id  =  0;  // identity
	const int OID_IC  =  1;  // a^{\dagger}_{\downarrow}
	const int OID_IA  =  2;  // a_{\downarrow}
	const int OID_IN  =  3;  // n_{\downarrow}
	const int OID_ZC  =  4;  // Z_{\uparrow} a^{\dagger}_{\downarrow}
	const int OID_ZA  =  5;  // Z_{\uparrow} a_{\downarrow}
	const int OID_CI  =  6;  // a^{\dagger}_{\uparrow}
	const int OID_AI  =  7;  // a_{\uparrow}
	const int OID_NI  =  8;  // n_{\uparrow}
	const int OID_CZ  =  9;  // a^{\dagger}_{\uparrow} Z_{\downarrow}
	const int OID_AZ  = 10;  // a_{\uparrow} Z_{\downarrow}
	const int OID_ZZ  = 11;  // Z_{\uparrow} Z_{\downarrow}
	assembly->num_local_ops = 12;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	struct dense_tensor opmap_single[NUM_MOLECULAR_OID];
	create_molecular_hamiltonian_operator_map(opmap_single);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_I], &opmap_single[MOLECULAR_OID_I], &assembly->opmap[OID_Id]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_I], &opmap_single[MOLECULAR_OID_C], &assembly->opmap[OID_IC]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_I], &opmap_single[MOLECULAR_OID_A], &assembly->opmap[OID_IA]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_I], &opmap_single[MOLECULAR_OID_N], &assembly->opmap[OID_IN]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_Z], &opmap_single[MOLECULAR_OID_C], &assembly->opmap[OID_ZC]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_Z], &opmap_single[MOLECULAR_OID_A], &assembly->opmap[OID_ZA]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_C], &opmap_single[MOLECULAR_OID_I], &assembly->opmap[OID_CI]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_A], &opmap_single[MOLECULAR_OID_I], &assembly->opmap[OID_AI]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_N], &opmap_single[MOLECULAR_OID_I], &assembly->opmap[OID_NI]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_C], &opmap_single[MOLECULAR_OID_Z], &assembly->opmap[OID_CZ]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_A], &opmap_single[MOLECULAR_OID_Z], &assembly->opmap[OID_AZ]);
	dense_tensor_kronecker_product(&opmap_single[MOLECULAR_OID_Z], &opmap_single[MOLECULAR_OID_Z], &assembly->opmap[OID_ZZ]);
	for (int i = 0; i < NUM_MOLECULAR_OID; i++) {
		delete_dense_tensor(&opmap_single[i]);
	}

	// coefficient map
	assembly->num_coeffs = 2 + 3 * nsites;
	double* coeffmap = ct_malloc(assembly->num_coeffs * sizeof(double));
	// first two entries must always be 0 and 1
	coeffmap[0] = 0.;
	coeffmap[1] = 1.;
	int* coeffc_cids = ct_malloc(nsites * sizeof(int));
	int* coeffa_cids = ct_malloc(nsites * sizeof(int));
	int* coeffn_cids = ct_malloc(nsites * sizeof(int));
	int c = 2;
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffc[i];
		coeffc_cids[i] = c;
		c++;
	}
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffa[i];
		coeffa_cids[i] = c;
		c++;
	}
	for (int i = 0; i < nsites; i++)
	{
		coeffmap[c] = coeffc[i] * coeffa[i];
		coeffn_cids[i] = c;
		c++;
	}
	assert(c == 2 + 3 * nsites);
	assembly->coeffmap = coeffmap;

	// construct operator graph

	assembly->graph.nsites = nsites;

	// vertices

	assembly->graph.verts     = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	assembly->graph.num_verts = ct_calloc(nsites + 1, sizeof(int));

	const qnumber qnum_spin[2] = { 1, -1 };

	// identity chains from the left and right
	int* vids_identity_l = allocate_vertex_ids(nsites);
	int* vids_identity_r = allocate_vertex_ids(nsites);
	for (int i = 0; i < nsites; i++) {
		vids_identity_l[i] = assembly->graph.num_verts[i]++;
	}
	for (int i = 1; i < nsites + 1; i++) {
		vids_identity_r[i] = assembly->graph.num_verts[i]++;
	}
	// vertices connecting creation and annihilation operators
	int* vids_ca = allocate_vertex_ids(nsites);
	int* vids_ac = allocate_vertex_ids(nsites);
	for (int i = 1; i < nsites; i++)
	{
		vids_ca[i] = assembly->graph.num_verts[i]++;
		vids_ac[i] = assembly->graph.num_verts[i]++;
	}

	for (int i = 0; i < nsites + 1; i++)
	{
		assembly->graph.verts[i] = ct_calloc(assembly->graph.num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	// identity chains from the left and right: using default initialization with zeros
	// vertices connecting creation and annihilation operators
	for (int i = 1; i < nsites; i++)
	{
		assembly->graph.verts[i][vids_ca[i]].qnum = encode_quantum_number_pair( 1, qnum_spin[sigma    ]);  // connect creation with annihilation operator
		assembly->graph.verts[i][vids_ac[i]].qnum = encode_quantum_number_pair(-1, qnum_spin[1 - sigma]);  // connect annihilation with creation operator
	}

	// edges

	// temporarily store edges in linked lists
	struct linked_list* edges = ct_calloc(nsites, sizeof(struct linked_list));

	// identities
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_identity_l[i + 1], OID_Id, CID_ONE));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_r[i], vids_identity_r[i + 1], OID_Id, CID_ONE));
	}
	// Z strings
	for (int i = 1; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ca[i], vids_ca[i + 1], OID_ZZ, CID_ONE));
	}
	for (int i = 1; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ac[i], vids_ac[i + 1], OID_ZZ, CID_ONE));
	}
	// number operators
	for (int i = 0; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_identity_r[i + 1], sigma == 0 ? OID_NI : OID_IN, coeffn_cids[i]));
	}
	// creation and annihilation operators
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_ca[i + 1], sigma == 0 ? OID_CZ : OID_IC, coeffc_cids[i]));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ca[i], vids_identity_r[i + 1], sigma == 0 ? OID_AI : OID_ZA, coeffa_cids[i]));
	}
	for (int i = 0; i < nsites - 1; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_identity_l[i], vids_ac[i + 1], sigma == 0 ? OID_AZ : OID_IA, coeffa_cids[i]));
	}
	for (int i = 1; i < nsites; i++) {
		linked_list_append(&edges[i], construct_mpo_graph_edge(vids_ac[i], vids_identity_r[i + 1], sigma == 0 ? OID_CI : OID_ZC, coeffc_cids[i]));
	}

	ct_free(vids_ac);
	ct_free(vids_ca);
	ct_free(vids_identity_r);
	ct_free(vids_identity_l);

	ct_free(coeffn_cids);
	ct_free(coeffa_cids);
	ct_free(coeffc_cids);

	// transfer edges into mpo_graph structure and connect vertices
	assembly->graph.edges     = ct_calloc(nsites, sizeof(struct mpo_graph_edge*));
	assembly->graph.num_edges = ct_calloc(nsites, sizeof(int));
	for (int i = 0; i < nsites; i++)
	{
		assembly->graph.num_edges[i] = edges[i].size;
		assembly->graph.edges[i] = ct_malloc(edges[i].size * sizeof(struct mpo_graph_edge));
		struct linked_list_node* edge_ref = edges[i].head;
		int eid = 0;
		while (edge_ref != NULL)
		{
			const struct mpo_graph_edge* edge = edge_ref->data;
			memcpy(&assembly->graph.edges[i][eid], edge, sizeof(struct mpo_graph_edge));

			// create references from graph vertices to edge
			assert(0 <= edge->vids[0] && edge->vids[0] < assembly->graph.num_verts[i]);
			assert(0 <= edge->vids[1] && edge->vids[1] < assembly->graph.num_verts[i + 1]);
			mpo_graph_vertex_add_edge(1, eid, &assembly->graph.verts[i    ][edge->vids[0]]);
			mpo_graph_vertex_add_edge(0, eid, &assembly->graph.verts[i + 1][edge->vids[1]]);

			edge_ref = edge_ref->next;
			eid++;
		}
		// note: opics pointers of edges have been retained in transfer
		delete_linked_list(&edges[i], ct_free);
	}

	ct_free(edges);

	assert(mpo_graph_is_consistent(&assembly->graph));
}
