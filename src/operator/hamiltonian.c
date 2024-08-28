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
		assembly->qsite[i] = fermi_hubbard_encode_quantum_numbers(qn[i], qs[i]);
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
	int oids_c0[] = { OID_CZ, OID_AI };  qnumber qnums_c0[] = { 0, fermi_hubbard_encode_quantum_numbers( 1,  1), 0 };
	int oids_c1[] = { OID_AZ, OID_CI };  qnumber qnums_c1[] = { 0, fermi_hubbard_encode_quantum_numbers(-1, -1), 0 };
	// spin-down kinetic hopping
	int oids_c2[] = { OID_IC, OID_ZA };  qnumber qnums_c2[] = { 0, fermi_hubbard_encode_quantum_numbers( 1, -1), 0 };
	int oids_c3[] = { OID_IA, OID_ZC };  qnumber qnums_c3[] = { 0, fermi_hubbard_encode_quantum_numbers(-1,  1), 0 };
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
	MOLECULAR_OID_I = 0,  //!< identity
	MOLECULAR_OID_C = 1,  //!< a^{\dagger}
	MOLECULAR_OID_A = 2,  //!< a
	MOLECULAR_OID_N = 3,  //!< numop
	MOLECULAR_OID_Z = 4,  //!< Z
};


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
	int** a_dag[2];        //!< a^{\dagger}_i
	int** a_ann[2];        //!< a_i
	int** a_dag_a_dag[2];  //!< a^{\dagger}_i a^{\dagger}_j
	int** a_ann_a_ann[2];  //!< a_i a_j
	int** a_dag_a_ann[2];  //!< a^{\dagger}_i a_j
	int   nsites;          //!< number of sites
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
		// a^{\dagger}_i a^{\dagger}_j (for i < j), a_i a_j (for i < j) and a^{\dagger}_i a_j chains, extending from boundary to center
		int chi2 = n * (n - 1) + n * n;
		// a^{\dagger}_i and a_i chains, reaching (almost) from one boundary to the other
		int chi3 = 2 * ((i < nsites - 1 ? nl : 0) + (i > 1 ? nr : 0));

		graph->num_verts[i] = chi1 + chi2 + chi3;
		graph->verts[i] = ct_calloc(graph->num_verts[i], sizeof(struct mpo_graph_vertex));
	}

	vids->nsites = nsites;

	int* vertex_counter = ct_calloc(nsites + 1, sizeof(int));

	// vertex IDs

	// identity chains from the left and right
	vids->identity[MOLECULAR_CONN_LEFT] = allocate_vertex_ids(nsites);
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

	// a^{\dagger}_i a^{\dagger}_j operators connected to left boundary
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

	// a_i a_j operators connected to left boundary
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

	// a^{\dagger}_i a^{\dagger}_j operators connected to right boundary
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

	// a_i a_j operators connected to right boundary
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

	// identities connected to left and right terminals
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

	// local operators
	// creation and annihilation operators for a single spin and lattice site
	const double a_ann[4] = { 0.,  1.,  0.,  0. };
	const double a_dag[4] = { 0.,  0.,  1.,  0. };
	// number operator
	const double numop[4] = { 0.,  0.,  0.,  1. };
	// Pauli-Z matrix required for Jordan-Wigner transformation
	const double z[4]     = { 1.,  0.,  0., -1. };

	// operator map
	assembly->num_local_ops = 5;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < assembly->num_local_ops; i++) {
		const long dim[2] = { assembly->d, assembly->d };
		allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
	}
	dense_tensor_set_identity(&assembly->opmap[MOLECULAR_OID_I]);
	memcpy(assembly->opmap[MOLECULAR_OID_C].data, a_dag, sizeof(a_dag));
	memcpy(assembly->opmap[MOLECULAR_OID_A].data, a_ann, sizeof(a_ann));
	memcpy(assembly->opmap[MOLECULAR_OID_N].data, numop, sizeof(numop));
	memcpy(assembly->opmap[MOLECULAR_OID_Z].data, z,     sizeof(z));

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
			long eid = 0;
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
