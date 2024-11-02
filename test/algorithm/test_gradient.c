#include <complex.h>
#include "chain_ops.h"
#include "gradient.h"
#include "mpo.h"
#include "aligned_memory.h"
#include "numerical_gradient.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


struct operator_inner_product_params
{
	const struct mps* psi;
	const struct mps* chi;
	const struct mpo_graph* graph;
	struct dense_tensor* opmap;
	int num_local_ops;
	int num_coeffs;
};

// wrapper of 'mpo_inner_product' as a function of the operator coefficients
static void mpo_inner_product_wrapper(const scomplex* restrict x, void* p, scomplex* restrict y)
{
	const struct operator_inner_product_params* params = p;

	assert(params->num_coeffs >= 2);
	scomplex* coeffmap = ct_malloc(params->num_coeffs * sizeof(scomplex));
	// first two coefficients must always be 0 and 1
	coeffmap[0] = 0;
	coeffmap[1] = 1;
	// use 'x' for the remaining coefficients
	memcpy(coeffmap + 2, x, (params->num_coeffs - 2) * sizeof(scomplex));

	struct mpo_assembly assembly = {
		.graph         = *params->graph,
		.opmap         = params->opmap,
		.coeffmap      = coeffmap,
		.qsite         = params->psi->qsite,  // copy pointer
		.d             = params->psi->d,
		.dtype         = params->opmap[0].dtype,
		.num_local_ops = params->num_local_ops,
		.num_coeffs    = params->num_coeffs,
	};

	// construct corresponding MPO
	struct mpo mpo;
	mpo_from_assembly(&assembly, &mpo);

	mpo_inner_product(params->chi, &mpo, params->psi, y);

	delete_mpo(&mpo);
	ct_free(coeffmap);
}


char* test_operator_average_coefficient_gradient()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_operator_average_coefficient_gradient.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_operator_average_coefficient_gradient failed";
	}

	// number of lattice sites
	const int nsites = 4;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	const qnumber qsite[3] = { -1, 1, 0 };

	// virtual bond quantum numbers for 'psi'
	const long dim_bonds_psi[5] = { 1, 8, 23, 9, 1 };
	qnumber** qbonds_psi = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_psi[i] = ct_malloc(dim_bonds_psi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_psi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_psi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'chi'
	const long dim_bonds_chi[5] = { 1, 11, 15, 5, 1 };
	qnumber** qbonds_chi = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_chi[i] = ct_malloc(dim_bonds_chi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_chi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_chi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(CT_SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_psi, (const qnumber**)qbonds_psi, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	struct mps chi;
	allocate_mps(CT_SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_chi, (const qnumber**)qbonds_chi, &chi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(chi.a[i].dtype, chi.a[i].ndim, chi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "chi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &chi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&chi)) {
		return "internal MPS consistency check failed";
	}

	// construct an operator graph

	int v0_eids_1[] = { 0, 1, 2, 3 };
	int v1_eids_0[] = { 1, 3 };
	int v1_eids_1[] = { 0 };
	int v2_eids_0[] = { 0 };
	int v2_eids_1[] = { 1 };
	int v3_eids_0[] = { 2 };
	int v3_eids_1[] = { 2 };
	int v4_eids_0[] = { 0 };
	int v4_eids_1[] = { 0 };
	int v5_eids_0[] = { 1, 2 };
	int v5_eids_1[] = { 1 };
	int v6_eids_0[] = { 0, 1 };
	int v6_eids_1[] = { 0, 1, 2 };
	int v7_eids_0[] = { 0, 1, 2 };
	struct mpo_graph_vertex vertex_list[] = {
		{ .eids = { NULL,      v0_eids_1 }, .num_edges = { 0,                 ARRLEN(v0_eids_1) }, .qnum =  0 },
		{ .eids = { v1_eids_0, v1_eids_1 }, .num_edges = { ARRLEN(v1_eids_0), ARRLEN(v1_eids_1) }, .qnum = -1 },
		{ .eids = { v2_eids_0, v2_eids_1 }, .num_edges = { ARRLEN(v2_eids_0), ARRLEN(v2_eids_1) }, .qnum =  0 },
		{ .eids = { v3_eids_0, v3_eids_1 }, .num_edges = { ARRLEN(v3_eids_0), ARRLEN(v3_eids_1) }, .qnum =  1 },
		{ .eids = { v4_eids_0, v4_eids_1 }, .num_edges = { ARRLEN(v4_eids_0), ARRLEN(v4_eids_1) }, .qnum = -2 },
		{ .eids = { v5_eids_0, v5_eids_1 }, .num_edges = { ARRLEN(v5_eids_0), ARRLEN(v5_eids_1) }, .qnum =  0 },
		{ .eids = { v6_eids_0, v6_eids_1 }, .num_edges = { ARRLEN(v6_eids_0), ARRLEN(v6_eids_1) }, .qnum = -1 },
		{ .eids = { v7_eids_0, NULL      }, .num_edges = { ARRLEN(v7_eids_0), 0                 }, .qnum =  0 },
	};
	struct mpo_graph_vertex* graph_vertices[] = {
		&vertex_list[0],
		&vertex_list[1],
		&vertex_list[4],
		&vertex_list[6],
		&vertex_list[7],
	};
	int graph_num_vertices[5] = { 1, 3, 2, 1, 1 };

	struct local_op_ref  e0_opics[] = { { .oid = 6, .cid = 4 }, };
	struct local_op_ref  e1_opics[] = { { .oid = 5, .cid = 7 }, };
	struct local_op_ref  e2_opics[] = { { .oid = 2, .cid = 7 }, };
	struct local_op_ref  e3_opics[] = { { .oid = 4, .cid = 3 }, };
	struct local_op_ref  e4_opics[] = { { .oid = 5, .cid = 1 }, { .oid = 9, .cid = 3 }, };
	struct local_op_ref  e5_opics[] = { { .oid = 0, .cid = 5 }, };
	struct local_op_ref  e6_opics[] = { { .oid = 1, .cid = 2 }, };
	struct local_op_ref  e7_opics[] = { { .oid = 2, .cid = 6 }, };
	struct local_op_ref  e8_opics[] = { { .oid = 4, .cid = 5 }, { .oid = 1, .cid = 3 }, { .oid = 9, .cid = 3 }, };
	struct local_op_ref  e9_opics[] = { { .oid = 3, .cid = 2 }, };
	struct local_op_ref e10_opics[] = { { .oid = 7, .cid = 8 }, };
	struct local_op_ref e11_opics[] = { { .oid = 2, .cid = 7 }, };
	struct mpo_graph_edge edge_list[] = {
		{ .vids = { 0, 1 }, .opics =  e0_opics, .nopics = ARRLEN( e0_opics) },
		{ .vids = { 0, 0 }, .opics =  e1_opics, .nopics = ARRLEN( e1_opics) },
		{ .vids = { 0, 2 }, .opics =  e2_opics, .nopics = ARRLEN( e2_opics) },
		{ .vids = { 0, 0 }, .opics =  e3_opics, .nopics = ARRLEN( e3_opics) },
		{ .vids = { 0, 0 }, .opics =  e4_opics, .nopics = ARRLEN( e4_opics) },
		{ .vids = { 1, 1 }, .opics =  e5_opics, .nopics = ARRLEN( e5_opics) },
		{ .vids = { 2, 1 }, .opics =  e6_opics, .nopics = ARRLEN( e6_opics) },
		{ .vids = { 0, 0 }, .opics =  e7_opics, .nopics = ARRLEN( e7_opics) },
		{ .vids = { 1, 0 }, .opics =  e8_opics, .nopics = ARRLEN( e8_opics) },
		{ .vids = { 0, 0 }, .opics =  e9_opics, .nopics = ARRLEN( e9_opics) },
		{ .vids = { 0, 0 }, .opics = e10_opics, .nopics = ARRLEN(e10_opics) },
		{ .vids = { 0, 0 }, .opics = e11_opics, .nopics = ARRLEN(e11_opics) },
	};
	struct mpo_graph_edge* graph_edges[] = {
		&edge_list[0],
		&edge_list[4],
		&edge_list[7],
		&edge_list[9],
	};
	int graph_num_edges[4] = { 4, 3, 2, 3 };

	// construct MPO graph
	struct mpo_graph graph = {
		.verts     = graph_vertices,
		.edges     = graph_edges,
		.num_verts = graph_num_vertices,
		.num_edges = graph_num_edges,
		.nsites    = nsites
	};
	if (!mpo_graph_is_consistent(&graph)) {
		return "internal MPO graph construction is inconsistent";
	}

	// read local operator map from disk
	const int num_local_ops = 10;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", H5T_NATIVE_FLOAT, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = ct_malloc(num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim, &opmap[i]);
		const scomplex* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(scomplex));
	}
	delete_dense_tensor(&opmap_tensor);

	// coefficient map; first two entries must always be 0 and 1
	const int num_coeffs = 9;
	scomplex* coeffmap = ct_malloc(num_coeffs * sizeof(scomplex));
	if (read_hdf5_dataset(file, "coeffmap", H5T_NATIVE_FLOAT, coeffmap) < 0) {
		return "reading coefficient map from disk failed";
	}

	struct mpo_assembly assembly = {
		.graph         = graph,
		.opmap         = opmap,
		.coeffmap      = coeffmap,
		.qsite         = (qnumber*)qsite,
		.d             = d,
		.dtype         = CT_SINGLE_COMPLEX,
		.num_local_ops = num_local_ops,
		.num_coeffs    = num_coeffs,
	};

	// compute gradient with respect to coefficients
	scomplex avr;
	scomplex* dcoeff = ct_malloc(num_coeffs * sizeof(scomplex));
	operator_average_coefficient_gradient(&assembly, &psi, &chi, &avr, dcoeff);

	// reference average value
	scomplex avr_ref;
	if (read_hdf5_dataset(file, "avr", H5T_NATIVE_FLOAT, &avr_ref) < 0) {
		return "reading operator inner product reference value from disk failed";
	}

	// numerical gradient based on finite-difference approximation
	const float h = 1e-2;  // relatively large value to avoid numerical cancellation errors
	struct operator_inner_product_params params = {
		.psi           = &psi,
		.chi           = &chi,
		.graph         = &graph,
		.opmap         = opmap,
		.num_local_ops = num_local_ops,
		.num_coeffs    = num_coeffs,
	};
	scomplex* dcoeff_ref = ct_malloc((num_coeffs - 2) * sizeof(scomplex));
	scomplex dy = 1;
	numerical_gradient_backward_c(mpo_inner_product_wrapper, &params, num_coeffs - 2, coeffmap + 2, 1, &dy, h, dcoeff_ref);

	// compare average value
	if (cabsf(avr - avr_ref) / cabsf(avr_ref) > 1e-6) {
		return "MPO inner product does not match reference value";
	}

	// compare gradient (except for first two coefficients, which must be kept fixed at 0 and 1)
	if (uniform_distance(CT_SINGLE_COMPLEX, num_coeffs - 2, dcoeff + 2, dcoeff_ref) > 5e-5) {
		return "gradient with respect to coefficients computed by 'operator_average_coefficient_gradient' does not match finite difference approximation";
	}

	// clean up
	ct_free(dcoeff_ref);
	ct_free(dcoeff);
	ct_free(coeffmap);
	for (int i = 0; i < num_local_ops; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	ct_free(opmap);
	delete_mps(&chi);
	delete_mps(&psi);
	for (int i = 0; i < nsites + 1; i++) {
		ct_free(qbonds_chi[i]);
		ct_free(qbonds_psi[i]);
	}
	ct_free(qbonds_chi);
	ct_free(qbonds_psi);

	H5Fclose(file);

	return 0;
}
