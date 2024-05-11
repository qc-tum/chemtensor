/// \file hamiltonian.c
/// \brief Construction of common quantum Hamiltonians.

#include <math.h>
#include <assert.h>
#include "hamiltonian.h"
#include "mpo_graph.h"
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
