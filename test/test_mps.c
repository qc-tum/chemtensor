#include <math.h>
#include <complex.h>
#include "mps.h"
#include "aligned_memory.h"


char* test_mps_vdot()
{
	hid_t file = H5Fopen("../test/data/test_mps_vdot.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_vdot failed";
	}

	// number of lattice sites
	const int nsites = 4;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers for 'psi'
	const long dim_bonds_psi[5] = { 1, 13, 17, 8, 1 };
	qnumber** qbonds_psi = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_psi[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds_psi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_psi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_psi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'chi'
	const long dim_bonds_chi[5] = { 1, 15, 20, 11, 1 };
	qnumber** qbonds_chi = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_chi[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds_chi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_chi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_chi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds_psi, (const qnumber**)qbonds_psi, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	struct mps chi;
	allocate_mps(DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds_chi, (const qnumber**)qbonds_chi, &chi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(chi.a[i].dtype, chi.a[i].ndim, chi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "chi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &chi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&chi)) {
		return "internal MPS consistency check failed";
	}

	// compute inner product
	dcomplex s;
	mps_vdot(&chi, &psi, &s);

	dcomplex s_ref;
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, &s_ref) < 0) {
		return "reading dot product reference value from disk failed";
	}

	// compare
	if (cabs(s - s_ref) / cabs(s_ref) > 1e-12) {
		return "inner product does not match reference value";
	}

	delete_mps(&chi);
	delete_mps(&psi);
	for (int i = 0; i < nsites + 1; i++)
	{
		aligned_free(qbonds_chi[i]);
		aligned_free(qbonds_psi[i]);
	}
	aligned_free(qbonds_chi);
	aligned_free(qbonds_psi);
	aligned_free(qsite);

	H5Fclose(file);

	return 0;
}


char* test_mps_orthonormalize_qr()
{
	hid_t file = H5Fopen("../test/data/test_mps_orthonormalize_qr.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_orthonormalize_qr failed";
	}

	// number of lattice sites
	const int nsites = 6;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers
	const long dim_bonds[7] = { 1, 4, 11, 9, 7, 3, 1 };
	qnumber** qbonds = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	for (int m = 0; m < 2; m++)
	{
		struct mps mps;
		allocate_mps(SINGLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &mps);

		// read MPS tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(mps.a[i].dtype, mps.a[i].ndim, mps.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &mps.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mps_is_consistent(&mps)) {
			return "internal MPS consistency check failed";
		}

		for (int i = 0; i < nsites + 1; i++) {
			if (mps_bond_dim(&mps, i) != dim_bonds[i]) {
				return "MPS virtual bond dimension does not match reference";
			}
		}

		// convert original MPS to state vector
		struct block_sparse_tensor vec_ref;
		mps_to_statevector(&mps, &vec_ref);

		double nrm = mps_orthonormalize_qr(&mps, m == 0 ? MPS_ORTHONORMAL_LEFT : MPS_ORTHONORMAL_RIGHT);

		if (!mps_is_consistent(&mps)) {
			return "internal MPS consistency check failed";
		}

		// convert normalized MPS to state vector
		struct block_sparse_tensor vec;
		mps_to_statevector(&mps, &vec);

		// must be normalized
		if (fabs(block_sparse_tensor_norm2(&vec) - 1) > 1e-6) {
			return "vector representation of MPS after orthonormalization does not have norm 1";
		}

		// scaled vector representation must agree with original vector
		float nrmf = (float)nrm;
		rscale_block_sparse_tensor(&nrmf, &vec);
		if (!block_sparse_tensor_allclose(&vec, &vec_ref, nrm * 1e-6)) {
			return "vector representation of MPS after orthonormalization does not match reference";
		}

		delete_block_sparse_tensor(&vec_ref);
		delete_block_sparse_tensor(&vec);
		delete_mps(&mps);
	}

	for (int i = 0; i < nsites + 1; i++)
	{
		aligned_free(qbonds[i]);
	}
	aligned_free(qbonds);
	aligned_free(qsite);

	H5Fclose(file);

	return 0;
}


char* test_mps_split_tensor_svd()
{
	hid_t file = H5Fopen("../test/data/test_mps_split_tensor_svd.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_split_tensor_svd failed";
	}

	// local physical dimensions
	const long d[2] = { 4, 5 };
	// virtual bond dimensions
	const long dim_bonds[2] = { 13, 17 };
	const long max_vdim = 100;

	// physical quantum numbers
	qnumber* qsite[2];
	for (int i = 0; i < 2; i++)
	{
		qsite[i] = aligned_alloc(MEM_DATA_ALIGN, d[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qsite%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qsite[i]) < 0) {
			return "reading physical quantum numbers from disk failed";
		}
	}
	// flattened quantum numbers
	qnumber* qsite_flat = aligned_alloc(MEM_DATA_ALIGN, d[0] * d[1] * sizeof(qnumber));
	for (long i = 0; i < d[0]; i++)
	{
		for (long j = 0; j < d[1]; j++)
		{
			qsite_flat[i*d[1] + j] = qsite[0][i] + qsite[1][j];
		}
	}

	// outer virtual bond quantum numbers
	qnumber* qbonds[2];
	for (int i = 0; i < 2; i++)
	{
		qbonds[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbonds%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	const long dim[3] = { dim_bonds[0], d[0]*d[1], dim_bonds[1] };

	// read dense tensor from disk
	struct dense_tensor a_pair_dns;
	allocate_dense_tensor(DOUBLE_REAL, 3, dim, &a_pair_dns);
	if (read_hdf5_dataset(file, "a_pair", H5T_NATIVE_DOUBLE, a_pair_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
	const qnumber* qnums[3] = { qbonds[0], qsite_flat, qbonds[1] };

	// convert dense to block-sparse tensor
	struct block_sparse_tensor a_pair;
	dense_to_block_sparse_tensor(&a_pair_dns, axis_dir, qnums, &a_pair);

	delete_dense_tensor(&a_pair_dns);

	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading tolerance from disk failed";
	}

	struct dense_tensor a_mrg_ref;
	allocate_dense_tensor(DOUBLE_REAL, 3, dim, &a_mrg_ref);
	if (read_hdf5_dataset(file, "a_mrg", H5T_NATIVE_DOUBLE, a_mrg_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// without or with truncation
	for (int i = 0; i < 2; i++)
	{
		// singular value distribution mode
		for (int j = 0; j < 2; j++)
		{
			const enum singular_value_distr svd_distr = (j == 0 ? SVD_DISTR_LEFT : SVD_DISTR_RIGHT);

			struct trunc_info info;
			struct block_sparse_tensor a0, a1;
			if (mps_split_tensor_svd(&a_pair, d, (const qnumber**)qsite, i == 0 ? 0. : tol, max_vdim, false, svd_distr, &a0, &a1, &info) < 0) {
				return "'mps_split_tensor_svd' failed internally";
			}

			struct block_sparse_tensor a_mrg;
			mps_merge_tensor_pair(&a0, &a1, &a_mrg);

			if (i == 0)
			{
				// merged tensor must agree with the original tensor
				if (!block_sparse_tensor_allclose(&a_mrg, &a_pair, 1e-13)) {
					return "merged MPS tensor after splitting does not match original pair tensor";
				}
			}
			else
			{
				struct dense_tensor a_mrg_dns;
				block_sparse_to_dense_tensor(&a_mrg, &a_mrg_dns);

				// compare
				if (!dense_tensor_allclose(&a_mrg_dns, &a_mrg_ref, 1e-13)) {
					return "merged tensor after splitting does not match reference";
				}

				delete_dense_tensor(&a_mrg_dns);
			}

			delete_block_sparse_tensor(&a_mrg);
			delete_block_sparse_tensor(&a1);
			delete_block_sparse_tensor(&a0);
		}
	}

	delete_dense_tensor(&a_mrg_ref);
	delete_block_sparse_tensor(&a_pair);
	for (int i = 0; i < 2; i++)
	{
		aligned_free(qbonds[i]);
		aligned_free(qsite[i]);
	}
	aligned_free(qsite_flat);

	H5Fclose(file);

	return 0;
}


char* test_mps_to_statevector()
{
	hid_t file = H5Fopen("../test/data/test_mps_to_statevector.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_to_statevector failed";
	}

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers
	const long dim_bonds[6] = { 1, 7, 10, 11, 5, 1 };
	qnumber** qbonds = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps mps;
	allocate_mps(DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &mps);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(mps.a[i].dtype, mps.a[i].ndim, mps.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &mps.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&mps)) {
		return "internal MPS consistency check failed";
	}

	for (int i = 0; i < nsites + 1; i++) {
		if (mps_bond_dim(&mps, i) != dim_bonds[i]) {
			return "MPS virtual bond dimension does not match reference";
		}
	}

	// convert to state vector
	struct block_sparse_tensor vec;
	mps_to_statevector(&mps, &vec);

	// convert to dense vector (for comparison with reference vector)
	struct dense_tensor vec_dns;
	block_sparse_to_dense_tensor(&vec, &vec_dns);

	// read reference state vector from disk
	struct dense_tensor vec_ref;
	const long dim_vec_ref[3] = { 1, ipow(d, nsites), 1 };  // include dummy virtual bond dimensions
	allocate_dense_tensor(DOUBLE_COMPLEX, 3, dim_vec_ref, &vec_ref);
	if (read_hdf5_dataset(file, "vec", H5T_NATIVE_DOUBLE, vec_ref.data) < 0) {
		return "reading state vector entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&vec_dns, &vec_ref, 1e-13)) {
		return "state vector obtained from MPS does not match reference";
	}

	delete_dense_tensor(&vec_ref);
	delete_dense_tensor(&vec_dns);
	delete_block_sparse_tensor(&vec);
	delete_mps(&mps);
	for (int i = 0; i < nsites + 1; i++)
	{
		aligned_free(qbonds[i]);
	}
	aligned_free(qbonds);
	aligned_free(qsite);

	H5Fclose(file);

	return 0;
}
