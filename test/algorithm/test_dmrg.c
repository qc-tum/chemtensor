#include <math.h>
#include <complex.h>
#include "dmrg.h"
#include "chain_ops.h"
#include "aligned_memory.h"


char* test_dmrg_singlesite()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_dmrg_singlesite.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dmrg_singlesite failed";
	}

	// number of lattice sites
	const int nsites = 7;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "h_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mpo(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &hamiltonian);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPO tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(hamiltonian.a[i].dtype, hamiltonian.a[i].ndim, hamiltonian.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "h_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &hamiltonian.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mpo_is_consistent(&hamiltonian)) {
			return "internal MPO consistency check failed";
		}
	}

	// initial state vector as MPS
	struct mps psi;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "psi_start_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &psi);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPS tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "psi_start_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mps_is_consistent(&psi)) {
			return "internal MPS consistency check failed";
		}
	}

	// run DMRG

	const int num_sweeps = 6;
	const int maxiter_lanczos = 25;

	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));

	if (dmrg_singlesite(&hamiltonian, num_sweeps, maxiter_lanczos, &psi, en_sweeps) < 0) {
		return "'dmrg_singlesite' failed internally";
	}

	// compare with reference data

	// energies
	double* en_sweeps_ref = ct_malloc(num_sweeps * sizeof(double));
	if (read_hdf5_dataset(file, "en_sweeps", H5T_NATIVE_DOUBLE, en_sweeps_ref) < 0) {
		return "reading reference energies of DMRG sweeps from disk failed";
	}
	if (uniform_distance(CT_DOUBLE_REAL, num_sweeps, en_sweeps, en_sweeps_ref) > 1e-12) {
		return "reference energies of DMRG sweeps do not match reference";
	}

	// optimized state vector must be normalized
	if (fabs(mps_norm(&psi) - 1) > 1e-12) {
		return "optimized state vector is not normalized";
	}

	// optimized reference state vector as MPS
	struct mps psi_ref;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "psi_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &psi_ref);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPS tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(psi_ref.a[i].dtype, psi_ref.a[i].ndim, psi_ref.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "psi_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &psi_ref.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mps_is_consistent(&psi_ref)) {
			return "internal MPS consistency check failed";
		}
	}

	// overlap must have absolute value 1
	dcomplex overlap;
	mps_vdot(&psi, &psi_ref, &overlap);
	if (fabs(cabs(overlap) - 1) > 1e-13) {
		return "overlap between optimized and reference state vector must have absolute value 1";
	}

	ct_free(en_sweeps_ref);
	ct_free(en_sweeps);
	delete_mps(&psi_ref);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);
	ct_free(qsite);

	H5Fclose(file);

	return 0;
}


char* test_dmrg_twosite()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_dmrg_twosite.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dmrg_twosite failed";
	}

	// number of lattice sites
	const int nsites = 11;
	// local physical dimension
	const long d = 2;

	// physical quantum numbers
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "h_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mpo(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &hamiltonian);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPO tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(hamiltonian.a[i].dtype, hamiltonian.a[i].ndim, hamiltonian.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "h_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &hamiltonian.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mpo_is_consistent(&hamiltonian)) {
			return "internal MPO consistency check failed";
		}
	}

	// initial state vector as MPS
	struct mps psi;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "psi_start_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &psi);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPS tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "psi_start_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mps_is_consistent(&psi)) {
			return "internal MPS consistency check failed";
		}
	}

	// run DMRG

	const int num_sweeps = 4;
	const int maxiter_lanczos = 25;
	double tol_split;
	if (read_hdf5_attribute(file, "tol_split", H5T_NATIVE_DOUBLE, &tol_split) < 0) {
		return "reading splitting tolerance from disk failed";
	}
	const long max_vdim = ipow(d, nsites / 2);
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));

	if (dmrg_twosite(&hamiltonian, num_sweeps, maxiter_lanczos, tol_split, max_vdim, &psi, en_sweeps, entropy) < 0) {
		return "'dmrg_twosite' failed internally";
	}

	// compare with reference data

	// energies
	double* en_sweeps_ref = ct_malloc(num_sweeps * sizeof(double));
	if (read_hdf5_dataset(file, "en_sweeps", H5T_NATIVE_DOUBLE, en_sweeps_ref) < 0) {
		return "reading reference energies of DMRG sweeps from disk failed";
	}
	if (uniform_distance(CT_DOUBLE_REAL, num_sweeps, en_sweeps, en_sweeps_ref) > 1e-12) {
		return "reference energies of DMRG sweeps do not match reference";
	}

	// optimized state vector must be normalized
	if (fabs(mps_norm(&psi) - 1) > 1e-12) {
		return "optimized state vector is not normalized";
	}

	// optimized reference state vector as MPS
	struct mps psi_ref;
	{
		// virtual bond quantum numbers
		long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
		qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
		for (int i = 0; i < nsites + 1; i++)
		{
			char varname[1024];
			sprintf(varname, "psi_qbond%i", i);
			hsize_t qdims[1];
			if (get_hdf5_attribute_dims(file, varname, qdims) < 0) {
				return "reading virtual bond quantum number dimensions from disk failed";
			}
			dim_bonds[i] = qdims[0];
			qbonds[i] = ct_malloc(dim_bonds[i] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}

		allocate_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &psi_ref);

		for (int i = 0; i < nsites + 1; i++)
		{
			ct_free(qbonds[i]);
		}
		ct_free(qbonds);
		ct_free(dim_bonds);

		// read MPS tensors from disk
		for (int i = 0; i < nsites; i++)
		{
			// read dense tensors from disk
			struct dense_tensor a_dns;
			allocate_dense_tensor(psi_ref.a[i].dtype, psi_ref.a[i].ndim, psi_ref.a[i].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "psi_a%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &psi_ref.a[i]);

			delete_dense_tensor(&a_dns);
		}

		if (!mps_is_consistent(&psi_ref)) {
			return "internal MPS consistency check failed";
		}
	}

	// overlap must have absolute value 1
	dcomplex overlap;
	mps_vdot(&psi, &psi_ref, &overlap);
	if (fabs(cabs(overlap) - 1) > 1e-13) {
		return "overlap between optimized and reference state vector must have absolute value 1";
	}

	ct_free(entropy);
	ct_free(en_sweeps_ref);
	ct_free(en_sweeps);
	delete_mps(&psi_ref);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);
	ct_free(qsite);

	H5Fclose(file);

	return 0;
}
