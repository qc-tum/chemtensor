#include <math.h>
#include "truncation.h"
#include "rng.h"
#include "aligned_memory.h"
#include "hdf5_util.h"


char* test_retained_bond_indices()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_retained_bond_indices.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_retained_bond_indices failed";
	}

	// singular values
	hsize_t sigma_dims[1];
	if (get_hdf5_dataset_dims(file, "sigma", sigma_dims)) {
		return "reading number of singular values from disk failed";
	}
	const ct_long n = sigma_dims[0];
	double* sigma = ct_malloc(n * sizeof(double));
	if (read_hdf5_dataset(file, "sigma", H5T_NATIVE_DOUBLE, sigma) < 0) {
		return "reading singular values from disk failed";
	}

	// truncation tolerance
	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading truncation tolerance from disk failed";
	}

	// load reference data from disk
	hsize_t ind_ref_dims[1];
	if (get_hdf5_dataset_dims(file, "ind", ind_ref_dims)) {
		return "reading number of reference indices from disk failed";
	}
	ct_long* ind_ref = ct_malloc(ind_ref_dims[0] * sizeof(ct_long));
	if (read_hdf5_dataset(file, "ind", H5T_NATIVE_LONG, ind_ref) < 0) {
		return "reading reference indices from disk failed";
	}
	double norm_sigma_ref;
	if (read_hdf5_dataset(file, "norm_sigma", H5T_NATIVE_DOUBLE, &norm_sigma_ref) < 0) {
		return "reading reference norm of singular values from disk failed";
	}
	double entropy_ref;
	if (read_hdf5_dataset(file, "entropy", H5T_NATIVE_DOUBLE, &entropy_ref) < 0) {
		return "reading reference entropy from disk failed";
	}

	// two versions: truncation determined by tolerance, and truncation determined by length cut-off
	for (int i = 0; i < 2; i++)
	{
		struct index_list list;
		struct trunc_info info;
		retained_bond_indices(sigma, n, i == 0 ? tol : 1e-5, true, i == 0 ? n : (ct_long)ind_ref_dims[0], &list, &info);

		// compare indices
		if (list.num != (ct_long)ind_ref_dims[0]) {
			return "number of retained singular values does not match reference";
		}
		for (ct_long i = 0; i < list.num; i++) {
			if (list.ind[i] != ind_ref[i]) {
				return "indices of retained singular values do not match reference";
			}
		}
		// compare norm of retained singular values
		if (fabs(info.norm_sigma - norm_sigma_ref) > 1e-13) {
			return "norm of retained singular values does not match reference";
		}
		// compare von Neumann entropy
		if (fabs(info.entropy - entropy_ref) > 1e-13) {
			return "entropy of retained singular values does not match reference";
		}

		delete_index_list(&list);
	}

	ct_free(ind_ref);
	ct_free(sigma);

	H5Fclose(file);

	return 0;
}


char* test_retained_bond_indices_multiplicities()
{
	struct rng_state rng_state;
	seed_rng_state(92, &rng_state);

	const ct_long n = 37;

	// fictitious singular values
	double* sigma = ct_malloc(n * sizeof(double));
	for (ct_long i = 0; i < n; i++)
	{
		sigma[i] = randu(&rng_state) * exp2(-32. * randu(&rng_state));
	}
	// set some singular values to 0
	sigma[12] = 0;
	sigma[19] = 0;

	// special case of all multiplicities equal to 1, or generic case
	for (int im = 0; im < 2; im++)
	{
		int* multiplicities = ct_malloc(n * sizeof(int));
		for (ct_long i = 0; i < n; i++)
		{
			multiplicities[i] = (im == 0 ? 1 : 1 + rand_interval(13, &rng_state));
		}

		// logical dimension
		ct_long n_logical = 0;
		for (ct_long i = 0; i < n; i++) {
			n_logical += multiplicities[i];
		}

		// logical singular values
		double* sigma_logical = ct_malloc(n_logical * sizeof(double));
		ct_long c = 0;
		for (ct_long i = 0; i < n; i++) {
			for (int j = 0; j < multiplicities[i]; j++) {
				sigma_logical[c++] = sigma[i];
			}
		}
		assert(c == n_logical);

		// same singular value intervals due to multiplicities
		ct_long* intervals = ct_malloc((n + 1) * sizeof(ct_long));
		intervals[0] = 0;
		for (ct_long i = 0; i < n; i++) {
			intervals[i + 1] = intervals[i] + multiplicities[i];
		}
		assert(intervals[n] == n_logical);

		// truncation tolerance
		const double tol_list[3] = { 0, 1e-4, 5.0 };
		for (int itol = 0; itol < 3; itol++)
		{
			const double tol = tol_list[itol];

			for (int relative_thresh = 0; relative_thresh < 2; relative_thresh++)
			{
				// two cases of maximum virtual bond dimensions
				for (int imvd = 0; imvd < 2; imvd++)
				{
					const ct_long max_vdim = (imvd == 0 ? 2*n_logical : n_logical / 3);

					struct index_list list;
					struct trunc_info info;
					retained_bond_indices_multiplicities(sigma, multiplicities, n, tol, (bool)relative_thresh, max_vdim, &list, &info);

					// expand for comparison with reference
					struct index_list list_logical;
					list_logical.num = 0;
					list_logical.ind = ct_malloc(n_logical * sizeof(ct_long));  // upper bound on required memory
					for (ct_long i = 0; i < list.num; i++)
					{
						const ct_long idx = list.ind[i];
						for (ct_long j = intervals[idx]; j < intervals[idx + 1]; j++) {
							list_logical.ind[list_logical.num++] = j;
						}
					}

					// reference calculation
					struct index_list list_ref;
					struct trunc_info info_ref;
					retained_bond_indices(sigma_logical, n_logical, tol, (bool)relative_thresh, max_vdim, &list_ref, &info_ref);
					if (list_ref.num > 0)
					{
						// adapt to the closest complete multiplicity interval

						// indices in 'list_ref' need not cover a continuous integer range within a multiplicity interval

						double sigma_smallest_retained = sigma_logical[list_ref.ind[0]];
						for (ct_long i = 0; i < list_ref.num; i++) {
							sigma_smallest_retained = fmin(sigma_smallest_retained, sigma_logical[list_ref.ind[i]]);
						}
						// count appearances of smallest retained singular value
						int count_smallest_retained = 0;
						for (ct_long i = 0; i < list_ref.num; i++) {
							if (sigma_logical[list_ref.ind[i]] == sigma_smallest_retained) {
								count_smallest_retained++;
							}
						}
						// multiplicity of smallest retained singular value
						int multiplicity_smallest_retained = 0;
						for (ct_long i = 0; i < n_logical; i++) {
							if (sigma_logical[i] == sigma_smallest_retained) {
								multiplicity_smallest_retained++;
							}
						}

						if (count_smallest_retained < multiplicity_smallest_retained)
						{
							// try to enlarge
							ct_long max_vdim_eff = list_ref.num + multiplicity_smallest_retained - count_smallest_retained;
							if (max_vdim_eff > max_vdim)
							{
								// bound by maximum bond dimension -> remove current multiplicity interval
								max_vdim_eff -= multiplicity_smallest_retained;
							}

							// recompute (with tolerance set to zero)
							delete_index_list(&list_ref);
							retained_bond_indices(sigma_logical, n_logical, 0., (bool)relative_thresh, max_vdim_eff, &list_ref, &info_ref);
						}
					}

					// compare indices
					if (list_logical.num != list_ref.num) {
						return "number of retained singular values with multiplicities does not match reference";
					}
					for (ct_long i = 0; i < list_ref.num; i++) {
						if (list_logical.ind[i] != list_ref.ind[i]) {
							return "indices of retained singular values with multiplicities do not match reference";
						}
					}
					// compare norm of retained singular values
					if (fabs(info.norm_sigma - info_ref.norm_sigma) > 1e-13) {
						return "norm of retained singular values with multiplicities does not match reference";
					}
					// compare von Neumann entropy
					if (fabs(info.entropy - info_ref.entropy) > 1e-13) {
						return "entropy of retained singular values with multiplicities does not match reference";
					}

					delete_index_list(&list_ref);
					delete_index_list(&list_logical);
					delete_index_list(&list);
				}
			}
		}

		ct_free(intervals);
		ct_free(sigma_logical);
		ct_free(multiplicities);
	}

	ct_free(sigma);

	return 0;
}
