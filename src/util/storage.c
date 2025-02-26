#include "storage.h"
#include "mps.h"
#include <hdf5.h>

int load_mps_hdf5(const char* filename, struct mps* mps) {
	return 0;
}

int save_mps_hdf5(const struct mps* mps, const char* filename) {
	herr_t status;

	hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hid_t root_group = H5Gopen1(file, "/");

	// attribute 'qsite': quantum numbers at each site
	{
		const hsize_t dims[1] = {mps->d};
		hid_t space = H5Screate_simple(1, dims, NULL);
		if (space < 0) {
			fprintf(stderr, "H5Screate_simple() failed for qsite, return value: %lld\n", space);
			return space;
		}

		hid_t attr = H5Acreate2(root_group, "qsite", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			fprintf(stderr, "H5Acreate() failed for qsite, return value: %lld\n", attr);
			return attr;
		}

		status = H5Awrite(attr, H5T_NATIVE_INT, mps->qsite);
		if (status < 0) {
			fprintf(stderr, "H5Awrite() failed for qsite, return value: %d\n", status);
			return status;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for qsite, return value: %d\n", status);
			return status;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for qsite, return value: %d\n", status);
			return status;
		}
	}

	// attribute 'd': physical dimension
	{
		hid_t space = H5Screate(H5S_SCALAR);
		if (space < 0) {
			fprintf(stderr, "H5Screate() failed for d, return value: %lld\n", space);
			return space;
		}

		hid_t attr = H5Acreate2(root_group, "d", H5T_NATIVE_LONG, space, H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			fprintf(stderr, "H5Acreate() failed for d, return value: %lld\n", attr);
			return attr;
		}

		status = H5Awrite(attr, H5T_NATIVE_LONG, &mps->d);
		if (status < 0) {
			fprintf(stderr, "H5Awrite() failed for d, return value: %d\n", status);
			return status;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for d, return value: %d\n", status);
			return status;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for d, return value: %d\n", status);
			return status;
		}
	}

	// attribute 'nsites': number of sites
	{
		hid_t space = H5Screate(H5S_SCALAR);
		if (space < 0) {
			fprintf(stderr, "H5Screate() failed for nsites, return value: %lld\n", space);
			return space;
		}

		hid_t attr = H5Acreate2(root_group, "nsites", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			fprintf(stderr, "H5Acreate() failed for nsites, return value: %lld\n", attr);
			return attr;
		}

		status = H5Awrite(attr, H5T_NATIVE_INT, &mps->nsites);
		if (status < 0) {
			fprintf(stderr, "H5Awrite() failed for nsites, return value: %d\n", status);
			return status;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for nsites, return value: %d\n", status);
			return status;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for nsites, return value: %d\n", status);
			return status;
		}
	}

	// dataspace: dense tensors for each site
	for (size_t site = 0; site < mps->nsites; site++) {
		struct dense_tensor dt;
		block_sparse_to_dense_tensor(&mps->a[site], &dt);

		delete_dense_tensor(&dt);
	}

	status = H5Gclose(root_group);
	if (status < 0) {
		fprintf(stderr, "H5Gclose() failed for root_group, return value: %d\n", status);
		return status;
	}

	status = H5Fclose(file);
	if (status < 0) {
		fprintf(stderr, "H5Fclose() failed for %s, return value: %d\n", filename, status);
		return status;
	}

	return 0;
}