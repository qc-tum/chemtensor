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
			return -1;
		}

		hid_t attr = H5Acreate2(root_group, "qsite", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			fprintf(stderr, "H5Acreate() failed for qsite, return value: %lld\n", attr);
			return -1;
		}

		status = H5Awrite(attr, H5T_NATIVE_INT, mps->qsite);
		if (status < 0) {
			fprintf(stderr, "H5Awrite() failed for qsite, return value: %d\n", status);
			return -1;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for qsite, return value: %d\n", status);
			return -1;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for qsite, return value: %d\n", status);
			return -1;
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
			return -1;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for d, return value: %d\n", status);
			return -1;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for d, return value: %d\n", status);
			return -1;
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
			return -1;
		}

		status = H5Aclose(attr);
		if (status < 0) {
			fprintf(stderr, "H5Aclose() failed for nsites, return value: %d\n", status);
			return -1;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for nsites, return value: %d\n", status);
			return -1;
		}
	}

	// dataspace: dense tensors for each site
	for (size_t site = 0; site < mps->nsites; site++) {
		struct dense_tensor dt;
		struct block_sparse_tensor* bst = &mps->a[site];
		block_sparse_to_dense_tensor(bst, &dt);

		char dataset_name[100];
		sprintf(dataset_name, "tensor-%zu", site);

		hid_t datatype;
		switch (dt.dtype) {
		case CT_SINGLE_REAL:
			datatype = H5T_NATIVE_FLOAT;
			break;
		case CT_DOUBLE_REAL:
			datatype = H5T_NATIVE_DOUBLE;
			break;
		default:
			fprintf(stderr, "Not implemented yet.\n");
			return -1;
		}

		// todo: casting long* to hsize_t* might cause troubles on some systems (?)
		hid_t space = H5Screate_simple(dt.ndim, (const hsize_t*)dt.dim, NULL);
		if (space < 0) {
			fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", dataset_name, space);
			return -1;
		}

		hid_t dataset = H5Dcreate(root_group, dataset_name, datatype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (dataset < 0) {
			fprintf(stderr, "H5Dcreate() failed for %s, return value: %d\n", dataset_name, status);
			return -1;
		}

		status = H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dt.data);
		if (status < 0) {
			fprintf(stderr, "H5Dwrite() failed for %s, return value: %d\n", dataset_name, status);
			return -1;
		}

		// attribute 'axis_dir'
		{
			const hsize_t dims[1] = {bst->ndim};
			hid_t space = H5Screate_simple(1, dims, NULL);
			if (space < 0) {
				fprintf(stderr, "H5Screate_simple() failed for axis_dir, return value: %lld\n", space);
				return -1;
			}

			hid_t attr = H5Acreate2(dataset, "axis_dir", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
			if (attr < 0) {
				fprintf(stderr, "H5Acreate() failed for axis_dir, return value: %lld\n", attr);
				return -1;
			}

			status = H5Awrite(attr, H5T_NATIVE_INT, bst->axis_dir);
			if (status < 0) {
				fprintf(stderr, "H5Awrite() failed for axis_dir, return value: %d\n", status);
				return -1;
			}

			status = H5Aclose(attr);
			if (status < 0) {
				fprintf(stderr, "H5Aclose() failed for axis_dir, return value: %d\n", status);
				return -1;
			}

			status = H5Sclose(space);
			if (status < 0) {
				fprintf(stderr, "H5Sclose() failed for axis_dir, return value: %d\n", status);
				return -1;
			}
		}

		// attributes 'qnums-{ndim}'
		for (size_t i = 0; i < bst->ndim; i++) {
			char attr_qnums_name[100];
			sprintf(attr_qnums_name, "qnums-%zu", i);

			const hsize_t dims[1] = {bst->dim_blocks[i]};
			hid_t space = H5Screate_simple(1, dims, NULL);
			if (space < 0) {
				fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", attr_qnums_name, space);
				return -1;
			}

			hid_t attr = H5Acreate2(dataset, attr_qnums_name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
			if (attr < 0) {
				fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", attr_qnums_name, attr);
				return -1;
			}

			status = H5Awrite(attr, H5T_NATIVE_INT, bst->qnums_logical[i]);
			if (status < 0) {
				fprintf(stderr, "H5Awrite() failed for %s, return value: %d\n", attr_qnums_name, status);
				return -1;
			}

			status = H5Aclose(attr);
			if (status < 0) {
				fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", attr_qnums_name, status);
				return -1;
			}

			status = H5Sclose(space);
			if (status < 0) {
				fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", attr_qnums_name, status);
				return -1;
			}
		}

		status = H5Dclose(dataset);
		if (status < 0) {
			fprintf(stderr, "H5Dclose() failed for %s, return value: %d\n", dataset_name, status);
			return -1;
		}

		status = H5Sclose(space);
		if (status < 0) {
			fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", dataset_name, status);
			return -1;
		}

		delete_dense_tensor(&dt);
	}

	status = H5Gclose(root_group);
	if (status < 0) {
		fprintf(stderr, "H5Gclose() failed for root_group, return value: %d\n", status);
		return -1;
	}

	status = H5Fclose(file);
	if (status < 0) {
		fprintf(stderr, "H5Fclose() failed for %s, return value: %d\n", filename, status);
		return -1;
	}

	return 0;
}