#include "storage.h"
#include "aligned_memory.h"
#include "mps.h"
#include <hdf5.h>

/// \brief Create HDF5 enumeration datatype for `tensor_axis_direction` and return its id.
hid_t get_axis_dir_enum_dtype();

int load_mps_hdf5(const char* filename, struct mps* mps) {
	herr_t status;

	hid_t file;
	if ((file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Fopen() failed for %s, return value: %lld\n", filename, file);
		return -1;
	}

	hid_t root_group;
	if ((root_group = H5Gopen1(file, "/")) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Gopen1() failed for /, return value: %lld\n", file);
		return -1;
	}

	long d; // attribute 'd': physical dimension
	{
		hid_t attr;
		if ((attr = H5Aopen(root_group, "d", H5P_DEFAULT)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Acreate() failed for d, return value: %lld\n", attr);
			return -1;
		}

		if ((status = H5Aread(attr, H5T_NATIVE_LONG, &d)) < 0) {
			fprintf(stderr, "H5Aread() failed for d, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for d, return value: %d\n", status);
			return -1;
		}
	}

	int nsites; // attribute 'nsites': number of sites
	{
		hid_t attr;
		if ((attr = H5Aopen(root_group, "nsites", H5P_DEFAULT)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Acreate() failed for d, return value: %lld\n", attr);
			return -1;
		}

		if ((status = H5Aread(attr, H5T_NATIVE_INT, &nsites)) < 0) {
			fprintf(stderr, "H5Aread() failed for d, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for d, return value: %d\n", status);
			return -1;
		}
	}

	qnumber qsite[d]; // attribute 'qsite': quantum numbers at each site
	{
		hid_t attr;
		if ((attr = H5Aopen(root_group, "qsite", H5P_DEFAULT)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Aopen() failed for qsite, return value: %lld\n", attr);
			return -1;
		}

		if ((status = H5Aread(attr, H5T_NATIVE_INT, (void*)&qsite)) < 0) {
			fprintf(stderr, "H5Aread() failed for qsite, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for qsite, return value: %d\n", status);
			return -1;
		}
	}

	allocate_empty_mps(nsites, d, (const qnumber*)&qsite, mps);

	for (size_t site = 0; site < nsites; site++) {
		char dset_name[128];
		sprintf(dset_name, "tensor-%zu", site);

		hid_t dset;
		if ((dset = H5Dopen2(file, dset_name, H5P_DEFAULT)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dopen2() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// extract dtype and convert to chemtensor numeric_type
		hid_t dtype;
		enum numeric_type dt_dtype;
		{
			if ((dtype = H5Dget_type(dset)) == H5I_INVALID_HID) {
				fprintf(stderr, "H5Dget_type() failed for %s, return value: %d\n", dset_name, status);
				return -1;
			}

			if (H5Tequal(dtype, H5T_NATIVE_FLOAT)) {
				dt_dtype = CT_SINGLE_REAL;
			} else if (H5Tequal(dtype, H5T_NATIVE_DOUBLE)) {
				dt_dtype = CT_DOUBLE_REAL;
			} else {
				fprintf(stderr, "Invalid dtype: %lld.\n", dtype);
				return -1;
			}
		}

		hid_t dset_space;
		if ((dset_space = H5Dget_space(dset)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dget_space() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		int ndim;
		if ((ndim = H5Sget_simple_extent_ndims(dset_space)) < 0) {
			fprintf(stderr, "H5Sget_simple_extent_ndims() failed for %s, return value: %d\n", dset_name, ndim);
			return -1;
		}

		hsize_t dims[ndim];
		if ((ndim = H5Sget_simple_extent_dims(dset_space, (hsize_t*)&dims, NULL)) < 0) {
			fprintf(stderr, "H5Sget_simple_extent_ndims() failed for %s, return value: %d\n", dset_name, ndim);
			return -1;
		}

		// dset_space not required anymore, hence, close handle
		if ((status = H5Sclose(dset_space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", dset_name, status);
		}

		struct dense_tensor dt;
		allocate_dense_tensor(dt_dtype, ndim, (const long*)&dims, &dt);

		if (H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dt.data) < 0) {
			fprintf(stderr, "H5Dread() failed for %s, return value: %d\n", dset_name, ndim);
			return -1;
		}

		// dtype not required anymore, hence, close handle.
		if ((status = H5Tclose(dtype)) < 0) {
			fprintf(stderr, "H5Tclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		enum tensor_axis_direction axis_dir[ndim]; // attribute 'axis_dir'
		{
			hid_t attr, enum_dtype;
			if ((attr = H5Aopen(dset, "axis_dir", H5P_DEFAULT)) == H5I_INVALID_HID) {
				fprintf(stderr, "H5Aopen() failed for axis_dir, return value: %lld\n", attr);
				return -1;
			}

			if ((enum_dtype = get_axis_dir_enum_dtype()) < 0) {
				return -1; // function already prints error message
			}

			if ((status = H5Aread(attr, enum_dtype, &axis_dir)) < 0) {
				fprintf(stderr, "H5Aread() failed for axis_dir, return value: %d\n", status);
				return -1;
			}

			if ((status = H5Tclose(enum_dtype)) < 0) {
				fprintf(stderr, "H5Tclose() failed for axis_dir(enum_dtype), return value: %d\n", status);
				return -1;
			}

			if ((status = H5Aclose(attr)) < 0) {
				fprintf(stderr, "H5Aclose() failed for axis_dir, return value: %d\n", status);
				return -1;
			}
		}

		qnumber** qnums = ct_calloc(ndim, sizeof(qnumber*)); // attribute qnums
		for (size_t i = 0; i < ndim; i++) {
			char qnums_name[128];
			sprintf(qnums_name, "qnums-%zu", i);

			hid_t attr;
			if ((attr = H5Aopen(dset, qnums_name, H5P_DEFAULT)) == H5I_INVALID_HID) {
				fprintf(stderr, "H5Aopen() failed for %s, return value: %lld\n", qnums_name, attr);
				return -1;
			}

			// retrieve dataspace information for qnums
			// dataspace for qnums is one-dimensional, hence, scalar variable
			hsize_t qnums_dim;
			{
				hid_t space_qnums;
				if ((space_qnums = H5Aget_space(attr)) == H5I_INVALID_HID) {
					fprintf(stderr, "H5Dget_space() failed for %s, return value: %d\n", qnums_name, status);
					return -1;
				}

				int rank;
				if ((rank = H5Sget_simple_extent_dims(space_qnums, &qnums_dim, NULL)) < 0) {
					fprintf(stderr, "H5Sget_simple_extent_ndims() failed for %s, return value: %d\n", qnums_name, rank);
					return -1;
				}

				assert(rank == 1);

				if ((status = H5Sclose(space_qnums)) < 0) {
					fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", qnums_name, status);
					return -1;
				}
			}

			qnums[i] = ct_malloc(qnums_dim * sizeof(qnumber));
			if ((status = H5Aread(attr, H5T_NATIVE_INT, qnums[i])) < 0) {
				fprintf(stderr, "H5Aread() failed for %s, return value: %d\n", qnums_name, status);
				return -1;
			}

			if ((status = H5Aclose(attr)) < 0) {
				fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", qnums_name, status);
				return -1;
			}
		}

		dense_to_block_sparse_tensor(&dt, axis_dir, (const qnumber**)qnums, &mps->a[site]);

		if ((status = H5Dclose(dset)) < 0) {
			fprintf(stderr, "H5Dclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}
	}

	if ((status = H5Gclose(root_group)) < 0) {
		fprintf(stderr, "H5Gclose() failed for root_group, return value: %d\n", status);
		return -1;
	}

	if ((status = H5Fclose(file)) < 0) {
		fprintf(stderr, "H5Fclose() failed for %s, return value: %d\n", filename, status);
		return -1;
	}

	return 0;
}

int save_mps_hdf5(const struct mps* mps, const char* filename) {
	herr_t status;
	hid_t file, root_group, space, attr, dtype, dset;

	if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Fcreate() failed for %s, return value: %lld\n", filename, file);
		return -1;
	}

	if ((root_group = H5Gopen1(file, "/")) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Gopen1() failed for /, return value: %lld\n", file);
		return -1;
	}

	// attribute 'd': physical dimension
	{
		if ((space = H5Screate(H5S_SCALAR)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Screate() failed for d, return value: %lld\n", space);
			return space;
		}

		if ((attr = H5Acreate2(root_group, "d", H5T_NATIVE_LONG, space, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
			fprintf(stderr, "H5Acreate() failed for d, return value: %lld\n", attr);
			return attr;
		}

		if ((status = H5Awrite(attr, H5T_NATIVE_LONG, &mps->d)) < 0) {
			fprintf(stderr, "H5Awrite() failed for d, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for d, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Sclose(space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for d, return value: %d\n", status);
			return -1;
		}
	}

	// attribute 'nsites': number of sites
	{
		if ((space = H5Screate(H5S_SCALAR)) < 0) {
			fprintf(stderr, "H5Screate() failed for nsites, return value: %lld\n", space);
			return space;
		}

		if ((attr = H5Acreate2(root_group, "nsites", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
			fprintf(stderr, "H5Acreate() failed for nsites, return value: %lld\n", attr);
			return attr;
		}

		if ((status = H5Awrite(attr, H5T_NATIVE_INT, &mps->nsites)) < 0) {
			fprintf(stderr, "H5Awrite() failed for nsites, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for nsites, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Sclose(space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for nsites, return value: %d\n", status);
			return -1;
		}
	}

	// attribute 'qsite': quantum numbers at each site
	{
		if ((space = H5Screate_simple(1, (hsize_t[]){mps->d}, NULL)) < 0) {
			fprintf(stderr, "H5Screate_simple() failed for qsite, return value: %lld\n", space);
			return -1;
		}

		if ((attr = H5Acreate2(root_group, "qsite", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
			fprintf(stderr, "H5Acreate() failed for qsite, return value: %lld\n", attr);
			return -1;
		}

		if ((status = H5Awrite(attr, H5T_NATIVE_INT, mps->qsite)) < 0) {
			fprintf(stderr, "H5Awrite() failed for qsite, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Aclose(attr)) < 0) {
			fprintf(stderr, "H5Aclose() failed for qsite, return value: %d\n", status);
			return -1;
		}

		if ((status = H5Sclose(space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for qsite, return value: %d\n", status);
			return -1;
		}
	}

	// dataspace: dense tensors for each site
	for (size_t site = 0; site < mps->nsites; site++) {
		char dset_name[128];
		sprintf(dset_name, "tensor-%zu", site);

		struct dense_tensor dt;
		struct block_sparse_tensor* bst = &mps->a[site];
		block_sparse_to_dense_tensor(bst, &dt);

		switch (dt.dtype) {
		case CT_SINGLE_REAL:
			dtype = H5T_NATIVE_FLOAT;
			break;
		case CT_DOUBLE_REAL:
			dtype = H5T_NATIVE_DOUBLE;
			break;
		case CT_SINGLE_COMPLEX:
		case CT_DOUBLE_COMPLEX:
		default:
			fprintf(stderr, "Invalid dtype: %lld.\n", dtype);
			return -1;
		}

		// todo: casting long* to hsize_t* might cause troubles on some systems (?)
		if ((space = H5Screate_simple(dt.ndim, (const hsize_t*)dt.dim, NULL)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", dset_name, space);
			return -1;
		}

		dset = H5Dcreate(root_group, dset_name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (dset == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dcreate() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		if ((status = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dt.data)) < 0) {
			fprintf(stderr, "H5Dwrite() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// attribute 'axis_dir'
		{
			hid_t space_axis, enum_dtype;
			if ((space_axis = H5Screate_simple(1, (hsize_t[]){bst->ndim}, NULL)) < 0) {
				fprintf(stderr, "H5Screate_simple() failed for axis_dir, return value: %lld\n", space_axis);
				return -1;
			}

			if ((enum_dtype = get_axis_dir_enum_dtype()) < 0) {
				return -1; // function already prints error message
			}

			if ((attr = H5Acreate2(dset, "axis_dir", enum_dtype, space_axis, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
				fprintf(stderr, "H5Acreate() failed for axis_dir, return value: %lld\n", attr);
				return -1;
			}

			if ((status = H5Awrite(attr, enum_dtype, bst->axis_dir)) < 0) {
				fprintf(stderr, "H5Awrite() failed for axis_dir, return value: %d\n", status);
				return -1;
			}

			if ((status = H5Tclose(enum_dtype)) < 0) {
				fprintf(stderr, "H5Tclose() failed for axis_dir(enum_dtype), return value: %d\n", status);
				return -1;
			}

			if ((status = H5Aclose(attr)) < 0) {
				fprintf(stderr, "H5Aclose() failed for axis_dir, return value: %d\n", status);
				return -1;
			}

			if ((status = H5Sclose(space_axis)) < 0) {
				fprintf(stderr, "H5Sclose() failed for axis_dir, return value: %d\n", status);
				return -1;
			}
		}

		// attributes 'qnums-{ndim}'
		hid_t space_qnums;
		for (size_t i = 0; i < bst->ndim; i++) {
			char qnums_name[128];
			sprintf(qnums_name, "qnums-%zu", i);

			if ((space_qnums = H5Screate_simple(1, (hsize_t[]){bst->dim_logical[i]}, NULL)) < 0) {
				fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", qnums_name, space_qnums);
				return -1;
			}

			if ((attr = H5Acreate2(dset, qnums_name, H5T_NATIVE_INT, space_qnums, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
				fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", qnums_name, attr);
				return -1;
			}

			if ((status = H5Awrite(attr, H5T_NATIVE_INT, bst->qnums_logical[i])) < 0) {
				fprintf(stderr, "H5Awrite() failed for %s, return value: %d\n", qnums_name, status);
				return -1;
			}

			if ((status = H5Aclose(attr)) < 0) {
				fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", qnums_name, status);
				return -1;
			}

			if ((status = H5Sclose(space_qnums)) < 0) {
				fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", qnums_name, status);
				return -1;
			}
		}

		if ((status = H5Dclose(dset)) < 0) {
			fprintf(stderr, "H5Dclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		if ((status = H5Sclose(space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		delete_dense_tensor(&dt);
	}

	if ((status = H5Gclose(root_group)) < 0) {
		fprintf(stderr, "H5Gclose() failed for root_group, return value: %d\n", status);
		return -1;
	}

	if ((status = H5Fclose(file)) < 0) {
		fprintf(stderr, "H5Fclose() failed for %s, return value: %d\n", filename, status);
		return -1;
	}

	return 0;
}

hid_t get_axis_dir_enum_dtype() {
	herr_t status;
	hid_t enum_dtype;
	if ((enum_dtype = H5Tenum_create(H5T_NATIVE_INT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Tenum_create() failed for axis_dir, return value: %lld\n", enum_dtype);
		return -1;
	}

	enum tensor_axis_direction in = TENSOR_AXIS_IN;
	if ((status = H5Tenum_insert(enum_dtype, "TENSOR_AXIS_IN", &in)) < 0) {
		fprintf(stderr, "H5Tenum_insert() failed for axis_dir(TENSOR_AXIS_IN), return value: %d\n", status);
		return -1;
	}

	enum tensor_axis_direction out = TENSOR_AXIS_OUT;
	if ((status = H5Tenum_insert(enum_dtype, "TENSOR_AXIS_OUT", &out)) < 0) {
		fprintf(stderr, "H5Tenum_insert() failed for axis_dir(TENSOR_AXIS_OUT), return value: %d\n", status);
		return -1;
	}

	return enum_dtype;
}