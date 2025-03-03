#include "storage.h"
#include "aligned_memory.h"
#include "mps.h"
#include <hdf5.h>

/// \brief Create HDF5 enumeration datatype for `tensor_axis_direction` and return its id.
hid_t get_axis_dir_enum_dtype();

/// \brief Create HDF5 scalar attribute with name and type attached to parent.
/// \returns 0 on success, -1 otherwise
int write_scalar_attribute(const char* name, hid_t type_id, const void* buf, hid_t parent_id);

/// \brief Create HDF5 vector attribute with name and type of given length attached to parent.
/// \returns 0 on success, -1 otherwise
int write_vector_attribute(const char* name, hid_t type_id, const hsize_t length, const void* buf, hid_t parent_id);

/// \brief Read HDF5 attribute with name and type into buf.
/// \returns 0 on success, -1 otherwise
int read_attribute(const char* name, hid_t type_id, hid_t parent_id, void* buf);

/// \brief Convert HDF5 datatype id to chemtensor numeric_type
/// \returns 0 on success, -1 otherwise
int hdf5_to_chemtensor_dtype(hid_t hdf5_dtype, enum numeric_type* ct_dtype);

/// \brief Convert HDF5 datatype id to chemtensor numeric_type
/// \returns HDF5 datatype id, or -1 on failure
hid_t chemtensor_to_hdf5_dtype(enum numeric_type ct_dtype);

/// \brief Return id of HDF5 compound type for single precision complex numbers
/// \returns id on success, -1 otherwise
hid_t get_single_complex_dtype();

/// \brief Return id of HDF5 compound type for double precision complex numbers
/// \returns id on success, -1 otherwise
hid_t get_double_complex_dtype();

/// \brief Import MPS from HDF5 file.
/// \note Uses HDF5 compound type for single and double complex numbers.
/// \returns 0 on success, -1 otherwise
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

	// read attributes attached to '/' group and allocate
	// empty mps with
	// - physical dimension:                d
	// - number of sites:                   nsites
	// - quantum numbers at physical sites: qsite
	long d;
	int nsites;
	if (read_attribute("d", H5T_NATIVE_LONG, root_group, (void*)&d) < 0 ||
		read_attribute("nsites", H5T_NATIVE_INT, root_group, (void*)&nsites) < 0) {
		return -1;
	}

	qnumber qsite[d];
	if (read_attribute("qsite", H5T_NATIVE_INT, root_group, (void*)&qsite) < 0) {
		return -1;
	}

	allocate_empty_mps(nsites, d, (const qnumber*)&qsite, mps);

	// read datasets tensor-{site}
	//
	// here we load dense tensors from and the accompying attributes
	// to reconstruct the block sparse tensors of the MPS
	for (size_t site = 0; site < nsites; site++) {
		char dset_name[128];
		sprintf(dset_name, "tensor-%zu", site);

		hid_t dset;
		if ((dset = H5Dopen2(file, dset_name, H5P_DEFAULT)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dopen2() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// read hdf5 dataspace of dataset containing the dimensions for
		// the empty dense tensors
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

		if ((status = H5Sclose(dset_space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// read hdf5 datatype of dataset and convert it to chemtensor's
		// numeric_type to allocate empty dense tensors
		hid_t hdf5_dtype;
		if ((hdf5_dtype = H5Dget_type(dset)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dget_type() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		enum numeric_type ct_dtype;
		if (hdf5_to_chemtensor_dtype(hdf5_dtype, &ct_dtype) < 0) {
			return -1;
		}

		struct dense_tensor dt;
		allocate_dense_tensor(ct_dtype, ndim, (const long*)&dims, &dt);

		// read raw data into empty dense tensor
		if (H5Dread(dset, hdf5_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dt.data) < 0) {
			fprintf(stderr, "H5Dread() failed for %s, return value: %d\n", dset_name, ndim);
			return -1;
		}

		if ((status = H5Tclose(hdf5_dtype)) < 0) {
			fprintf(stderr, "H5Tclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// read attributes attached to tensor-{site} dataset and
		// reconstruct block sparse from dense tensor with
		// - tensor axis directions: axis_dir
		// - quantum numbers:        qnums-{dim}
		//
		// uses a custom type for the axis dir enumeration, hence,
		// call a helper function to construct it
		hid_t enum_type_id;
		enum tensor_axis_direction axis_dir[ndim];
		if ((enum_type_id = get_axis_dir_enum_dtype()) < 0) {
			return -1;
		}

		if (read_attribute("axis_dir", enum_type_id, dset, &axis_dir) < 0) {
			return -1;
		}

		if ((status = H5Tclose(enum_type_id)) < 0) {
			fprintf(stderr, "H5Tclose() failed for axis_dir, return value: %d\n", status);
			return -1;
		}

		qnumber** qnums = ct_calloc(ndim, sizeof(qnumber*));
		for (size_t dim = 0; dim < ndim; dim++) {
			char qnums_name[128];
			sprintf(qnums_name, "qnums-%zu", dim);

			hid_t attr;
			if ((attr = H5Aopen(dset, qnums_name, H5P_DEFAULT)) == H5I_INVALID_HID) {
				fprintf(stderr, "H5Aopen() failed for %s, return value: %lld\n", qnums_name, attr);
				return -1;
			}

			// retrieve dataspace information for qnums
			// dataspace for qnums is one-dimensional, hence, scalar variable suffices
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

				if ((status = H5Sclose(space_qnums)) < 0) {
					fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", qnums_name, status);
					return -1;
				}
			}

			qnums[dim] = ct_malloc(qnums_dim * sizeof(qnumber));
			if ((status = H5Aread(attr, H5T_NATIVE_INT, qnums[dim])) < 0) {
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

/// \brief Dump MPS to HDF5 file.
/// \note Uses HDF5 compound type for single and double complex numbers.
/// \returns 0 on success, -1 otherwise
int save_mps_hdf5(const struct mps* mps, const char* filename) {
	herr_t status;

	hid_t file;
	if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Fcreate() failed for %s, return value: %lld\n", filename, file);
		return -1;
	}

	hid_t root_group;
	if ((root_group = H5Gopen1(file, "/")) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Gopen1() failed for /, return value: %lld\n", file);
		return -1;
	}

	// create attributes attached to '/' group
	// - physical dimension:                d
	// - number of sites:                   nsites
	// - quantum numbers at physical sites: qsite
	if (write_scalar_attribute("d", H5T_NATIVE_LONG, (void*)&mps->d, root_group) < 0 ||
		write_scalar_attribute("nsites", H5T_NATIVE_INT, (void*)&mps->nsites, root_group) < 0 ||
		write_vector_attribute("qsite", H5T_NATIVE_INT, (hsize_t)mps->d, (void*)mps->qsite, root_group) < 0) {
		return -1;
	}

	// create datasets tensor-{site} for each block sparse tensor
	//
	// here we convert block sparse to dense tensors and store
	// the accompying quantum numbers as attributes to reconstruct
	// the sparse tensors when importing
	for (size_t site = 0; site < mps->nsites; site++) {
		char dset_name[128];
		sprintf(dset_name, "tensor-%zu", site);

		struct dense_tensor dt;
		struct block_sparse_tensor* bst = &mps->a[site];
		block_sparse_to_dense_tensor(bst, &dt);

		hid_t dtype = chemtensor_to_hdf5_dtype(dt.dtype);

		hid_t space;
		if ((space = H5Screate_simple(dt.ndim, (const hsize_t*)dt.dim, NULL)) == H5I_INVALID_HID) {
			fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", dset_name, space);
			return -1;
		}

		hid_t dset;
		dset = H5Dcreate(root_group, dset_name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (dset == H5I_INVALID_HID) {
			fprintf(stderr, "H5Dcreate() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		if ((status = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dt.data)) < 0) {
			fprintf(stderr, "H5Dwrite() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		if ((status = H5Sclose(space)) < 0) {
			fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", dset_name, status);
			return -1;
		}

		// only compound type must be closed
		// H5Tclose will fail for immutable datatype, otherwise
		if (dt.dtype == CT_SINGLE_COMPLEX || dt.dtype == CT_DOUBLE_COMPLEX) {
			if ((status = H5Tclose(dtype)) < 0) {
				fprintf(stderr, "H5Tclose() failed for %s, return value: %d\n", dset_name, status);
				return -1;
			}
		}

		// create attributes attached to tensor-{site} dataset
		// - tensor axis directions:                axis_dir
		// - quantum numbers to reconstruct block
		//   sparse tensors:                        qnums-{dim}
		//
		// uses a custom type for the axis dir enumeration, hence,
		// call a helper function to construct it
		hid_t enum_type_id;
		if ((enum_type_id = get_axis_dir_enum_dtype()) < 0) {
			return -1;
		}

		if (write_vector_attribute("axis_dir", enum_type_id, (hsize_t)bst->ndim, (void*)bst->axis_dir, dset) < 0) {
			return -1;
		}

		if ((status = H5Tclose(enum_type_id)) < 0) {
			fprintf(stderr, "H5Tclose() failed for axis_dir, return value: %d\n", status);
			return -1;
		}

		for (size_t dim = 0; dim < bst->ndim; dim++) {
			char qnums_name[128];
			sprintf(qnums_name, "qnums-%zu", dim);

			if (write_vector_attribute(qnums_name, H5T_NATIVE_INT, (hsize_t)bst->dim_logical[dim], (void*)bst->qnums_logical[dim], dset) < 0) {
				return -1;
			}
		}

		if ((status = H5Dclose(dset)) < 0) {
			fprintf(stderr, "H5Dclose() failed for %s, return value: %d\n", dset_name, status);
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

hid_t get_single_complex_dtype() {
	herr_t status;
	hid_t complex_dtype;

	if ((complex_dtype = H5Tcreate(H5T_COMPOUND, sizeof(scomplex))) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Tcreate() failed for single complex dtype, return value: %lld\n", complex_dtype);
		return -1;
	}

	if ((status = H5Tinsert(complex_dtype, "real", 0, H5T_NATIVE_FLOAT)) < 0) {
		fprintf(stderr, "H5Tinsert() failed for single complex dtype (real), return value: %d\n", status);
		return -1;
	}

	if ((status = H5Tinsert(complex_dtype, "imag", sizeof(float), H5T_NATIVE_FLOAT)) < 0) {
		fprintf(stderr, "H5Tinsert() failed for single complex dtype (real), return value: %d\n", status);
		return -1;
	}

	return complex_dtype;
}

hid_t get_double_complex_dtype() {
	herr_t status;
	hid_t complex_dtype;

	if ((complex_dtype = H5Tcreate(H5T_COMPOUND, sizeof(dcomplex))) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Tcreate() failed for double complex dtype, return value: %lld\n", complex_dtype);
		return -1;
	}

	if ((status = H5Tinsert(complex_dtype, "real", 0, H5T_NATIVE_DOUBLE)) < 0) {
		fprintf(stderr, "H5Tinsert() failed for double complex dtype (real), return value: %d\n", status);
		return -1;
	}

	if ((status = H5Tinsert(complex_dtype, "imag", sizeof(double), H5T_NATIVE_DOUBLE)) < 0) {
		fprintf(stderr, "H5Tinsert() failed for double complex dtype (real), return value: %d\n", status);
		return -1;
	}

	return complex_dtype;
}

int write_scalar_attribute(const char* name, hid_t type_id, const void* buf, hid_t parent_id) {
	herr_t status;

	hid_t space, attr;
	if ((space = H5Screate(H5S_SCALAR)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Screate() failed for %s, return value: %lld\n", name, space);
		return -1;
	}

	if ((attr = H5Acreate2(parent_id, name, type_id, space, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Awrite(attr, type_id, buf)) < 0) {
		fprintf(stderr, "H5Awrite() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Aclose(attr)) < 0) {
		fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Sclose(space)) < 0) {
		fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}

int write_vector_attribute(const char* name, hid_t type_id, const hsize_t length, const void* buf, hid_t parent_id) {
	herr_t status;

	hid_t space, attr;
	if ((space = H5Screate_simple(1, (hsize_t[]){length}, NULL)) < 0) {
		fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", name, space);
		return -1;
	}

	if ((attr = H5Acreate2(parent_id, name, type_id, space, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Awrite(attr, type_id, buf)) < 0) {
		fprintf(stderr, "H5Awrite() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Aclose(attr)) < 0) {
		fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Sclose(space)) < 0) {
		fprintf(stderr, "H5Sclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}

int read_attribute(const char* name, hid_t type_id, hid_t parent_id, void* buf) {
	herr_t status;

	hid_t attr;
	if ((attr = H5Aopen(parent_id, name, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Aread(attr, type_id, buf)) < 0) {
		fprintf(stderr, "H5Aread() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Aclose(attr)) < 0) {
		fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}

int hdf5_to_chemtensor_dtype(hid_t hdf5_dtype, enum numeric_type* ct_dtype) {
	int ret = 0;
	hid_t single_complex_id = get_single_complex_dtype();
	hid_t double_complex_id = get_double_complex_dtype();

	if (H5Tequal(hdf5_dtype, H5T_NATIVE_FLOAT)) {
		*ct_dtype = CT_SINGLE_REAL;
	} else if (H5Tequal(hdf5_dtype, H5T_NATIVE_DOUBLE)) {
		*ct_dtype = CT_DOUBLE_REAL;
	} else if (H5Tequal(hdf5_dtype, single_complex_id)) {
		*ct_dtype = CT_SINGLE_COMPLEX;
	} else if (H5Tequal(hdf5_dtype, double_complex_id)) {
		*ct_dtype = CT_DOUBLE_COMPLEX;
	} else {
		fprintf(stderr, "Invalid dtype: %lld.\n", hdf5_dtype);
		ret = -1;
	}

	H5Tclose(single_complex_id);
	H5Tclose(double_complex_id);

	return ret;
}

hid_t chemtensor_to_hdf5_dtype(enum numeric_type ct_dtype) {
	switch (ct_dtype) {
	case CT_SINGLE_REAL:
		return H5T_NATIVE_FLOAT;
	case CT_DOUBLE_REAL:
		return H5T_NATIVE_DOUBLE;
	case CT_SINGLE_COMPLEX:
		return get_single_complex_dtype();
	case CT_DOUBLE_COMPLEX:
		return get_double_complex_dtype();
	}

	return -1;
}