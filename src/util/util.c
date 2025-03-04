/// \file util.c
/// \brief Utility functions.

#include <math.h>
#include <memory.h>
#include <complex.h>
#include <cblas.h>
#include <assert.h>
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Calculate the product of a list of integer numbers.
///
long integer_product(const long* x, const int n)
{
	assert(n >= 0); // n == 0 is still reasonable

	long prod = 1;
	for (int i = 0; i < n; i++)
	{
		prod *= x[i];
	}

	return prod;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the integer power `base^exp`.
///
long ipow(long base, int exp)
{
	assert(exp >= 0);

	long result = 1;
	while (exp != 0)
	{
		if ((exp & 1) == 1) {
			result *= base;
		}
		exp >>= 1;
		base *= base;
	}
	return result;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether a permutation is the identity permutation.
///
bool is_identity_permutation(const int* perm, const int n)
{
	for (int i = 0; i < n; i++) {
		if (perm[i] != i) {
			return false;
		}
	}
	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between 'x' and 'y'.
///
/// The result is cast to double format (even if the actual entries are of single precision).
///
double uniform_distance(const enum numeric_type dtype, const long n, const void* restrict x, const void* restrict y)
{
	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* xv = x;
			const float* yv = y;
			float d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmaxf(d, fabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_DOUBLE_REAL:
		{
			const double* xv = x;
			const double* yv = y;
			double d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmax(d, fabs(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* xv = x;
			const scomplex* yv = y;
			float d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmaxf(d, cabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* xv = x;
			const dcomplex* yv = y;
			double d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmax(d, cabs(xv[i] - yv[i]));
			}
			return d;
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Euclidean norm of a vector.
///
double norm2(const enum numeric_type dtype, const long n, const void* x)
{
	assert(n >= 0);

	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			return cblas_snrm2(n, x, 1);
		}
		case CT_DOUBLE_REAL:
		{
			return cblas_dnrm2(n, x, 1);
		}
		case CT_SINGLE_COMPLEX:
		{
			return cblas_scnrm2(n, x, 1);
		}
		case CT_DOUBLE_COMPLEX:
		{
			return cblas_dznrm2(n, x, 1);
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the number of dimensions (degree) of an HDF5 dataset.
///
herr_t get_hdf5_dataset_ndims(hid_t file, const char* name, int* ndims)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed, return value: %" PRId64 "\n", name, dset);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space' for '%s' failed, return value: %" PRId64 "\n", name, space);
		return -1;
	}

	*ndims = H5Sget_simple_extent_ndims(space);

	H5Sclose(space);
	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the dimensions of an HDF5 dataset.
///
herr_t get_hdf5_dataset_dims(hid_t file, const char* name, hsize_t* dims)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed, return value: %" PRId64 "\n", name, dset);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space' for '%s' failed, return value: %" PRId64 "\n", name, space);
		return -1;
	}

	// get dimensions
	H5Sget_simple_extent_dims(space, dims, NULL);

	H5Sclose(space);
	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 dataset from a file.
///
herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed, return value: %" PRId64 "\n", name, dset);
		return -1;
	}

	herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dread' failed, return value: %d\n", status);
		return status;
	}

	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the dimensions of an HDF5 attribute.
///
herr_t get_hdf5_attribute_dims(hid_t file, const char* name, hsize_t* dims)
{
	hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed, return value: %" PRId64 "\n", name, attr);
		return -1;
	}

	hid_t space = H5Aget_space(attr);
	if (space < 0)
	{
		fprintf(stderr, "'H5Aget_space' for '%s' failed, return value: %" PRId64 "\n", name, space);
		return -1;
	}

	// get dimensions
	H5Sget_simple_extent_dims(space, dims, NULL);

	H5Sclose(space);
	H5Aclose(attr);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 attribute from a file.
/// \returns 0 on success, -1 otherwise.
///
herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data)
{
	herr_t status;

	hid_t attr;
	if ((attr = H5Aopen(file, name, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Aopen() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Aread(attr, mem_type, data)) < 0) {
		fprintf(stderr, "H5Aread() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	if ((status = H5Aclose(attr)) < 0) {
		fprintf(stderr, "H5Aclose() failed for %s, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 dataset to a file.
///
herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate_simple(degree, dims, NULL);
	if (space < 0) {
		fprintf(stderr, "'H5Screate_simple' failed, return value: %" PRId64 "\n", space);
		return -1;
	}
	
	// property list to disable time tracking
	hid_t cplist = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_obj_track_times(cplist, 0);
	if (status < 0) {
		fprintf(stderr, "creating property list failed, return value: %d\n", status);
		return status;
	}

	// create dataset
	hid_t dset = H5Dcreate(file, name, mem_type_store, space, H5P_DEFAULT, cplist, H5P_DEFAULT);
	if (dset < 0) {
		fprintf(stderr, "'H5Dcreate' failed, return value: %" PRId64 "\n", dset);
		return -1;
	}

	// write the data to the dataset
	status = H5Dwrite(dset, mem_type_input, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0) {
		fprintf(stderr, "'H5Dwrite' failed, return value: %d\n", status);
		return status;
	}

	H5Dclose(dset);
	H5Pclose(cplist);
	H5Sclose(space);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 scalar attribute to a file.
/// \returns 0 on success, -1 otherwise
///
herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	herr_t status;

	hid_t space, attr;
	if ((space = H5Screate(H5S_SCALAR)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Screate() failed for %s, return value: %lld\n", name, space);
		return -1;
	}

	if ((attr = H5Acreate2(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Awrite(attr, mem_type_input, data)) < 0) {
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


//________________________________________________________________________________________________________________________
///
/// \brief Create HDF5 vector attribute with name and type of given length attached to parent.
/// \returns 0 on success, -1 otherwise
///
herr_t write_hdf5_vector_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const hsize_t length, const void* data) {
	herr_t status;

	hid_t space, attr;
	if ((space = H5Screate_simple(1, (hsize_t[]){length}, NULL)) < 0) {
		fprintf(stderr, "H5Screate_simple() failed for %s, return value: %lld\n", name, space);
		return -1;
	}

	if ((attr = H5Acreate2(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT)) == H5I_INVALID_HID) {
		fprintf(stderr, "H5Acreate() failed for %s, return value: %lld\n", name, attr);
		return -1;
	}

	if ((status = H5Awrite(attr, mem_type_input, data)) < 0) {
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


//________________________________________________________________________________________________________________________
///
/// \brief Create HDF5 enumeration datatype for `tensor_axis_direction` and return its id.
///
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


//________________________________________________________________________________________________________________________
///
/// \brief Return id of HDF5 compound type for single precision complex numbers
/// \returns id on success, -1 otherwise
///
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


//________________________________________________________________________________________________________________________
///
/// \brief Return id of HDF5 compound type for double precision complex numbers
/// \returns id on success, -1 otherwise
///
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


//________________________________________________________________________________________________________________________
///
/// \brief Convert HDF5 datatype id to chemtensor numeric_type
/// \returns 0 on success, -1 otherwise
///
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


//________________________________________________________________________________________________________________________
///
/// \brief Convert HDF5 datatype id to chemtensor numeric_type
/// \returns HDF5 datatype id, or -1 on failure
///
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
