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
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space'for '%s' failed\n", name);
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
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Dget_space(dset);
	if (space < 0)
	{
		fprintf(stderr, "'H5Dget_space' for '%s' failed\n", name);
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
		fprintf(stderr, "'H5Dopen' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dread' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	H5Dclose(dset);

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
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate_simple' for '%s' failed\n", name);
		return -1;
	}

	// property list to disable time tracking
	hid_t cplist = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_obj_track_times(cplist, 0);
	if (status < 0)
	{
		fprintf(stderr, "creating property list failed, return value: %d\n", status);
		return status;
	}

	// create dataset
	hid_t dset = H5Dcreate(file, name, mem_type_store, space, H5P_DEFAULT, cplist, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dcreate' for '%s' failed\n", name);
		return -1;
	}

	// write the data to the dataset
	status = H5Dwrite(dset, mem_type_input, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dwrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	H5Dclose(dset);
	H5Pclose(cplist);
	H5Sclose(space);

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
		fprintf(stderr, "'H5Aopen' for '%s' failed\n", name);
		return -1;
	}

	hid_t space = H5Aget_space(attr);
	if (space < 0)
	{
		fprintf(stderr, "'H5Aget_space' for '%s' failed\n", name);
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
///
herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Aread(attr, mem_type, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aread' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 scalar attribute to a file.
///
herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	hid_t space = H5Screate(H5S_SCALAR);
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate' for '%s' failed\n", name);
		return -1;
	}

	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Acreate' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Awrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}
	status = H5Sclose(space);
	if (status < 0)
	{
		fprintf(stderr, "'H5Sclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 vector attribute to a file.
///
herr_t write_hdf5_vector_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const long length, const void* data)
{
	hsize_t dims[1] = { length };
	hid_t space = H5Screate_simple(1, dims, NULL);
	if (space < 0)
	{
		fprintf(stderr, "'H5Screate_simple' for '%s' failed\n", name);
		return -1;
	}

	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Acreate' for '%s' failed\n", name);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Awrite' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}
	status = H5Sclose(space);
	if (status < 0)
	{
		fprintf(stderr, "'H5Sclose' for '%s' failed, return value: %d\n", name, status);
		return status;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the identifier of the HDF5 compound type for single precision complex numbers, or -1 on failure.
/// 'storage' indicates whether the data type is used for storage (fixed endian convention) or as a memory type used in a program.
///
hid_t construct_hdf5_single_complex_dtype(const bool storage)
{
	hid_t type_id = H5Tcreate(H5T_COMPOUND, sizeof(scomplex));
	if (type_id < 0)
	{
		fprintf(stderr, "'H5Tcreate' failed for single complex data type\n");
		return -1;
	}

	// name real and imaginary parts "r" and "i", for compatibility with Python/h5py
	herr_t status = H5Tinsert(type_id, "r", 0, storage ? H5T_IEEE_F32LE : H5T_NATIVE_FLOAT);
	if (status < 0)
	{
		fprintf(stderr, "'H5Tinsert' failed for single complex data type (real part), return value: %d\n", status);
		return -1;
	}
	status = H5Tinsert(type_id, "i", sizeof(float), storage ? H5T_IEEE_F32LE : H5T_NATIVE_FLOAT);
	if (status < 0)
	{
		fprintf(stderr, "'H5Tinsert' failed for single complex data type (imaginary part), return value: %d\n", status);
		return -1;
	}

	return type_id;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the identifier of the HDF5 compound type for double precision complex numbers, or -1 on failure.
/// 'storage' indicates whether the data type is used for storage (fixed endian convention) or as a memory type used in a program.
///
hid_t construct_hdf5_double_complex_dtype(const bool storage)
{
	hid_t type_id = H5Tcreate(H5T_COMPOUND, sizeof(dcomplex));
	if (type_id < 0)
	{
		fprintf(stderr, "'H5Tcreate' failed for double complex data type\n");
		return -1;
	}

	// name real and imaginary parts "r" and "i", for compatibility with Python/h5py
	herr_t status = H5Tinsert(type_id, "r", 0, storage ? H5T_IEEE_F64LE : H5T_NATIVE_DOUBLE);
	if (status < 0)
	{
		fprintf(stderr, "'H5Tinsert' failed for double complex data type (real part), return value: %d\n", status);
		return -1;
	}
	status = H5Tinsert(type_id, "i", sizeof(double), storage ? H5T_IEEE_F64LE : H5T_NATIVE_DOUBLE);
	if (status < 0)
	{
		fprintf(stderr, "'H5Tinsert' failed for double complex data type (imaginary part), return value: %d\n", status);
		return -1;
	}

	return type_id;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an HDF5 data type identifier to the corresponding numeric type.
/// \returns numeric data type, or -1 on failure
///
enum numeric_type hdf5_to_numeric_dtype(const hid_t dtype)
{
	if (H5Tequal(dtype, H5T_NATIVE_FLOAT) || H5Tequal(dtype, H5T_IEEE_F32LE))
	{
		return CT_SINGLE_REAL;
	}

	if (H5Tequal(dtype, H5T_NATIVE_DOUBLE) || H5Tequal(dtype, H5T_IEEE_F64LE))
	{
		return CT_DOUBLE_REAL;
	}

	const hid_t single_complex_id_s = construct_hdf5_single_complex_dtype(true);
	const hid_t single_complex_id_m = construct_hdf5_single_complex_dtype(false);
	if (H5Tequal(dtype, single_complex_id_s) || H5Tequal(dtype, single_complex_id_m))
	{
		H5Tclose(single_complex_id_m);
		H5Tclose(single_complex_id_s);
		return CT_SINGLE_COMPLEX;
	}
	H5Tclose(single_complex_id_m);
	H5Tclose(single_complex_id_s);

	const hid_t double_complex_id_s = construct_hdf5_double_complex_dtype(true);
	const hid_t double_complex_id_m = construct_hdf5_double_complex_dtype(false);
	if (H5Tequal(dtype, double_complex_id_s) || H5Tequal(dtype, double_complex_id_m))
	{
		H5Tclose(double_complex_id_m);
		H5Tclose(double_complex_id_s);
		return CT_DOUBLE_COMPLEX;
	}
	H5Tclose(double_complex_id_m);
	H5Tclose(double_complex_id_s);

	// invalid data type
	assert(false);
	return -1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a numeric type to the corresponding HDF5 data type.
/// 'storage' indicates whether the data type is used for storage (fixed endian convention) or as a memory type used in a program.
/// \returns HDF5 data type identifier, or -1 on failure
///
hid_t numeric_to_hdf5_dtype(const enum numeric_type dtype, const bool storage)
{
	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			return (storage ? H5T_IEEE_F32LE : H5T_NATIVE_FLOAT);
		}
		case CT_DOUBLE_REAL:
		{
			return (storage ? H5T_IEEE_F64LE : H5T_NATIVE_DOUBLE);
		}
		case CT_SINGLE_COMPLEX:
		{
			return construct_hdf5_single_complex_dtype(storage);
		}
		case CT_DOUBLE_COMPLEX:
		{
			return construct_hdf5_double_complex_dtype(storage);
		}
		default:
		{
			// unknown data type
			assert(false);
			return -1;
		}
	}
}
