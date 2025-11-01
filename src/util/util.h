/// \file util.h
/// \brief Utility functions.

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <hdf5.h>
#include "numeric.h"


/// \brief Universal "long" integer, used for tensor dimensions.
typedef int64_t ct_long;


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two integers.
///
static inline int imin(const int a, const int b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two integers.
///
static inline int imax(const int a, const int b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two long integers.
///
static inline ct_long lmin(const ct_long a, const ct_long b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two long integers.
///
static inline ct_long lmax(const ct_long a, const ct_long b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Square function x -> x^2.
///
static inline double square(const double x)
{
	return x*x;
}


//________________________________________________________________________________________________________________________
//


ct_long integer_product(const ct_long* x, const int n);


ct_long ipow(ct_long base, int exp);


//________________________________________________________________________________________________________________________
//


bool is_identity_permutation(const int* perm, const int n);


//________________________________________________________________________________________________________________________
//


double uniform_distance(const enum numeric_type dtype, const ct_long n, const void* restrict x, const void* restrict y);


double norm2(const enum numeric_type dtype, const ct_long n, const void* x);


//________________________________________________________________________________________________________________________
//

// HDF5 utility functions

herr_t get_hdf5_dataset_ndims(const hid_t file, const char* name, int* ndims);

herr_t get_hdf5_dataset_dims(const hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_dataset(const hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_dataset(const hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t get_hdf5_attribute_dims(const hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_attribute(const hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_scalar_attribute(const hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t write_hdf5_vector_attribute(const hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const ct_long length, const void* data);

hid_t construct_hdf5_single_complex_dtype(const bool storage);

hid_t construct_hdf5_double_complex_dtype(const bool storage);

enum numeric_type hdf5_to_numeric_dtype(const hid_t dtype);

hid_t numeric_to_hdf5_dtype(const enum numeric_type dtype, const bool storage);
