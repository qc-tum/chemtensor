/// \file util.h
/// \brief Utility functions.

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <hdf5.h>
#include "numeric.h"
#include "qnumber.h"


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
static inline long lmin(const long a, const long b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two long integers.
///
static inline long lmax(const long a, const long b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two quantum numbers.
///
static inline qnumber qmin(const qnumber a, const qnumber b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two quantum numbers.
///
static inline qnumber qmax(const qnumber a, const qnumber b)
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


long integer_product(const long* x, const int n);


long ipow(long base, int exp);


//________________________________________________________________________________________________________________________
//


bool is_identity_permutation(const int* perm, const int n);


//________________________________________________________________________________________________________________________
//


double uniform_distance(const enum numeric_type dtype, const long n, const void* restrict x, const void* restrict y);


double norm2(const enum numeric_type dtype, const long n, const void* x);


//________________________________________________________________________________________________________________________
//

// HDF5 utility functions

herr_t get_hdf5_dataset_ndims(hid_t file, const char* name, int* ndims);

herr_t get_hdf5_dataset_dims(hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t get_hdf5_attribute_dims(hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t write_hdf5_vector_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const long length, const void* data);

hid_t construct_hdf5_single_complex_dtype(const bool storage);

hid_t construct_hdf5_double_complex_dtype(const bool storage);

enum numeric_type hdf5_to_numeric_dtype(const hid_t dtype);

hid_t numeric_to_hdf5_dtype(const enum numeric_type dtype, const bool storage);
