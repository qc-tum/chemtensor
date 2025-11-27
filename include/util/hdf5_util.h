/// \file hdf5_util.h
/// \brief HDF5 utility functions.

#pragma once

#include <hdf5.h>
#include "numeric.h"
#include "util.h"


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
