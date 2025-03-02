#pragma once

#include "mps.h"

/// \brief Import MPS from HDF5 file.
/// \note Uses HDF5 compound type for single and double complex numbers.
/// \returns 0 on success, -1 otherwise
int load_mps_hdf5(const char* filename, struct mps* mps);

/// \brief Dump MPS to HDF5 file.
/// \note Uses HDF5 compound type for single and double complex numbers.
/// \returns 0 on success, -1 otherwise
int save_mps_hdf5(const struct mps* mps, const char* filename);