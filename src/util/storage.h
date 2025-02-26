#pragma once

#include "mps.h"

int load_mps_hdf5(const char* filename, struct mps* mps);
int save_mps_hdf5(const struct mps* mps, const char* filename);