/// \file cblas_ct.h
/// \brief BLAS function declarations on various platforms.

#pragma once


#ifndef __APPLE__

#include <cblas.h>

#else

#include <Accelerate/Accelerate.h>

#endif
