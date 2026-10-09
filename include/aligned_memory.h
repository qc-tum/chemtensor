/// \file aligned_memory.h
/// \brief Aligned memory allocation.

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>


/// \brief Memory alignment used in dynamic memory allocation, must be a power of 2.
#define CT_MEM_DATA_ALIGN 16


//________________________________________________________________________________________________________________________
///
/// \brief Allocate 'size' bytes of uninitialized storage, and return a pointer to the allocated memory block.
///
static inline void* ct_malloc(size_t size)
{
	#ifdef _WIN32
	return _aligned_malloc(size, CT_MEM_DATA_ALIGN);
	#else
	// rounding up would wrap around to a tiny size
	if (size > SIZE_MAX - (CT_MEM_DATA_ALIGN - 1)) {
		return NULL;
	}
	// round 'size' up to the next multiple of 'CT_MEM_DATA_ALIGN', which must be a power of 2
	return aligned_alloc(CT_MEM_DATA_ALIGN, (size + CT_MEM_DATA_ALIGN - 1) & ~((size_t)CT_MEM_DATA_ALIGN - 1));
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Deallocate a previously allocated memory block.
///
static inline void ct_free(void* memblock)
{
	#ifdef _WIN32
	_aligned_free(memblock);
	#else
	free(memblock);
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate storage for an array of 'num' objects of 'size' bytes, initialize the storage with zeros,
/// and return a pointer to the allocated memory block.
///
static inline void* ct_calloc(size_t num, size_t size)
{
	// check if 'num * size' would overflow
	if (size != 0 && num > SIZE_MAX / size) {
		return NULL;
	}
	const size_t total = num * size;
	void* p = ct_malloc(total);
	if (p != NULL) {
		memset(p, 0, total);
	}
	return p;
}
