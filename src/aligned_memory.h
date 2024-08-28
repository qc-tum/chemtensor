/// \file aligned_memory.h
/// \brief Aligned memory allocation.

#pragma once

#include <stdlib.h>
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
	// round 'size' up to the next multiple of 'CT_MEM_DATA_ALIGN', which must be a power of 2
	return aligned_alloc(CT_MEM_DATA_ALIGN, (size + CT_MEM_DATA_ALIGN - 1) & (-CT_MEM_DATA_ALIGN));
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
	void* p = ct_malloc(num * size);
	if (p != NULL) {
		memset(p, 0, num * size);
	}
	return p;
}
