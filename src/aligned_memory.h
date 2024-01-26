/// \file aligned_memory.h
/// \brief Aligned memory allocation.

#pragma once

#include <stdlib.h>
#include <malloc.h>
#include <memory.h>


#define MEM_DATA_ALIGN 64


#ifdef _WIN32

inline void* aligned_alloc(size_t alignment, size_t size)
{
	return _aligned_malloc(size, alignment);
}

inline void aligned_free(void* memblock)
{
	_aligned_free(memblock);
}

#else

static inline void aligned_free(void* memblock)
{
	free(memblock);
}

#endif


static inline void* aligned_calloc(size_t alignment, size_t num, size_t size)
{
	void* p = aligned_alloc(alignment, num * size);
	if (p != NULL) {
		memset(p, 0, num * size);
	}
	return p;
}
