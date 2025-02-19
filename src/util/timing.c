/// \file timing.c
/// \brief Timing utility functions.

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "timing.h"


//________________________________________________________________________________________________________________________
///
/// \brief Get the current time as number of clock ticks.
///
int64_t get_time_ticks()
{
	#ifdef _WIN32

	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	return t.QuadPart;

	#else

	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return 1000000000LL * t.tv_sec + t.tv_nsec;

	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the performance timer resolution.
///
int64_t get_tick_resolution()
{
	#ifdef _WIN32

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return freq.QuadPart;

	#else  // clock_gettime has nanosecond resolution

	return 1000000000LL;

	#endif
}
