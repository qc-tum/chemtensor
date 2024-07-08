/// \file pcg_basic.c
/// \brief PCG random number generation.

// PCG Random Number Generation for C.
//
// Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// For additional information about the PCG random number generation scheme,
// including its license and other licensing options, visit
//
//     http://www.pcg-random.org


#include "pcg_basic.h"


//________________________________________________________________________________________________________________________
///
/// \brief Seed the 32-bit RNG. Specified in two parts, state initializer and a sequence selection constant (a.k.a. stream id).
///
void pcg32_srandom_r(const uint64_t initstate, const uint64_t initseq, struct pcg32_random* rng)
{
	rng->state = 0U;
	rng->inc = (initseq << 1u) | 1u;
	pcg32_random_r(rng);
	rng->state += initstate;
	pcg32_random_r(rng);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed 32-bit random number.
///
uint32_t pcg32_random_r(struct pcg32_random* rng)
{
	uint64_t oldstate = rng->state;
	rng->state = oldstate * 6364136223846793005ULL + rng->inc;
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed number, r, where 0 <= r < bound.
///
uint32_t pcg32_boundedrand_r(const uint32_t bound, struct pcg32_random* rng)
{
	// To avoid bias, we need to make the range of the RNG a multiple of
	// bound, which we do by dropping output less than a threshold.
	// A naive scheme to calculate the threshold would be to do
	//
	//     uint32_t threshold = 0x100000000ull % bound;
	//
	// but 64-bit div/mod is slower than 32-bit div/mod (especially on
	// 32-bit platforms).  In essence, we do
	//
	//     uint32_t threshold = (0x100000000ull-bound) % bound;
	//
	// because this version will calculate the same modulus, but the LHS
	// value is less than 2^32.

	uint32_t threshold = -bound % bound;

	// Uniformity guarantees that this loop will terminate.  In practice, it
	// should usually terminate quickly; on average (assuming all bounds are
	// equally likely), 82.25% of the time, we can expect it to require just
	// one iteration.  In the worst case, someone passes a bound of 2^31 + 1
	// (i.e., 2147483649), which invalidates almost 50% of the range.  In 
	// practice, bounds are typically small and only a tiny amount of the range
	// is eliminated.
	for (;;) {
		uint32_t r = pcg32_random_r(rng);
		if (r >= threshold) {
			return r % bound;
		}
	}
}


// This code shows how you can cope if you're on a 32-bit platform (or a
// 64-bit platform with a mediocre compiler) that doesn't support 128-bit math,
// or if you're using the basic version of the library which only provides
// 32-bit generation.
//
// Here we build a 64-bit generator by tying together two 32-bit generators.
// Note that we can do this because we set up the generators so that each
// 32-bit generator has a *totally different* different output sequence
// -- if you tied together two identical generators, that wouldn't be nearly
// as good.
//
// For simplicity, we keep the period fixed at 2^64.  The state space is
// approximately 2^254 (actually  2^64 * 2^64 * 2^63 * (2^63 - 1)), which
// is huge.


//________________________________________________________________________________________________________________________
///
/// \brief Seed the 64-bit RNG.
///
void pcg32x2_srandom_r(uint64_t seed1, uint64_t seed2, uint64_t seq1, uint64_t seq2, struct pcg32x2_random* rng)
{
	uint64_t mask = ~0ull >> 1;
	// The stream for each of the two generators *must* be distinct
	if ((seq1 & mask) == (seq2 & mask)) {
		seq2 = ~seq2;
	}
	pcg32_srandom_r(seed1, seq1, &rng->gen[0]);
	pcg32_srandom_r(seed2, seq2, &rng->gen[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed 64-bit random number.
///
uint64_t pcg32x2_random_r(struct pcg32x2_random* rng)
{
	return ((uint64_t)(pcg32_random_r(&rng->gen[0])) << 32)
	                 | pcg32_random_r(&rng->gen[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed 64-bit random number, r, where 0 <= r < bound.
///
uint64_t pcg32x2_boundedrand_r(const uint64_t bound, struct pcg32x2_random* rng)
{
	uint64_t threshold = -bound % bound;
	for (;;) {
		uint64_t r = pcg32x2_random_r(rng);
		if (r >= threshold) {
			return r % bound;
		}
	}
}
