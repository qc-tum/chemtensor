/// \file pcg_basic.h
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


#pragma once

#include <stdint.h>


//________________________________________________________________________________________________________________________
///
/// \brief PCG 32-bit random number generator state.
///
struct pcg32_random
{
	uint64_t state;  //!< RNG state; all values are possible
	uint64_t inc;    //!< controls which RNG sequence (stream) is selected; must *always* be odd
};

void pcg32_srandom_r(const uint64_t initstate, const uint64_t initseq, struct pcg32_random* rng);


uint32_t pcg32_random_r(struct pcg32_random* rng);

uint32_t pcg32_boundedrand_r(const uint32_t bound, struct pcg32_random* rng);


//________________________________________________________________________________________________________________________
///
/// \brief PCG 64-bit random number generator state consisting of two 32-bit generators.
///
struct pcg32x2_random
{
	struct pcg32_random gen[2];  //!< 32-bit RNG states
};

void pcg32x2_srandom_r(uint64_t seed1, uint64_t seed2, uint64_t seq1, uint64_t seq2, struct pcg32x2_random* rng);


uint64_t pcg32x2_random_r(struct pcg32x2_random* rng);

uint64_t pcg32x2_boundedrand_r(const uint64_t bound, struct pcg32x2_random* rng);
