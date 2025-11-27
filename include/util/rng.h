/// \file rng.h
/// \brief Pseudo-random number generation.

#pragma once

#include <complex.h>
#include "pcg_basic.h"


//________________________________________________________________________________________________________________________
///
/// \brief Random number generator state.
///
struct rng_state
{
	struct pcg32x2_random pcgstate;  //!< PCG random number generator state
};


void seed_rng_state(const uint64_t seed, struct rng_state* state);


//________________________________________________________________________________________________________________________
//

// uniformly distributed integer random numbers

uint32_t rand_uint32(struct rng_state* state);

uint64_t rand_uint64(struct rng_state* state);

uint64_t rand_interval(const uint64_t bound, struct rng_state* state);


//________________________________________________________________________________________________________________________
//

// random choice without replacement

void rand_choice(const uint64_t bound, const uint64_t num_samples, struct rng_state* state, uint64_t* ret);


//________________________________________________________________________________________________________________________
//

// uniformly distributed random numbers from the interval [0, 1)

double randu(struct rng_state* state);

float randuf(struct rng_state* state);


//________________________________________________________________________________________________________________________
//

// standard normal (Gaussian) random numbers

double randn(struct rng_state* state);

float randnf(struct rng_state* state);

double complex crandn(struct rng_state* state);

float complex crandnf(struct rng_state* state);
