/// \file rng.c
/// \brief Pseudo-random number generation.

#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include "rng.h"


//________________________________________________________________________________________________________________________
///
/// \brief Seed the random number generator state.
///
void seed_rng_state(const uint64_t seed, struct rng_state* state)
{
	// use a temporary random number generator for the actual seeds
	struct pcg32_random rng_seed;
	const uint64_t mult = 6364136223846793005ULL;
	const uint64_t inc  = 1442695040888963407ULL;
	pcg32_srandom_r(seed, mult * seed + inc, &rng_seed);
	uint64_t seed1 = ((uint64_t)(pcg32_random_r(&rng_seed)) << 32) | pcg32_random_r(&rng_seed);
	uint64_t seed2 = ((uint64_t)(pcg32_random_r(&rng_seed)) << 32) | pcg32_random_r(&rng_seed);
	uint64_t seq1  = ((uint64_t)(pcg32_random_r(&rng_seed)) << 32) | pcg32_random_r(&rng_seed);
	uint64_t seq2  = ((uint64_t)(pcg32_random_r(&rng_seed)) << 32) | pcg32_random_r(&rng_seed);
	pcg32x2_srandom_r(seed1, seed2, seq1, seq2, &state->pcgstate);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed 32-bit random number.
///
uint32_t rand_uint32(struct rng_state* state)
{
	// solely use the first 32-bit generator state
	return pcg32_random_r(&state->pcgstate.gen[0]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed 64-bit random number.
///
uint64_t rand_uint64(struct rng_state* state)
{
	return pcg32x2_random_r(&state->pcgstate);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed integer random number from the interval [0, bound).
///
uint64_t rand_interval(const uint64_t bound, struct rng_state* state)
{
	return pcg32x2_boundedrand_r(bound, &state->pcgstate);
}


//________________________________________________________________________________________________________________________
///
/// \brief Draw 'num_samples' random numbers from the interval [0, bound) without replacement.
///
void rand_choice(const uint64_t bound, const uint64_t num_samples, struct rng_state* state, uint64_t* ret)
{
	assert(num_samples <= bound);

	// Floyd's algorithm
	for (uint64_t i = 0; i < num_samples; i++)
	{
		const uint64_t j = bound - num_samples + 1 + i;
		uint64_t t = rand_interval(j, state);
		// test whether 't' has already been drawn
		bool drawn = false;
		// TODO: faster search using a hash table or tree
		for (uint16_t k = 0; k < i; k++) {
			if (ret[k] == t) {
				drawn = true;
				break;
			}
		}
		ret[i] = (drawn ? j - 1 : t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed pseudo-random number from the interval [0, 1)
/// that has been rounded down to the nearest multiple of 1/2^{64}.
///
double randu(struct rng_state* state)
{
	return ldexp((double)rand_uint64(state), -64);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate a uniformly distributed pseudo-random number from the interval [0, 1)
/// that has been rounded down to the nearest multiple of 1/2^{32}.
///
float randuf(struct rng_state* state)
{
	return ldexpf((float)rand_uint32(state), -32);
}


//________________________________________________________________________________________________________________________
///
/// \brief Draw a standard normal (Gaussian) random number, using the Box-Muller transform.
///
/// Real double precision version.
///
double randn(struct rng_state* state)
{
	const double M_2PI = 6.2831853071795864769;

	double u1 = randu(state);
	double u2 = randu(state);
	// use log(u1) instead of log(1 - u1) to avoid loss of significant digits
	if (u1 == 0) { u1 = 1; }
	return sqrt(-2 * log(u1)) * sin(M_2PI * u2);
}


//________________________________________________________________________________________________________________________
///
/// \brief Draw a standard normal (Gaussian) random number, using the Box-Muller transform.
///
/// Real single precision version.
///
float randnf(struct rng_state* state)
{
	const float M_2PI = 6.2831853071795864769;

	float u1 = randuf(state);
	float u2 = randuf(state);
	// use log(u1) instead of log(1 - u1) to avoid loss of significant digits
	if (u1 == 0) { u1 = 1; }
	return sqrtf(-2 * logf(u1)) * sinf(M_2PI * u2);
}


//________________________________________________________________________________________________________________________
///
/// \brief Draw a standard complex normal (Gaussian) random number.
///
/// Complex double precision version.
///
double complex crandn(struct rng_state* state)
{
	double x = randn(state);
	double y = randn(state);
	return (x + _Complex_I * y) / sqrt(2.0);
}


//________________________________________________________________________________________________________________________
///
/// \brief Draw a standard complex normal (Gaussian) random number.
///
/// Complex single precision version.
///
float complex crandnf(struct rng_state* state)
{
	float x = randnf(state);
	float y = randnf(state);
	return (x + _Complex_I * y) / sqrtf(2.f);
}
