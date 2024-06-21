/// \file integer_linear_algebra.c
/// \brief Linear algebra operations for integer matrix and vector entries.

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "integer_linear_algebra.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the matrix-vector product a @ x and store the result in 'b'.
///
void integer_gemv(const int m, const int n, const int* restrict a, const int* restrict x, int* restrict b)
{
	for (int i = 0; i < m; i++)
	{
		b[i] = 0;
		for (int j = 0; j < n; j++) {
			b[i] += a[i*n + j] * x[j];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Swap two rows of an integer square matrix.
///
static inline void integer_matrix_swap_rows(const int n, const int i, const int j, int* a)
{
	for (int k = 0; k < n; k++)
	{
		int tmp    = a[i*n + k];
		a[i*n + k] = a[j*n + k];
		a[j*n + k] = tmp;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale a row of an integer square matrix.
///
static inline void integer_matrix_scale_row(const int n, const int i, const int s, int* a)
{
	for (int k = 0; k < n; k++) {
		a[i*n + k] *= s;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Subtract row 'j' scaled by 's' from row 'i'.
///
static inline void integer_matrix_subtract_row(const int n, const int i, const int j, const int s, int* a)
{
	for (int k = 0; k < n; k++) {
		a[i*n + k] -= s * a[j*n + k];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Initialize an integer square matrix as identity.
///
static void integer_matrix_identity(const int n, int* a)
{
	memset(a, 0, n*n * sizeof(int));
	for (int i = 0; i < n; i++) {
		a[i*n + i] = 1;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Hermite normal form and the associated unimodular matrix of a square matrix, returning 0 if successful,
/// and interrupting the algorithm if the matrix is singular with return value -1.
///
int integer_hermite_normal_form(const int n, const int* restrict a, int* restrict h, int* restrict u)
{
	memcpy(h, a, n*n * sizeof(int));

	integer_matrix_identity(n, u);

	// implementation is straightforward, probably not most efficient algorithm

	for (int k = 0; k < n; k++)
	{
		int i_max;
		while (1)
		{
			// search for largest element in k-th column, starting from entry at (k, k)
			i_max = k;
			int p = abs(h[k*n + k]);
			for (int i = k + 1; i < n; i++) {
				if (abs(h[i*n + k]) > p) {
					i_max = i;
					p = abs(h[i*n + k]);
				}
			}
			if (p == 0) {
				// matrix is singular
				return -1;
			}

			// search for second-largest element in k-th column, starting from entry at (k, k)
			int i_snd = -1;
			int q = 0;
			for (int i = k; i < n; i++) {
				if (i == i_max) {
					continue;
				}
				if (abs(h[i*n + k]) > q) {
					i_snd = i;
					q = abs(h[i*n + k]);
				}
			}

			if (q == 0) {
				break;
			}

			// subtract multiple of row 'i_snd' from row 'i_max'
			assert(h[i_snd*n + k] != 0);
			int s = h[i_max*n + k] / h[i_snd*n + k];
			integer_matrix_subtract_row(n, i_max, i_snd, s, h);
			integer_matrix_subtract_row(n, i_max, i_snd, s, u);  // record operation in 'u'
		}

		// swap pivot row with current row
		if (i_max != k) {
			integer_matrix_swap_rows(n, k, i_max, h);
			integer_matrix_swap_rows(n, k, i_max, u);  // record operation in 'u'
		}

		// multiply by (-1) if necessary to ensure positive sign of pivot element
		if (h[k*n + k] < 0) {
			integer_matrix_scale_row(n, k, -1, h);
			integer_matrix_scale_row(n, k, -1, u);  // record operation in 'u'
		}

		// pivot element must now be positive
		assert(h[k*n + k] > 0);

		// subtract pivot row from upper rows
		for (int i = 0; i < k; i++)
		{
			int s = h[i*n + k] / h[k*n + k];
			// ensure that h[i*n + k] will be non-negative
			if (h[i*n + k] - s*h[k*n + k] < 0) {
				s--;
			}
			if (s != 0) {
				integer_matrix_subtract_row(n, i, k, s, h);
				integer_matrix_subtract_row(n, i, k, s, u);  // record operation in 'u'
			}
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Solve the integer linear system 'a @ x = b' for 'x' by back-substutution, assuming that 'a' is an upper triangular matrix with non-zero diagonal entries.
/// Returns 0 if successful, and -1 of no integer solution can be found.
///
int integer_backsubstitute(const int* restrict a, const int n, const int* restrict b, int* restrict x)
{
	for (int i = n - 1; i >= 0; i--)
	{
		int z = b[i];
		for (int j = i + 1; j < n; j++) {
			z -= a[i*n + j] * x[j];
		}

		if (z % a[i*n + i] != 0) {
			// linear system has no integer solution
			return -1;
		}
		x[i] = z / a[i*n + i];
	}

	return 0;
}
