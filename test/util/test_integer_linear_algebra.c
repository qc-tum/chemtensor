#include "integer_linear_algebra.h"
#include "aligned_memory.h"
#include "util.h"


static int integer_distance(const int n, const int* restrict a, const int* restrict b)
{
	int dist = 0;
	for (int i = 0; i < n; i++)
	{
		dist = imax(dist, abs(a[i] - b[i]));
	}
	return dist;
}


static void integer_matrix_multiply(const int n, const int* restrict a, const int* restrict b, int* restrict ret)
{
	for (int i = 0; i < n; i++)
	{
		for (int k = 0; k < n; k++)
		{
			ret[i*n + k] = 0;
			for (int j = 0; j < n; j++) {
				ret[i*n + k] += a[i*n + j] * b[j*n + k];
			}
		}
	}
}


static int integer_matrix_determinant(const int n, const int* a)
{
	if (n == 1) {
		return a[0];
	}
	if (n == 2) {
		return a[0]*a[3] - a[1]*a[2];
	}

	// Laplace expansion
	int* t = ct_malloc((n - 1)*(n - 1) * sizeof(int));
	int d = 0;
	for (int i = 0; i < n; i++)
	{
		// copy (n - 1) x (n - 1) sub-block
		for (int j = 0; j < i; j++) {
			for (int k = 0; k < n - 1; k++) {
				t[j*(n - 1) + k] = a[j*n + (k + 1)];
			}
		}
		for (int j = i; j < n - 1; j++) {
			for (int k = 0; k < n - 1; k++) {
				t[j*(n - 1) + k] = a[(j + 1)*n + (k + 1)];
			}
		}

		d += (1 - 2 * (i % 2)) * a[i*n] * integer_matrix_determinant(n - 1, t);
	}

	ct_free(t);

	return d;
}


char* test_integer_hermite_normal_form()
{
	const int n = 6;

	const int a[6][6] = {
		{  2, -3,  1,  3,  2,  3 },
		{ -2, -2,  3,  4,  4,  0 },
		{ -2, -4,  4, -2, -1,  2 },
		{  1, -4,  1,  3, -3,  5 },
		{  0,  4, -2,  4,  4, -3 },
		{  1,  3,  2, -4, -3, -4 },
	};

	int* h = ct_malloc(n*n * sizeof(int));
	int* u = ct_malloc(n*n * sizeof(int));
	int ret = integer_hermite_normal_form(n, (const int*)a, h, u);
	if (ret < 0) {
		return "Hermite decomposition incorrectly reports a singular matrix";
	}

	// test whether u @ a = h
	int* ua = ct_malloc(n*n * sizeof(int));
	integer_matrix_multiply(n, u, (const int*)a, ua);
	if (integer_distance(n*n, ua, h) != 0) {
		return "unimodular matrix times input matrix does not result in Hermite normal form matrix";
	}

	// determinant of 'u' must be 1 or -1
	int d = integer_matrix_determinant(n, u);
	if (abs(d) != 1) {
		return "determinant of unimodular matrix must be 1 or -1";
	}

	// reference matrix
	const int h_ref[6][6] = {
		{ 1, 0, 2, 1, 0, 243 },
		{ 0, 1, 1, 1, 0, 231 },
		{ 0, 0, 3, 0, 0, 176 },
		{ 0, 0, 0, 2, 0, 182 },
		{ 0, 0, 0, 0, 1, 190 },
		{ 0, 0, 0, 0, 0, 267 },
	};

	if (integer_distance(n*n, h, (const int*)h_ref) != 0) {
		return "Hermite normal form does not match reference matrix";
	}

	// clean up
	ct_free(ua);
	ct_free(u);
	ct_free(h);

	return 0;
}


char* test_integer_backsubstitute()
{
	const int n = 6;

	// upper triangular matrix
	const int a[6][6] = {
		{  3, -5,  2,  0, -1,  4 },
		{  0,  3,  3,  3, -2,  5 },
		{  0,  0, -2,  4, -3, -1 },
		{  0,  0,  0, -4,  5, -3 },
		{  0,  0,  0,  0,  5,  2 },
		{  0,  0,  0,  0,  0, -1 },
	};

	const int b[6] = { -46, -29, 0, 22, -1, 3 };

	int x[6];
	if (integer_backsubstitute((const int*)a, n, b, x) < 0) {
		return "linear system has integer solution, but return value indicates otherwise";
	}

	// test whether a @ x == b
	int ax[6];
	integer_gemv(n, n, (const int*)a, x, ax);
	if (integer_distance(n, ax, b) != 0) {
		return "linear system solution based on back-substitution is inconsistent";
	}

	const int x_ref[6] = { -5, 2, -4, -2, 1, -3 };
	if (integer_distance(n, x, x_ref) != 0) {
		return "linear system solution based on back-substitution does not match reference";
	}

	// test case with no integer solution
	const int c[6] = { -5, 1, -2, -5, -2, -4 };
	if (integer_backsubstitute((const int*)a, n, c, x) != -1) {
		return "linear system has no integer solution, but return value indicates otherwise";
	}

	return 0;
}
