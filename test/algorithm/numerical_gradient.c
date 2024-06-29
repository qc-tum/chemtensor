#include <complex.h>
#include "numerical_gradient.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the forward-mode gradient via the difference quotient (f(x + h dir) - f(x - h dir)) / (2 h).
///
/// Real single precision version.
///
void numerical_gradient_forward_s(generic_func_s f, void* params, const long n, const float* restrict x, const float* restrict dir, const long m, const float h, float* restrict grad)
{
	float* xd = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(float));
	float* yp = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(float));
	float* yn = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(float));

	// yp = f(x + h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] + h * dir[i];
	}
	f(xd, params, yp);

	// yn = f(x - h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] - h * dir[i];
	}
	f(xd, params, yn);

	for (long i = 0; i < m; i++)
	{
		grad[i] = (yp[i] - yn[i]) / (2 * h);
	}

	aligned_free(yn);
	aligned_free(yp);
	aligned_free(xd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient via the difference quotient (f(x + h e_i) - f(x - h e_i)) / (2 h).
///
/// Real single precision version.
///
void numerical_gradient_backward_s(generic_func_s f, void* params, const long n, const float* restrict x, const long m, const float* restrict dy, const float h, float* restrict grad)
{
	float* xmod = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(float));
	memcpy(xmod, x, n * sizeof(float));

	// y = f(x)
	float* y = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(float));

	for (long i = 0; i < n; i++)
	{
		float ydotdy;

		// f(x + h e_i)
		xmod[i] = x[i] + h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] = ydotdy;

		// f(x - h e_i)
		xmod[i] = x[i] - h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= ydotdy;

		grad[i] /= (2 * h);

		// restore original value
		xmod[i] = x[i];
	}

	aligned_free(y);
	aligned_free(xmod);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the forward-mode gradient via the difference quotient (f(x + h dir) - f(x - h dir)) / (2 h).
///
/// Real double precision version.
///
void numerical_gradient_forward_d(generic_func_d f, void* params, const long n, const double* restrict x, const double* restrict dir, const long m, const double h, double* restrict grad)
{
	double* xd = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	double* yp = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	double* yn = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));

	// yp = f(x + h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] + h * dir[i];
	}
	f(xd, params, yp);

	// yn = f(x - h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] - h * dir[i];
	}
	f(xd, params, yn);

	for (long i = 0; i < m; i++)
	{
		grad[i] = (yp[i] - yn[i]) / (2 * h);
	}

	aligned_free(yn);
	aligned_free(yp);
	aligned_free(xd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient via the difference quotient (f(x + h e_i) - f(x - h e_i)) / (2 h).
///
/// Real double precision version.
///
void numerical_gradient_backward_d(generic_func_d f, void* params, const long n, const double* restrict x, const long m, const double* restrict dy, const double h, double* restrict grad)
{
	double* xmod = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	memcpy(xmod, x, n * sizeof(double));

	// y = f(x)
	double* y = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));

	for (long i = 0; i < n; i++)
	{
		double ydotdy;

		// f(x + h e_i)
		xmod[i] = x[i] + h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] = ydotdy;

		// f(x - h e_i)
		xmod[i] = x[i] - h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= ydotdy;

		grad[i] /= (2 * h);

		// restore original value
		xmod[i] = x[i];
	}

	aligned_free(y);
	aligned_free(xmod);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the forward-mode gradient via the difference quotient (f(x + h dir) - f(x - h dir)) / (2 h).
///
/// Complex single precision version.
///
void numerical_gradient_forward_c(generic_func_c f, void* params, const long n, const scomplex* restrict x, const scomplex* restrict dir, const long m, const float h, scomplex* restrict grad)
{
	scomplex* xd = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(scomplex));
	scomplex* yp = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(scomplex));
	scomplex* yn = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(scomplex));

	// yp = f(x + h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] + h * dir[i];
	}
	f(xd, params, yp);

	// yn = f(x - h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] - h * dir[i];
	}
	f(xd, params, yn);

	for (long i = 0; i < m; i++)
	{
		grad[i] = (yp[i] - yn[i]) / (2 * h);
	}

	aligned_free(yn);
	aligned_free(yp);
	aligned_free(xd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient (Wirtinger convention) via difference quotients.
///
/// Complex single precision version.
///
void numerical_gradient_backward_c(generic_func_c f, void* params, const long n, const scomplex* restrict x, const long m, const scomplex* restrict dy, const float h, scomplex* restrict grad)
{
	scomplex* xmod = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(scomplex));
	memcpy(xmod, x, n * sizeof(scomplex));

	// y = f(x)
	scomplex* y = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(scomplex));

	for (long i = 0; i < n; i++)
	{
		scomplex ydotdy;

		// f(x + h e_i)
		xmod[i] = x[i] + h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] = ydotdy;

		// f(x - h e_i)
		xmod[i] = x[i] - h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= ydotdy;

		// h -> i*h

		// f(x + i h e_i)
		xmod[i] = x[i] + _Complex_I * h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= _Complex_I * ydotdy;

		// f(x - i h e_i)
		xmod[i] = x[i] - _Complex_I * h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] += _Complex_I * ydotdy;

		grad[i] /= (4 * h);

		// restore original value
		xmod[i] = x[i];
	}

	aligned_free(y);
	aligned_free(xmod);
}



//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the forward-mode gradient via the difference quotient (f(x + h dir) - f(x - h dir)) / (2 h).
///
/// Complex double precision version.
///
void numerical_gradient_forward_z(generic_func_z f, void* params, const long n, const dcomplex* restrict x, const dcomplex* restrict dir, const long m, const double h, dcomplex* restrict grad)
{
	dcomplex* xd = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(dcomplex));
	dcomplex* yp = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(dcomplex));
	dcomplex* yn = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(dcomplex));

	// yp = f(x + h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] + h * dir[i];
	}
	f(xd, params, yp);

	// yn = f(x - h*dir)
	for (long i = 0; i < n; i++)
	{
		xd[i] = x[i] - h * dir[i];
	}
	f(xd, params, yn);

	for (long i = 0; i < m; i++)
	{
		grad[i] = (yp[i] - yn[i]) / (2 * h);
	}

	aligned_free(yn);
	aligned_free(yp);
	aligned_free(xd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient (Wirtinger convention) via difference quotients.
///
/// Complex double precision version.
///
void numerical_gradient_backward_z(generic_func_z f, void* params, const long n, const dcomplex* restrict x, const long m, const dcomplex* restrict dy, const double h, dcomplex* restrict grad)
{
	dcomplex* xmod = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(dcomplex));
	memcpy(xmod, x, n * sizeof(dcomplex));

	// y = f(x)
	dcomplex* y = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(dcomplex));

	for (long i = 0; i < n; i++)
	{
		dcomplex ydotdy;

		// f(x + h e_i)
		xmod[i] = x[i] + h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] = ydotdy;

		// f(x - h e_i)
		xmod[i] = x[i] - h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= ydotdy;

		// h -> i*h

		// f(x + i h e_i)
		xmod[i] = x[i] + _Complex_I * h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= _Complex_I * ydotdy;

		// f(x - i h e_i)
		xmod[i] = x[i] - _Complex_I * h;
		f(xmod, params, y);
		ydotdy = 0;
		for (long j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] += _Complex_I * ydotdy;

		grad[i] /= (4 * h);

		// restore original value
		xmod[i] = x[i];
	}

	aligned_free(y);
	aligned_free(xmod);
}
