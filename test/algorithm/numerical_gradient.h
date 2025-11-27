#pragma once

#include "numeric.h"
#include "util.h"


typedef void (*generic_func_s)(const float* x, void* params, float* y);

typedef void (*generic_func_d)(const double* x, void* params, double* y);

typedef void (*generic_func_c)(const scomplex* x, void* params, scomplex* y);

typedef void (*generic_func_z)(const dcomplex* x, void* params, dcomplex* y);


void numerical_gradient_forward_s(generic_func_s f, void* params, const ct_long n, const float* x, const float* dir, const ct_long m, const float h, float* grad);
void numerical_gradient_backward_s(generic_func_s f, void* params, const ct_long n, const float* x, const ct_long m, const float* dy, const float h, float* grad);

void numerical_gradient_forward_d(generic_func_d f, void* params, const ct_long n, const double* x, const double* dir, const ct_long m, const double h, double* grad);
void numerical_gradient_backward_d(generic_func_d f, void* params, const ct_long n, const double* x, const ct_long m, const double* dy, const double h, double* grad);

void numerical_gradient_forward_c(generic_func_c f, void* params, const ct_long n, const scomplex* x, const scomplex* dir, const ct_long m, const float h, scomplex* grad);
void numerical_gradient_backward_c(generic_func_c f, void* params, const ct_long n, const scomplex* x, const ct_long m, const scomplex* dy, const float h, scomplex* grad);

void numerical_gradient_forward_z(generic_func_z f, void* params, const ct_long n, const dcomplex* x, const dcomplex* dir, const ct_long m, const double h, dcomplex* grad);
void numerical_gradient_backward_z(generic_func_z f, void* params, const ct_long n, const dcomplex* x, const ct_long m, const dcomplex* dy, const double h, dcomplex* grad);
