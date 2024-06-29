#pragma once

#include "numeric.h"


typedef void (*generic_func_s)(const float* restrict x, void* params, float* restrict y);

typedef void (*generic_func_d)(const double* restrict x, void* params, double* restrict y);

typedef void (*generic_func_c)(const scomplex* restrict x, void* params, scomplex* restrict y);

typedef void (*generic_func_z)(const dcomplex* restrict x, void* params, dcomplex* restrict y);


void numerical_gradient_forward_s(generic_func_s f, void* params, const long n, const float* restrict x, const float* restrict dir, const long m, const float h, float* restrict grad);
void numerical_gradient_backward_s(generic_func_s f, void* params, const long n, const float* restrict x, const long m, const float* restrict dy, const float h, float* restrict grad);

void numerical_gradient_forward_d(generic_func_d f, void* params, const long n, const double* restrict x, const double* restrict dir, const long m, const double h, double* restrict grad);
void numerical_gradient_backward_d(generic_func_d f, void* params, const long n, const double* restrict x, const long m, const double* restrict dy, const double h, double* restrict grad);

void numerical_gradient_forward_c(generic_func_c f, void* params, const long n, const scomplex* restrict x, const scomplex* restrict dir, const long m, const float h, scomplex* restrict grad);
void numerical_gradient_backward_c(generic_func_c f, void* params, const long n, const scomplex* restrict x, const long m, const scomplex* restrict dy, const float h, scomplex* restrict grad);

void numerical_gradient_forward_z(generic_func_z f, void* params, const long n, const dcomplex* restrict x, const dcomplex* restrict dir, const long m, const double h, dcomplex* restrict grad);
void numerical_gradient_backward_z(generic_func_z f, void* params, const long n, const dcomplex* restrict x, const long m, const dcomplex* restrict dy, const double h, dcomplex* restrict grad);
