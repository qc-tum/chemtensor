/// \file integer_linear_algebra.h
/// \brief Linear algebra operations for integer matrix and vector entries.

#pragma once


void integer_gemv(const int m, const int n, const int* a, const int* x, int* b);

int integer_hermite_normal_form(const int n, const int* a, int* h, int* u);

int integer_backsubstitute(const int* a, const int n, const int* b, int* x);
