/// \file integer_linear_algebra.h
/// \brief Linear algebra operations for integer matrix and vector entries.

#pragma once


void integer_gemv(const int m, const int n, const int* restrict a, const int* restrict x, int* restrict b);

int integer_hermite_normal_form(const int n, const int* restrict a, int* restrict h, int* restrict u);

int integer_backsubstitute(const int* restrict a, const int n, const int* restrict b, int* restrict x);
