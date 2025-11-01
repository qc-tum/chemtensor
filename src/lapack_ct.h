// LAPACK function declarations on various platforms.

#pragma once


#ifdef LAPACK_H_AVAILABLE


#ifdef __APPLE__

#define lapack_int __LAPACK_int

// The following LAPACK function are currently used by chemtensor.

// QR decomposition

#define LAPACK_sgeqrf sgeqrf_
#define LAPACK_sorgqr sorgqr_
#define LAPACK_dgeqrf dgeqrf_
#define LAPACK_dorgqr dorgqr_
#define LAPACK_cgeqrf cgeqrf_
#define LAPACK_cungqr cungqr_
#define LAPACK_zgeqrf zgeqrf_
#define LAPACK_zungqr zungqr_

// RQ decomposition

#define LAPACK_sgerqf sgerqf_
#define LAPACK_sorgrq sorgrq_
#define LAPACK_dgerqf dgerqf_
#define LAPACK_dorgrq dorgrq_
#define LAPACK_cgerqf cgerqf_
#define LAPACK_cungrq cungrq_
#define LAPACK_zgerqf zgerqf_
#define LAPACK_zungrq zungrq_

// eigendecomposition of general symmetric matrices

#define LAPACK_dsyev dsyev_

// eigendecomposition of general symmetric matrices, using a divide and conquer algorithm

#define LAPACK_ssyevd ssyevd_
#define LAPACK_dsyevd dsyevd_
#define LAPACK_cheevd cheevd_
#define LAPACK_zheevd zheevd_

// eigendecomposition of symmetric tridiagonal matrices

#define LAPACK_dsteqr dsteqr_

// singular value decomposition of general matrices

#define LAPACK_sgesvd sgesvd_
#define LAPACK_dgesvd dgesvd_
#define LAPACK_cgesvd cgesvd_
#define LAPACK_zgesvd zgesvd_

#endif  // __APPLE__

#include <lapack.h>


#else  // LAPACK_H_AVAILABLE not defined


// Substitute of lapack.h in case this header file is not available (e.g., on some older Linux distributions).

#include <stdlib.h>
#include <stdint.h>


#ifndef LAPACK_GLOBAL
#if defined(LAPACK_GLOBAL_PATTERN_LC) || defined(ADD_)
#define LAPACK_GLOBAL(lcname,UCNAME)  lcname##_
#elif defined(LAPACK_GLOBAL_PATTERN_UC) || defined(UPPER)
#define LAPACK_GLOBAL(lcname,UCNAME)  UCNAME
#elif defined(LAPACK_GLOBAL_PATTERN_MC) || defined(NOCHANGE)
#define LAPACK_GLOBAL(lcname,UCNAME)  lcname
#else
#define LAPACK_GLOBAL(lcname,UCNAME)  lcname##_
#endif
#endif

#ifndef lapack_int
#if defined(LAPACK_ILP64)
#define lapack_int        int64_t
#else
#define lapack_int        int32_t
#endif
#endif

#define lapack_complex_float    float _Complex
#define lapack_complex_double   double _Complex

// It seems all current Fortran compilers put strlen at end.
// Some historical compilers put strlen after the str argument
// or make the str argument into a struct.
#define LAPACK_FORTRAN_STRLEN_END


// QR decomposition

#define LAPACK_sgeqrf LAPACK_GLOBAL(sgeqrf,SGEQRF)
void LAPACK_sgeqrf(
	lapack_int const* m, lapack_int const* n,
	float* A, lapack_int const* lda,
	float* tau,
	float* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_sorgqr LAPACK_GLOBAL(sorgqr,SORGQR)
void LAPACK_sorgqr(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	float* A, lapack_int const* lda,
	float const* tau,
	float* work, lapack_int const* lwork,
	lapack_int* info);


#define LAPACK_dgeqrf LAPACK_GLOBAL(dgeqrf,DGEQRF)
void LAPACK_dgeqrf(
	lapack_int const* m, lapack_int const* n,
	double* A, lapack_int const* lda,
	double* tau,
	double* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_dorgqr LAPACK_GLOBAL(dorgqr,DORGQR)
void LAPACK_dorgqr(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	double* A, lapack_int const* lda,
	double const* tau,
	double* work, lapack_int const* lwork,
	lapack_int* info);


#define LAPACK_cgeqrf LAPACK_GLOBAL(cgeqrf,CGEQRF)
void LAPACK_cgeqrf(
	lapack_int const* m, lapack_int const* n,
	lapack_complex_float* A, lapack_int const* lda,
	lapack_complex_float* tau,
	lapack_complex_float* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_cungqr LAPACK_GLOBAL(cungqr,CUNGQR)
void LAPACK_cungqr(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	lapack_complex_float* A, lapack_int const* lda,
	lapack_complex_float const* tau,
	lapack_complex_float* work, lapack_int const* lwork,
	lapack_int* info);

	
#define LAPACK_zgeqrf LAPACK_GLOBAL(zgeqrf,ZGEQRF)
void LAPACK_zgeqrf(
	lapack_int const* m, lapack_int const* n,
	lapack_complex_double* A, lapack_int const* lda,
	lapack_complex_double* tau,
	lapack_complex_double* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_zungqr LAPACK_GLOBAL(zungqr,ZUNGQR)
void LAPACK_zungqr(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	lapack_complex_double* A, lapack_int const* lda,
	lapack_complex_double const* tau,
	lapack_complex_double* work, lapack_int const* lwork,
	lapack_int* info);


// RQ decomposition

#define LAPACK_sgerqf LAPACK_GLOBAL(sgerqf,SGERQF)
void LAPACK_sgerqf(
	lapack_int const* m, lapack_int const* n,
	float* A, lapack_int const* lda,
	float* tau,
	float* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_sorgrq LAPACK_GLOBAL(sorgrq,SORGRQ)
void LAPACK_sorgrq(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	float* A, lapack_int const* lda,
	float const* tau,
	float* work, lapack_int const* lwork,
	lapack_int* info);


#define LAPACK_dgerqf LAPACK_GLOBAL(dgerqf,DGERQF)
void LAPACK_dgerqf(
	lapack_int const* m, lapack_int const* n,
	double* A, lapack_int const* lda,
	double* tau,
	double* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_dorgrq LAPACK_GLOBAL(dorgrq,DORGRQ)
void LAPACK_dorgrq(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	double* A, lapack_int const* lda,
	double const* tau,
	double* work, lapack_int const* lwork,
	lapack_int* info);


#define LAPACK_cgerqf LAPACK_GLOBAL(cgerqf,CGERQF)
void LAPACK_cgerqf(
	lapack_int const* m, lapack_int const* n,
	lapack_complex_float* A, lapack_int const* lda,
	lapack_complex_float* tau,
	lapack_complex_float* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_cungrq LAPACK_GLOBAL(cungrq,CUNGRQ)
void LAPACK_cungrq(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	lapack_complex_float* A, lapack_int const* lda,
	lapack_complex_float const* tau,
	lapack_complex_float* work, lapack_int const* lwork,
	lapack_int* info);


#define LAPACK_zgerqf LAPACK_GLOBAL(zgerqf,ZGERQF)
void LAPACK_zgerqf(
	lapack_int const* m, lapack_int const* n,
	lapack_complex_double* A, lapack_int const* lda,
	lapack_complex_double* tau,
	lapack_complex_double* work, lapack_int const* lwork,
	lapack_int* info);

#define LAPACK_zungrq LAPACK_GLOBAL(zungrq,ZUNGRQ)
void LAPACK_zungrq(
	lapack_int const* m, lapack_int const* n, lapack_int const* k,
	lapack_complex_double* A, lapack_int const* lda,
	lapack_complex_double const* tau,
	lapack_complex_double* work, lapack_int const* lwork,
	lapack_int* info);


// eigendecomposition of general symmetric matrices

#define LAPACK_dsyev_base LAPACK_GLOBAL(dsyev,DSYEV)
void LAPACK_dsyev_base(
	char const* jobz, char const* uplo,
	lapack_int const* n,
	double* A, lapack_int const* lda,
	double* W,
	double* work, lapack_int const* lwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_dsyev(...) LAPACK_dsyev_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_dsyev(...) LAPACK_dsyev_base(__VA_ARGS__)
#endif


// eigendecomposition of general symmetric matrices, using a divide and conquer algorithm

#define LAPACK_ssyevd_base LAPACK_GLOBAL(ssyevd,SSYEVD)
void LAPACK_ssyevd_base(
	char const* jobz, char const* uplo,
	lapack_int const* n,
	float* A, lapack_int const* lda,
	float* W,
	float* work, lapack_int const* lwork,
	lapack_int* iwork, lapack_int const* liwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_ssyevd(...) LAPACK_ssyevd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_ssyevd(...) LAPACK_ssyevd_base(__VA_ARGS__)
#endif


#define LAPACK_dsyevd_base LAPACK_GLOBAL(dsyevd,DSYEVD)
void LAPACK_dsyevd_base(
	char const* jobz, char const* uplo,
	lapack_int const* n,
	double* A, lapack_int const* lda,
	double* W,
	double* work, lapack_int const* lwork,
	lapack_int* iwork, lapack_int const* liwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_dsyevd(...) LAPACK_dsyevd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_dsyevd(...) LAPACK_dsyevd_base(__VA_ARGS__)
#endif


#define LAPACK_cheevd_base LAPACK_GLOBAL(cheevd,CHEEVD)
void LAPACK_cheevd_base(
	char const* jobz, char const* uplo,
	lapack_int const* n,
	lapack_complex_float* A, lapack_int const* lda,
	float* W,
	lapack_complex_float* work, lapack_int const* lwork,
	float* rwork, lapack_int const* lrwork,
	lapack_int* iwork, lapack_int const* liwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_cheevd(...) LAPACK_cheevd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_cheevd(...) LAPACK_cheevd_base(__VA_ARGS__)
#endif


#define LAPACK_zheevd_base LAPACK_GLOBAL(zheevd,ZHEEVD)
void LAPACK_zheevd_base(
	char const* jobz, char const* uplo,
	lapack_int const* n,
	lapack_complex_double* A, lapack_int const* lda,
	double* W,
	lapack_complex_double* work, lapack_int const* lwork,
	double* rwork, lapack_int const* lrwork,
	lapack_int* iwork, lapack_int const* liwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_zheevd(...) LAPACK_zheevd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_zheevd(...) LAPACK_zheevd_base(__VA_ARGS__)
#endif


// eigendecomposition of symmetric tridiagonal matrices

#define LAPACK_dsteqr_base LAPACK_GLOBAL(dsteqr,DSTEQR)
void LAPACK_dsteqr_base(
	char const* compz,
	lapack_int const* n,
	double* D,
	double* E,
	double* Z, lapack_int const* ldz,
	double* work,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_dsteqr(...) LAPACK_dsteqr_base(__VA_ARGS__, 1)
#else
	#define LAPACK_dsteqr(...) LAPACK_dsteqr_base(__VA_ARGS__)
#endif


// singular value decomposition of general matrices

#define LAPACK_sgesvd_base LAPACK_GLOBAL(sgesvd,SGESVD)
void LAPACK_sgesvd_base(
	char const* jobu, char const* jobvt,
	lapack_int const* m, lapack_int const* n,
	float* A, lapack_int const* lda,
	float* S,
	float* U, lapack_int const* ldu,
	float* VT, lapack_int const* ldvt,
	float* work, lapack_int const* lwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_sgesvd(...) LAPACK_sgesvd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_sgesvd(...) LAPACK_sgesvd_base(__VA_ARGS__)
#endif


#define LAPACK_dgesvd_base LAPACK_GLOBAL(dgesvd,DGESVD)
void LAPACK_dgesvd_base(
	char const* jobu, char const* jobvt,
	lapack_int const* m, lapack_int const* n,
	double* A, lapack_int const* lda,
	double* S,
	double* U, lapack_int const* ldu,
	double* VT, lapack_int const* ldvt,
	double* work, lapack_int const* lwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_dgesvd(...) LAPACK_dgesvd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_dgesvd(...) LAPACK_dgesvd_base(__VA_ARGS__)
#endif


#define LAPACK_cgesvd_base LAPACK_GLOBAL(cgesvd,CGESVD)
void LAPACK_cgesvd_base(
	char const* jobu, char const* jobvt,
	lapack_int const* m, lapack_int const* n,
	lapack_complex_float* A, lapack_int const* lda,
	float* S,
	lapack_complex_float* U, lapack_int const* ldu,
	lapack_complex_float* VT, lapack_int const* ldvt,
	lapack_complex_float* work, lapack_int const* lwork,
	float* rwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_cgesvd(...) LAPACK_cgesvd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_cgesvd(...) LAPACK_cgesvd_base(__VA_ARGS__)
#endif


#define LAPACK_zgesvd_base LAPACK_GLOBAL(zgesvd,ZGESVD)
void LAPACK_zgesvd_base(
	char const* jobu, char const* jobvt,
	lapack_int const* m, lapack_int const* n,
	lapack_complex_double* A, lapack_int const* lda,
	double* S,
	lapack_complex_double* U, lapack_int const* ldu,
	lapack_complex_double* VT, lapack_int const* ldvt,
	lapack_complex_double* work, lapack_int const* lwork,
	double* rwork,
	lapack_int* info
#ifdef LAPACK_FORTRAN_STRLEN_END
	, size_t, size_t
#endif
);
#ifdef LAPACK_FORTRAN_STRLEN_END
	#define LAPACK_zgesvd(...) LAPACK_zgesvd_base(__VA_ARGS__, 1, 1)
#else
	#define LAPACK_zgesvd(...) LAPACK_zgesvd_base(__VA_ARGS__)
#endif


#endif
