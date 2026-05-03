/// \file wigner.c
/// \brief Evaluate the Wigner D-matrix.

#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "wigner.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the Wigner "small" d-matrix for quantum number 'j' and rotation angle 'theta' (y-axis rotation convention).
/// The 'm' quantum numbers are enumerated as -j, ..., j.
/// 'j' is the logical quantum number times 2.
/// 'w' must point to an array with (j + 1)^2 entries.
///
void wigner_small_d(const qnumber j, const double theta, double* w)
{
	assert(j >= 0);

	switch (j)
	{
		case 0:  // logical quantum number 0
		{
			w[0] = 1;
			break;
		}
		case 1:  // logical quantum number 1/2
		{
			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);

			w[0] =  c;    w[1] =  s;
			w[2] = -s;    w[3] =  c;

			break;
		}
		case 2:  // logical quantum number 1
		{
			const double sqrt2 = 1.4142135623730950488;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);
			const double c2 = cos(theta);
			const double cc = c * c;
			const double cs = sqrt2 * c * s;
			const double ss = s * s;

			w[0] =  cc;    w[1] =  cs;    w[2] =  ss;
			w[3] = -cs;    w[4] =  c2;    w[5] =  cs;
			w[6] =  ss;    w[7] = -cs;    w[8] =  cc;

			break;
		}
		case 3:  // logical quantum number 3/2
		{
			const double sqrt3 = 1.7320508075688772935;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);
			const double c2 = cos(theta);
			const double ccc = c * c * c;
			const double ccs = sqrt3 * c * c * s;
			const double css = sqrt3 * c * s * s;
			const double sss = s * s * s;
			const double c3c21 = 0.5 * c * (3*c2 - 1);
			const double s3c21 = 0.5 * s * (3*c2 + 1);

			w[ 0] =  ccc;    w[ 1] =  ccs;    w[ 2] =  css;    w[ 3] =  sss;
			w[ 4] = -ccs;    w[ 5] =  c3c21;  w[ 6] =  s3c21;  w[ 7] =  css;
			w[ 8] =  css;    w[ 9] = -s3c21;  w[10] =  c3c21;  w[11] =  ccs;
			w[12] = -sss;    w[13] =  css;    w[14] = -ccs;    w[15] =  ccc;

			break;
		}
		case 4:  // logical quantum number 2
		{
			const double sqrt6 = 2.4494897427831780982;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);
			const double c2 = cos(theta);
			const double cc = c * c;
			const double ss = s * s;
			const double cccc = cc * cc;
			const double cccs = 2 * c * cc * s;
			const double ccss = sqrt6 * cc * ss;
			const double csss = 2 * c * s * ss;
			const double ssss = ss * ss;
			const double cc2s = sqrt6 * c * c2 * s;
			const double cc2c21 = cc * (2 * c2 - 1);
			const double ss2c21 = ss * (2 * c2 + 1);
			const double c2c213 = 0.5 * (3 * c2 * c2 - 1);

			w[ 0] =  cccc;    w[ 1] =  cccs;    w[ 2] =  ccss;    w[ 3] =  csss;    w[ 4] =  ssss;
			w[ 5] = -cccs;    w[ 6] =  cc2c21;  w[ 7] =  cc2s;    w[ 8] =  ss2c21;  w[ 9] =  csss;
			w[10] =  ccss;    w[11] = -cc2s;    w[12] =  c2c213;  w[13] =  cc2s;    w[14] =  ccss;
			w[15] = -csss;    w[16] =  ss2c21;  w[17] = -cc2s;    w[18] =  cc2c21;  w[19] =  cccs;
			w[20] =  ssss;    w[21] = -csss;    w[22] =  ccss;    w[23] = -cccs;    w[24] =  cccc;

			break;
		}
		case 5:  // logical quantum number 5/2
		{
			const double invsq2 = 0.70710678118654752440;
			const double sqrt5  = 2.2360679774997896964;
			const double sqrt10 = 3.1622776601683793320;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);
			const double c2 = cos(theta);
			const double c4 = cos(2 * theta);
			const double cc = c * c;
			const double ss = s * s;
			const double ccc = c * cc;
			const double sss = s * ss;
			const double cccc = cc * cc;
			const double ssss = ss * ss;
			const double ccccc = c * cccc;
			const double ccccs = sqrt5  * cccc * s;
			const double cccss = sqrt10 * ccc * ss;
			const double ccsss = sqrt10 * cc * sss;
			const double cssss = sqrt5  * c * ssss;
			const double sssss = s * ssss;
			const double ccc5c23 = 0.5 * ccc * (5 * c2 - 3);
			const double sss5c23 = 0.5 * sss * (5 * c2 + 3);
			const double ccs5c21 = invsq2 * cc * s * (5 * c2 - 1);
			const double css5c21 = invsq2 * c * ss * (5 * c2 + 1);
			const double c34c25c4 = 0.25 * c * (3 - 4 * c2 + 5 * c4);
			const double s34c25c4 = 0.25 * s * (3 + 4 * c2 + 5 * c4);

			w[ 0] =  ccccc;    w[ 1] =  ccccs;    w[ 2] =  cccss;    w[ 3] =  ccsss;    w[ 4] =  cssss;    w[ 5] =  sssss;
			w[ 6] = -ccccs;    w[ 7] =  ccc5c23;  w[ 8] =  ccs5c21;  w[ 9] =  css5c21;  w[10] =  sss5c23;  w[11] =  cssss;
			w[12] =  cccss;    w[13] = -ccs5c21;  w[14] =  c34c25c4; w[15] =  s34c25c4; w[16] =  css5c21;  w[17] =  ccsss;
			w[18] = -ccsss;    w[19] =  css5c21;  w[20] = -s34c25c4; w[21] =  c34c25c4; w[22] =  ccs5c21;  w[23] =  cccss;
			w[24] =  cssss;    w[25] = -sss5c23;  w[26] =  css5c21;  w[27] = -ccs5c21;  w[28] =  ccc5c23;  w[29] =  ccccs;
			w[30] = -sssss;    w[31] =  cssss;    w[32] = -ccsss;    w[33] =  cccss;    w[34] = -ccccs;    w[35] =  ccccc;

			break;
		}
		default:
		{
			// 'j' not supported yet
			fprintf(stderr, "quantum number j = %i in Wigner \"small\" d-matrix evaluation not supported yet\n", j);
			assert(false);
			break;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the Wigner D-matrix for quantum number 'j' and rotation angles 'psi', 'theta', 'phi' (Euler z-y-z rotation convention).
/// The 'm' quantum numbers are enumerated as -j, ..., j.
/// 'j' is the logical quantum number times 2.
/// 'w' must point to an array with (j + 1)^2 entries.
///
void wigner_d(const qnumber j, const double psi, const double theta, const double phi, dcomplex* w)
{
	assert(j >= 0);

	double* wy = ct_malloc((j + 1) * (j + 1) * sizeof(double));
	wigner_small_d(j, theta, wy);

	dcomplex* phases_phi = ct_malloc((j + 1) * sizeof(dcomplex));
	dcomplex* phases_psi = ct_malloc((j + 1) * sizeof(dcomplex));
	for (int im = 0; im < j + 1; im++)
	{
		const double s = (-0.5*j + im) * phi;
		const double t = (-0.5*j + im) * psi;
		phases_phi[im] = cos(s) - _Complex_I * sin(s);
		phases_psi[im] = cos(t) - _Complex_I * sin(t);
	}

	for (int im = 0; im < j + 1; im++) {
		for (int in = 0; in < j + 1; in++) {
			w[im * (j + 1) + in] = wy[im * (j + 1) + in] * phases_psi[im] * phases_phi[in];
		}
	}

	ct_free(phases_psi);
	ct_free(phases_phi);
	ct_free(wy);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the real-valued Wigner D-matrix for quantum number 'j' and
/// rotation angles 'psi', 'theta', 'phi' (Euler z-y-z rotation convention).
/// The real form is designed so that multiplication with the spherical harmonics
/// corresponds to an Euler rotation of the unit vector that parametrizes them.
/// The 'm' quantum numbers are enumerated as -j, ..., j.
/// 'j' is the logical quantum number times 2, and must be even.
/// 'w' must point to an array with (j + 1)^2 entries.
///
void real_wigner_d(const qnumber j, const double psi, const double theta, const double phi, double* w)
{
	assert(j >= 0);
	assert(j % 2 == 0);

	switch (j)
	{
		case 0:  // logical quantum number 0
		{
			w[0] = 1;
			break;
		}
		case 2:  // logical quantum number 1
		{
			const double c_psi = cos(psi);
			const double s_psi = sin(psi);
			const double c_the = cos(theta);
			const double s_the = sin(theta);
			const double c_phi = cos(phi);
			const double s_phi = sin(phi);

			w[0] =  c_psi * c_phi - s_psi * c_the * s_phi;    w[1] =  s_psi * s_the;    w[2] =  c_psi * s_phi + s_psi * c_the * c_phi;
			w[3] =  s_the * s_phi;                            w[4] =  c_the;            w[5] = -s_the * c_phi;
			w[6] = -c_psi * c_the * s_phi - s_psi * c_phi;    w[7] =  c_psi * s_the;    w[8] =  c_psi * c_the * c_phi - s_psi * s_phi;

			break;
		}
		case 4:  // logical quantum number 2
		{
			const double sqrt3 = 1.7320508075688772935;

			const double c_psi = cos(psi);
			const double s_psi = sin(psi);
			const double c_the = cos(theta);
			const double s_the = sin(theta);
			const double c_phi = cos(phi);
			const double s_phi = sin(phi);

			const double s_the_sq = s_the * s_the;

			// not using double-angle formulas to retain higher precision
			const double c_2psi = cos(2 * psi);
			const double s_2psi = sin(2 * psi);
			const double c_2the = cos(2 * theta);
			const double s_2the = sin(2 * theta);
			const double c_2phi = cos(2 * phi);
			const double s_2phi = sin(2 * phi);

			w[ 0] =  c_the * c_2phi * c_2psi - (3 + c_2the) * c_phi * c_psi * s_phi * s_psi;    w[ 1] =  s_the * (c_phi * c_2psi - 2 * c_the * c_psi * s_phi * s_psi);    w[ 2] =  sqrt3 * c_psi * s_the_sq * s_psi;    w[ 3] =  s_the * (c_2psi * s_phi + c_the * c_phi * s_2psi);         w[ 4] =  c_the * c_2psi * s_2phi + 0.25 * (3 + c_2the) * c_2phi * s_2psi;
			w[ 5] =  s_the * (c_the * s_2phi * s_psi - c_2phi * c_psi);                         w[ 6] =  c_the * c_phi * c_psi - c_2the * s_phi * s_psi;                  w[ 7] =  sqrt3 * c_the * s_the * s_psi;       w[ 8] =  c_the * c_psi * s_phi + c_2the * c_phi * s_psi;            w[ 9] = -s_the * (c_psi * s_2phi + c_the * c_2phi * s_psi);
			w[10] = -sqrt3 * c_phi * s_the_sq * s_phi;                                          w[11] =  sqrt3 * c_the * s_the * s_phi;                                   w[12] =  0.25 * (1 + 3 * c_2the);             w[13] = -sqrt3 * c_the * c_phi * s_the;                             w[14] =  0.5 * sqrt3 * c_2phi * s_the_sq;
			w[15] =  s_the * (c_the * c_psi * s_2phi + c_2phi * s_psi);                         w[16] = -c_2the * c_psi * s_phi - c_the * c_phi * s_psi;                  w[17] =  sqrt3 * c_the * c_psi * s_the;       w[18] =  c_2the * c_phi * c_psi - c_the * s_phi * s_psi;            w[19] = -0.5 * c_2phi * c_psi * s_2the + s_the * s_2phi * s_psi;
			w[20] = -0.25 * (3 + c_2the) * c_2psi * s_2phi - c_the * c_2phi * s_2psi;           w[21] = -s_the * (c_the * c_2psi * s_phi + c_phi * s_2psi);               w[22] =  0.5 * sqrt3 * c_2psi * s_the_sq;     w[23] =  0.5 * c_phi * c_2psi * s_2the - s_the * s_phi * s_2psi;    w[24] =  0.25 * (3 + c_2the) * c_2phi * c_2psi - c_the * s_2phi * s_2psi;

			break;
		}
		default:
		{
			// 'j' not supported yet
			fprintf(stderr, "quantum number j = %i in real-valued Wigner D-matrix evaluation not supported yet\n", j);
			assert(false);
			break;
		}
	}
}
