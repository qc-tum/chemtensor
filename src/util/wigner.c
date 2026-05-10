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

			const double c2   = c * c;
			const double s2   = s * s;
			const double cs   = sqrt2 * c * s;
			const double c212 = 2 * c2 - 1;

			w[0] =  c2;    w[1] =  cs;    w[2] =  s2;
			w[3] = -cs;    w[4] =  c212;  w[5] =  cs;
			w[6] =  s2;    w[7] = -cs;    w[8] =  c2;

			break;
		}
		case 3:  // logical quantum number 3/2
		{
			const double sqrt3 = 1.7320508075688772935;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);

			const double c2    = c * c;
			const double s2    = s * s;
			const double c3    = c * c2;
			const double s3    = s * s2;
			const double c2s   = sqrt3 * c2 * s;
			const double cs2   = sqrt3 * c * s2;
			const double cc223 = c * (3 * c2 - 2);
			const double c213s = (3 * c2 - 1) * s;

			w[ 0] =  c3;     w[ 1] =  c2s;    w[ 2] =  cs2;    w[ 3] =  s3;
			w[ 4] = -c2s;    w[ 5] =  cc223;  w[ 6] =  c213s;  w[ 7] =  cs2;
			w[ 8] =  cs2;    w[ 9] = -c213s;  w[10] =  cc223;  w[11] =  c2s;
			w[12] = -s3;     w[13] =  cs2;    w[14] = -c2s;    w[15] =  c3;

			break;
		}
		case 4:  // logical quantum number 2
		{
			const double sqrt6 = 2.4494897427831780982;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);

			const double c2     = c * c;
			const double s2     = s * s;
			const double c4     = c2 * c2;
			const double s4     = s2 * s2;
			const double c3s    = 2 * c * c2 * s;
			const double cs3    = 2 * c * s * s2;
			const double c2s2   = sqrt6 * c2 * s2;
			const double c2c234 = c2 * (4 * c2 - 3);
			const double c214s2 = (4 * c2 - 1) * s2;
			const double cc212s = sqrt6 * c * (2 * c2 - 1) * s;
			const double c2c416 = 1 - 6 * c2 + 6 * c4;

			w[ 0] =  c4;      w[ 1] =  c3s;     w[ 2] =  c2s2;    w[ 3] =  cs3;     w[ 4] =  s4;
			w[ 5] = -c3s;     w[ 6] =  c2c234;  w[ 7] =  cc212s;  w[ 8] =  c214s2;  w[ 9] =  cs3;
			w[10] =  c2s2;    w[11] = -cc212s;  w[12] =  c2c416;  w[13] =  cc212s;  w[14] =  c2s2;
			w[15] = -cs3;     w[16] =  c214s2;  w[17] = -cc212s;  w[18] =  c2c234;  w[19] =  c3s;
			w[20] =  s4;      w[21] = -cs3;     w[22] =  c2s2;    w[23] = -c3s;     w[24] =  c4;

			break;
		}
		case 5:  // logical quantum number 5/2
		{
			const double sqrt2  = 1.4142135623730950488;
			const double sqrt5  = 2.2360679774997896964;
			const double sqrt10 = 3.1622776601683793320;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);

			const double c2      = c * c;
			const double s2      = s * s;
			const double c3      = c * c2;
			const double s3      = s * s2;
			const double c4      = c2 * c2;
			const double s4      = s2 * s2;
			const double c5      = c * c4;
			const double s5      = s * s4;
			const double c4s     = sqrt5  * c4 * s;
			const double c3s2    = sqrt10 * c3 * s2;
			const double c2s3    = sqrt10 * c2 * s3;
			const double cs4     = sqrt5  * c  * s4;
			const double c3c245  = c3 * (5 * c2 - 4);
			const double cc225s2 = sqrt2 * c * (5 * c2 - 2) * s2;
			const double c2c235s = sqrt2 * c2 * (5 * c2 - 3) * s;
			const double c215s3  = (5 * c2 - 1) * s3;
			const double cc2c4   = c * (3 - 12 * c2 + 10 * c4);
			const double c2c4s   = (1 - 8 * c2 + 10 * c4) * s;

			w[ 0] =  c5;       w[ 1] =  c4s;      w[ 2] =  c3s2;     w[ 3] =  c2s3;     w[ 4] =  cs4;      w[ 5] =  s5;
			w[ 6] = -c4s;      w[ 7] =  c3c245;   w[ 8] =  c2c235s;  w[ 9] =  cc225s2;  w[10] =  c215s3;   w[11] =  cs4;
			w[12] =  c3s2;     w[13] = -c2c235s;  w[14] =  cc2c4;    w[15] =  c2c4s;    w[16] =  cc225s2;  w[17] =  c2s3;
			w[18] = -c2s3;     w[19] =  cc225s2;  w[20] = -c2c4s;    w[21] =  cc2c4;    w[22] =  c2c235s;  w[23] =  c3s2;
			w[24] =  cs4;      w[25] = -c215s3;   w[26] =  cc225s2;  w[27] = -c2c235s;  w[28] =  c3c245;   w[29] =  c4s;
			w[30] = -s5;       w[31] =  cs4;      w[32] = -c2s3;     w[33] =  c3s2;     w[34] = -c4s;      w[35] =  c5;

			break;
		}
		case 6:  // logical quantum number 3
		{
			const double sqrt6  = 2.4494897427831780982;
			const double sqrt10 = 3.1622776601683793320;
			const double sqrt12 = 3.4641016151377545871;
			const double sqrt15 = 3.8729833462074168852;
			const double sqrt20 = 4.4721359549995793928;
			const double sqrt30 = 5.4772255750516611346;

			const double c = cos(0.5 * theta);
			const double s = sin(0.5 * theta);

			const double c2      = c * c;
			const double s2      = s * s;
			const double c3      = c * c2;
			const double s3      = s * s2;
			const double c4      = c2 * c2;
			const double s4      = s2 * s2;
			const double c5      = c * c4;
			const double s5      = s * s4;
			const double c6      = c2 * c4;
			const double s6      = s2 * s4;
			const double c5s     = sqrt6  * c5 * s;
			const double c4s2    = sqrt15 * c4 * s2;
			const double c3s3    = sqrt20 * c3 * s3;
			const double c2s4    = sqrt15 * c2 * s4;
			const double cs5     = sqrt6  * c  * s5;
			const double c4s216  = c4 * (1 - 6 * s2);
			const double c216s4  = (6 * c2 - 1) * s4;
			const double c2s2s2  = sqrt30 * c2 * (2 * c2 - 1) * s2;
			const double c2c4s2  = (1 - 10 * c2 + 15 * c4) * s2;
			const double c3c223s = sqrt10 * c3 * (3 * c2 - 2) * s;
			const double cs223s3 = sqrt10 * c * (2 - 3 * s2) * s3;
			const double c2c2c4  = c2 * (6 - 20 * c2 + 15 * c4);
			const double cc2c4s  = sqrt12 * c * (1 - 5 * c2 + 5 * c4) * s;
			const double c2c4c6  = -1 + 12 * c2 - 30 * c4 + 20 * c6;

			w[ 0] =  c6;       w[ 1] =  c5s;      w[ 2] =  c4s2;     w[ 3] =  c3s3;     w[ 4] =  c2s4;     w[ 5] =  cs5;      w[ 6] =  s6;
			w[ 7] = -c5s;      w[ 8] =  c4s216;   w[ 9] =  c3c223s;  w[10] =  c2s2s2;   w[11] =  cs223s3;  w[12] =  c216s4;   w[13] =  cs5;
			w[14] =  c4s2;     w[15] = -c3c223s;  w[16] =  c2c2c4;   w[17] =  cc2c4s;   w[18] =  c2c4s2;   w[19] =  cs223s3;  w[20] =  c2s4;
			w[21] = -c3s3;     w[22] =  c2s2s2;   w[23] = -cc2c4s;   w[24] =  c2c4c6;   w[25] =  cc2c4s;   w[26] =  c2s2s2;   w[27] =  c3s3;
			w[28] =  c2s4;     w[29] = -cs223s3;  w[30] =  c2c4s2;   w[31] = -cc2c4s;   w[32] =  c2c2c4;   w[33] =  c3c223s;  w[34] =  c4s2;
			w[35] = -cs5;      w[36] =  c216s4;   w[37] = -cs223s3;  w[38] =  c2s2s2;   w[39] = -c3c223s;  w[40] =  c4s216;   w[41] =  c5s;
			w[42] =  s6;       w[43] = -cs5;      w[44] =  c2s4;     w[45] = -c3s3;     w[46] =  c4s2;     w[47] = -c5s;      w[48] =  c6;

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
