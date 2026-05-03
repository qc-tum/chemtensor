/// \file spherical_harmonics.c
/// \brief Evaluate spherical harmonics.

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "spherical_harmonics.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the real-valued spherical harmonics \f$Y_{lm}\f$ for the given 'l' quantum number
/// at the provided unit vector 'v' (Cartesian coordinates).
/// The 'm' quantum numbers are enumerated as -l, ..., l.
/// 'yv' must point to an array with 2*l + 1 entries and is filled by the function.
///
/// The implementation follows the Wikipedia definition.
///
void real_spherical_harmonics(const qnumber l, const double v[3], double* yv)
{
	assert(l >= 0);
	assert(fabs(norm2(CT_DOUBLE_REAL, 3, v) - 1) < 1e-13);

	switch (l)
	{
		case 0:
		{
			// 1 / sqrt(4 pi)
			yv[0] = 0.28209479177387814347;
			break;
		}
		case 1:
		{
			// sqrt(3 / (4 pi))
			const double fac = 0.48860251190291992159;
			const double x = v[0];
			const double y = v[1];
			const double z = v[2];

			yv[0] = fac * y;
			yv[1] = fac * z;
			yv[2] = fac * x;

			break;
		}
		case 2:
		{
			// sqrt(15 / (4 pi))
			const double fac = 1.0925484305920790705;
			// 1 / (2 sqrt(3))
			const double inv2sq3 = 0.28867513459481288225;
			const double x = v[0];
			const double y = v[1];
			const double z = v[2];

			yv[0] = fac * x * y;
			yv[1] = fac * y * z;
			yv[2] = fac * inv2sq3 * (3 * z*z - 1);
			yv[3] = fac * x * z;
			yv[4] = fac * 0.5 * (x*x - y*y);

			break;
		}
		case 3:
		{
			const double x = v[0];
			const double y = v[1];
			const double z = v[2];
			const double x2 = x*x;
			const double y2 = y*y;
			const double _5z2_1 = 5*z*z - 1;

			yv[0] = 0.59004358992664351035 * y * (3*x2 - y2);   // 1/4 sqrt(35 / (2 pi))
			yv[1] = 2.8906114426405540554  * x * y * z;         // 1/2 sqrt(105 / pi)
			yv[2] = 0.45704579946446573616 * y *  _5z2_1;       // 1/4 sqrt(21 / (2 pi))
			yv[3] = 0.37317633259011539141 * z * (_5z2_1 - 2);  // 1/4 sqrt(7 / pi)
			yv[4] = 0.45704579946446573616 * x *  _5z2_1;       // 1/4 sqrt(21 / (2 pi))
			yv[5] = 1.4453057213202770277  * z * (x2 - y2);     // 1/4 sqrt(105 / pi)
			yv[6] = 0.59004358992664351035 * x * (x2 - 3*y2);   // 1/4 sqrt(35 / (2 pi))

			break;
		}
		default:
		{
			// 'l' not supported yet
			fprintf(stderr, "quantum number l = %i in spherical harmonics evaluation not supported yet\n", l);
			assert(false);
			break;
		}
	}
}
