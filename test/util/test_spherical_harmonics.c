#define _USE_MATH_DEFINES
#include <math.h>
#include "spherical_harmonics.h"
#include "wigner.h"
#include "cblas_ct.h"
#include "rng.h"
#include "aligned_memory.h"
#include "hdf5_util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the entries of the Euler y-axis rotation matrix.
///
static void euler_y_rotation_matrix(const double theta, double mat[9])
{
	const double c = cos(theta);
	const double s = sin(theta);

	mat[0] =  c;    mat[1] =  0;    mat[2] =  s;
	mat[3] =  0;    mat[4] =  1;    mat[5] =  0;
	mat[6] = -s;    mat[7] =  0;    mat[8] =  c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the entries of the Euler z-axis rotation matrix.
///
static void euler_z_rotation_matrix(const double theta, double mat[9])
{
	const double c = cos(theta);
	const double s = sin(theta);

	mat[0] =  c;    mat[1] = -s;    mat[2] =  0;
	mat[3] =  s;    mat[4] =  c;    mat[5] =  0;
	mat[6] =  0;    mat[7] =  0;    mat[8] =  1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply a square matrix with a vector.
///
static void multiply_square_matrix_vector(const ct_long n, const double* restrict mat, const double* restrict v, double* restrict ret)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1., mat, n, v, 1, 0., ret, 1);
}


char* test_real_spherical_harmonics()
{
	hid_t file = H5Fopen("../test/util/data/test_real_spherical_harmonics.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_real_spherical_harmonics failed";
	}

	double v[3];
	if (read_hdf5_attribute(file, "v", H5T_NATIVE_DOUBLE, v) < 0) {
		return "reading 'v' unit vector from disk failed";
	}

	for (qnumber l = 0; l <= 3; l++)
	{
		double* yv = ct_malloc((2*l + 1) * sizeof(double));
		real_spherical_harmonics(l, v, yv);

		double* yv_ref = ct_malloc((2*l + 1) * sizeof(double));
		char varname[1024];
		sprintf(varname, "yv%i", l);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, yv_ref) < 0) {
			return "reading real-valued spherical harmonics from disk failed";
		}

		if (uniform_distance(CT_DOUBLE_REAL, 2*l + 1, yv, yv_ref) > 1e-14) {
			return "real-valued spherical harmonics do not match reference";
		}

		ct_free(yv_ref);
		ct_free(yv);
	}

	// transformation by Wigner D-matrix must correspond to an
	// Euler rotation of the unit vector that parametrizes the spherical harmonics

	struct rng_state rng_state;
	seed_rng_state(112, &rng_state);

	// random rotation angles
	const double psi   = 2 * M_PI * randu(&rng_state);
	const double theta = 2 * M_PI * randu(&rng_state);
	const double phi   = 2 * M_PI * randu(&rng_state);

	// transform 'v' by z-y-z Euler rotation
	double v_rot[3];
	{
		double rot[9];

		double tmp0[3];
		euler_z_rotation_matrix(phi, rot);
		multiply_square_matrix_vector(3, rot, v, tmp0);

		double tmp1[3];
		euler_y_rotation_matrix(theta, rot);
		multiply_square_matrix_vector(3, rot, tmp0, tmp1);

		euler_z_rotation_matrix(psi, rot);
		multiply_square_matrix_vector(3, rot, tmp1, v_rot);
	}

	for (qnumber l = 0; l <= 2; l++)
	{
		double* yv = ct_malloc((2*l + 1) * sizeof(double));
		real_spherical_harmonics(l, v, yv);

		double* w = ct_malloc((2*l + 1) * (2*l + 1) * sizeof(double));
		real_wigner_d(2*l, psi, theta, phi, w);

		double* yv_w = ct_malloc((2*l + 1) * sizeof(double));
		multiply_square_matrix_vector(2*l + 1, w, yv, yv_w);

		double* yv_rot = ct_malloc((2*l + 1) * sizeof(double));
		real_spherical_harmonics(l, v_rot, yv_rot);

		// compare
		if (uniform_distance(CT_DOUBLE_REAL, 2*l + 1, yv_w, yv_rot) > 1e-14) {
			return "transformation by Wigner D-matrix does not match Euler rotation of the unit vector that parametrizes the spherical harmonics";
		}

		ct_free(yv_rot);
		ct_free(yv_w);
		ct_free(w);
		ct_free(yv);
	}

	H5Fclose(file);

	return 0;
}
