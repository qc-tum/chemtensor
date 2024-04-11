#include <math.h>
#include "clebsch_gordan.h"


char* test_clebsch_gordan_coefficients()
{
	if (fabs(clebsch_gordan(1, 3, 2, 1, 0, 0) - (-sqrt(3)/2)) > 1e-13) {
		return "Clebsch-Gordan coefficient does not agree with reference value";
	}

	if (fabs(clebsch_gordan(3, 4, 5, 2, 0, 1) - (-4/sqrt(35))) > 1e-13) {
		return "Clebsch-Gordan coefficient does not agree with reference value";
	}

	// 'j3' out of range
	if (fabs(clebsch_gordan(3, 1, 5, 2, 1, 2) - 0) > 1e-13) {
		return "Clebsch-Gordan coefficient does not agree with reference value";
	}

	// parity of 'j3' does not match
	if (fabs(clebsch_gordan(4, 2, 3, 2, 0, 1) - 0) > 1e-13) {
		return "Clebsch-Gordan coefficient does not agree with reference value";
	}

	// 'm' quantum numbers do not sum to zero
	if (fabs(clebsch_gordan(2, 3, 1, 0, 3, 0) - 0) > 1e-13) {
		return "Clebsch-Gordan coefficient does not agree with reference value";
	}

	return 0;
}
