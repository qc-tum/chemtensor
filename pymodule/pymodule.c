#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "hamiltonian.h"
#include "dmrg.h"
#include "gradient.h"
#include "aligned_memory.h"


static inline enum NPY_TYPES numeric_to_numpy_type(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			return NPY_FLOAT;
		}
		case DOUBLE_REAL:
		{
			return NPY_DOUBLE;
		}
		case SINGLE_COMPLEX:
		{
			return NPY_CFLOAT;
		}
		case DOUBLE_COMPLEX:
		{
			return NPY_CDOUBLE;
		}
		default:
		{
			// unknown data type
			assert(false);
			return NPY_VOID;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Python MPS object.
///
typedef struct
{
	PyObject_HEAD
	struct mps mps;
}
PyMPSObject;


static PyObject* PyMPS_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds))
{
	PyMPSObject* self = (PyMPSObject*)type->tp_alloc(type, 0);
	if (self != NULL) {
		memset(&self->mps, 0, sizeof(self->mps));
	}
	return (PyObject*)self;
}


static void PyMPS_dealloc(PyMPSObject* self)
{
	if (self->mps.a != NULL) {
		// assuming that the MPS has been initialized
		delete_mps(&self->mps);
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* PyMPS_bond_quantum_numbers(PyMPSObject* self, PyObject* args)
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	// bond index
	int i;
	if (!PyArg_ParseTuple(args, "i", &i)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: bond_quantum_numbers(i: int)");
		return NULL;
	}

	if (i < 0 || i >= self->mps.nsites + 1) {
		char msg[1024];
		sprintf(msg, "invalid bond index i = %i, must be in the range 0 <= i < nsites + 1 with nsites = %i", i, self->mps.nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	const long bond_dim = mps_bond_dim(&self->mps, i);
	const qnumber* qnums = (i < self->mps.nsites) ? self->mps.a[i].qnums_logical[0] : self->mps.a[self->mps.nsites - 1].qnums_logical[2];

	PyObject* list = PyList_New(bond_dim);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (long j = 0; j < bond_dim; j++) {
		if (PyList_SetItem(list, j, PyLong_FromLong(qnums[j])) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static PyObject* PyMPS_to_statevector(PyMPSObject* self, PyObject* Py_UNUSED(args))
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	// return vector representation of MPS as dense NumPy array
	struct block_sparse_tensor vsparse;
	mps_to_statevector(&self->mps, &vsparse);
	// convert to dense tensor
	struct dense_tensor v;
	block_sparse_to_dense_tensor(&vsparse, &v);
	delete_block_sparse_tensor(&vsparse);

	// dummy virtual bond dimensions are retained
	assert(v.ndim == 3);
	assert(v.dim[0] == 1 && v.dim[2] == 1);

	npy_intp dims[1] = { v.dim[1] };
	PyArrayObject* py_v = (PyArrayObject*)PyArray_SimpleNew(1, dims, numeric_to_numpy_type(v.dtype));
	if (py_v == NULL) {
		delete_dense_tensor(&v);
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy vector");
		return NULL;
	}
	memcpy(PyArray_DATA(py_v), v.data, dense_tensor_num_elements(&v) * sizeof_numeric_type(v.dtype));

	delete_dense_tensor(&v);

	return (PyObject*)py_v;
}


static PyMethodDef PyMPS_methods[] = {
	{
		.ml_name  = "bond_quantum_numbers",
		.ml_meth  = (PyCFunction)PyMPS_bond_quantum_numbers,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Return the quantum numbers of the i-th virtual bond."
	},
	{
		.ml_name  = "to_statevector",
		.ml_meth  = (PyCFunction)PyMPS_to_statevector,
		.ml_flags = METH_NOARGS,
		.ml_doc   = "Construct the vector representation of the matrix product state on the full Hilbert space."
	},
	{
		0  // sentinel
	},
};


static PyObject* PyMPS_d(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mps.d);
}


static PyObject* PyMPS_qsite(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	PyObject* list = PyList_New(self->mps.d);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (long i = 0; i < self->mps.d; i++) {
		if (PyList_SetItem(list, i, PyLong_FromLong(self->mps.qsite[i])) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static PyObject* PyMPS_nsites(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mps.nsites);
}


static PyObject* PyMPS_bond_dims(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS has not been initialized yet");
		return NULL;
	}

	PyObject* list = PyList_New(self->mps.nsites + 1);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (int i = 0; i < self->mps.nsites + 1; i++) {
		if (PyList_SetItem(list, i, PyLong_FromLong(mps_bond_dim(&self->mps, i))) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static struct PyGetSetDef PyMPS_getset[] = {
	{
		.name    = "d",
		.get     = (getter)PyMPS_d,
		.set     = NULL,
		.doc     = "local physical dimension of each site",
		.closure = NULL,
	},
	{
		.name    = "qsite",
		.get     = (getter)PyMPS_qsite,
		.set     = NULL,
		.doc     = "physical quantum numbers at each site",
		.closure = NULL,
	},
	{
		.name    = "nsites",
		.get     = (getter)PyMPS_nsites,
		.set     = NULL,
		.doc     = "number of sites",
		.closure = NULL,
	},
	{
		.name    = "bond_dims",
		.get     = (getter)PyMPS_bond_dims,
		.set     = NULL,
		.doc     = "virtual bond dimensions",
		.closure = NULL,
	},
	{
		0  // sentinel
	},
};


static PyTypeObject PyMPSType = {
	.ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "chemtensor.MPS",
	.tp_doc       = PyDoc_STR("MPS object"),
	.tp_basicsize = sizeof(PyMPSObject),
	.tp_itemsize  = 0,
	.tp_flags     = Py_TPFLAGS_DEFAULT,
	.tp_new       = PyMPS_new,
	.tp_init      = NULL,
	.tp_dealloc   = (destructor)PyMPS_dealloc,
	.tp_methods   = PyMPS_methods,
	.tp_getset    = PyMPS_getset,
};


//________________________________________________________________________________________________________________________
///
/// \brief Python MPO object.
///
typedef struct
{
	PyObject_HEAD
	// storing both an MPO assembly and the corresponding MPO
	struct mpo_assembly assembly;
	struct mpo mpo;
}
PyMPOObject;


static PyObject* PyMPO_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds))
{
	PyMPOObject* self = (PyMPOObject*)type->tp_alloc(type, 0);
	if (self != NULL) {
		memset(&self->assembly, 0, sizeof(self->assembly));
		memset(&self->mpo,      0, sizeof(self->mpo));
	}
	return (PyObject*)self;
}


static void PyMPO_dealloc(PyMPOObject* self)
{
	if (self->mpo.a != NULL) {
		// assuming that the MPO has been initialized
		delete_mpo(&self->mpo);
	}
	if (self->assembly.d != 0) {
		// assuming that the MPO assembly has been initialized
		delete_mpo_assembly(&self->assembly);
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject* PyMPO_bond_quantum_numbers(PyMPOObject* self, PyObject* args)
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	// bond index
	int i;
	if (!PyArg_ParseTuple(args, "i", &i)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: bond_quantum_numbers(i: int)");
		return NULL;
	}

	if (i < 0 || i >= self->mpo.nsites + 1) {
		char msg[1024];
		sprintf(msg, "invalid bond index i = %i, must be in the range 0 <= i < nsites + 1 with nsites = %i", i, self->mpo.nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	const long bond_dim = mpo_bond_dim(&self->mpo, i);
	const qnumber* qnums = (i < self->mpo.nsites) ? self->mpo.a[i].qnums_logical[0] : self->mpo.a[self->mpo.nsites - 1].qnums_logical[3];

	PyObject* list = PyList_New(bond_dim);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (long j = 0; j < bond_dim; j++) {
		if (PyList_SetItem(list, j, PyLong_FromLong(qnums[j])) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static PyObject* PyMPO_to_matrix(PyMPOObject* self, PyObject* Py_UNUSED(args))
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	// return matrix representation of MPO as dense NumPy array (not using sparsity structure for simplicity)
	struct block_sparse_tensor msparse;
	mpo_to_matrix(&self->mpo, &msparse);
	// convert to dense tensor
	struct dense_tensor mat;
	block_sparse_to_dense_tensor(&msparse, &mat);
	delete_block_sparse_tensor(&msparse);

	// dummy virtual bond dimensions are retained
	assert(mat.ndim == 4);
	assert(mat.dim[0] == 1 && mat.dim[3] == 1);

	npy_intp dims[2] = { mat.dim[1], mat.dim[2] };
	PyArrayObject* py_mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, numeric_to_numpy_type(mat.dtype));
	if (py_mat == NULL) {
		delete_dense_tensor(&mat);
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy matrix");
		return NULL;
	}
	memcpy(PyArray_DATA(py_mat), mat.data, dense_tensor_num_elements(&mat) * sizeof_numeric_type(mat.dtype));

	delete_dense_tensor(&mat);

	return (PyObject*)py_mat;
}


static PyMethodDef PyMPO_methods[] = {
	{
		.ml_name  = "bond_quantum_numbers",
		.ml_meth  = (PyCFunction)PyMPO_bond_quantum_numbers,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Return the quantum numbers of the i-th virtual bond."
	},
	{
		.ml_name  = "to_matrix",
		.ml_meth  = (PyCFunction)PyMPO_to_matrix,
		.ml_flags = METH_NOARGS,
		.ml_doc   = "Construct the (dense) matrix representation of the matrix product operator on the full Hilbert space."
	},
	{
		0  // sentinel
	},
};


static PyObject* PyMPO_d(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mpo.d);
}


static PyObject* PyMPO_qsite(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	PyObject* list = PyList_New(self->mpo.d);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (long i = 0; i < self->mpo.d; i++) {
		if (PyList_SetItem(list, i, PyLong_FromLong(self->mpo.qsite[i])) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static PyObject* PyMPO_nsites(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mpo.nsites);
}


static PyObject* PyMPO_bond_dims(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}

	PyObject* list = PyList_New(self->mpo.nsites + 1);
	if (list == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create list");
		return NULL;
	}
	for (int i = 0; i < self->mpo.nsites + 1; i++) {
		if (PyList_SetItem(list, i, PyLong_FromLong(mpo_bond_dim(&self->mpo, i))) < 0) {
			Py_DECREF(list);
			PyErr_SetString(PyExc_RuntimeError, "set list item");
			return NULL;
		}
	}

	return list;
}


static PyObject* PyMPO_coeffmap(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->assembly.d == 0) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return NULL;
	}
	if ((self->assembly.num_coeffs < 2) || (self->assembly.coeffmap == NULL)) {
		PyErr_SetString(PyExc_RuntimeError, "PyMPO object is internally inconsistent");
		return NULL;
	}

	npy_intp dims[1] = { self->assembly.num_coeffs };
	PyArrayObject* py_coeffmap = (PyArrayObject*)PyArray_SimpleNew(1, dims, numeric_to_numpy_type(self->assembly.dtype));
	if (py_coeffmap == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy vector");
		return NULL;
	}
	memcpy(PyArray_DATA(py_coeffmap), self->assembly.coeffmap, self->assembly.num_coeffs * sizeof_numeric_type(self->assembly.dtype));

	return (PyObject*)py_coeffmap;
}


static int PyMPO_set_coeffmap(PyMPOObject* self, PyObject* arg, void* Py_UNUSED(closure))
{
	if (self->assembly.d == 0) {
		PyErr_SetString(PyExc_ValueError, "MPO has not been initialized yet");
		return -1;
	}
	if ((self->assembly.num_coeffs < 2) || (self->assembly.coeffmap == NULL)) {
		PyErr_SetString(PyExc_RuntimeError, "PyMPO object is internally inconsistent");
		return -1;
	}

	// convert input argument to NumPy array
	PyArrayObject* py_coeffmap = (PyArrayObject*)PyArray_ContiguousFromObject(arg, numeric_to_numpy_type(self->assembly.dtype), 1, 1);
	if (py_coeffmap == NULL) {
		PyErr_SetString(PyExc_ValueError, "converting input argument to a NumPy array with appropriate data type failed");
		return -1;
	}
	if (PyArray_NDIM(py_coeffmap) != 1) {
		PyErr_SetString(PyExc_ValueError, "expecting a one-dimensional NumPy array");
		Py_DECREF(py_coeffmap);
		return -1;
	}
	if (PyArray_DIM(py_coeffmap, 0) != self->assembly.num_coeffs) {
		char msg[1024];
		sprintf(msg, "number of coefficients cannot change: expecting %i coefficients, received %li", self->assembly.num_coeffs, PyArray_DIM(py_coeffmap, 0));
		PyErr_SetString(PyExc_ValueError, msg);
		Py_DECREF(py_coeffmap);
		return -1;
	}

	// first two entries must always be 0 and 1
	switch (self->assembly.dtype)
	{
		case SINGLE_REAL:
		{
			const float* data = (float*)PyArray_DATA(py_coeffmap);
			if ((data[0] != 0) || (data[1] != 1)) {
				char msg[1024];
				sprintf(msg, "first two coefficients must always be 0 and 1, received %g and %g", data[0], data[1]);
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(py_coeffmap);
				return -1;
			}
			break;
		}
		case DOUBLE_REAL:
		{
			const double* data = (double*)PyArray_DATA(py_coeffmap);
			if ((data[0] != 0) || (data[1] != 1)) {
				char msg[1024];
				sprintf(msg, "first two coefficients must always be 0 and 1, received %g and %g", data[0], data[1]);
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(py_coeffmap);
				return -1;
			}
			break;
		}
		case SINGLE_COMPLEX:
		{
			const scomplex* data = (scomplex*)PyArray_DATA(py_coeffmap);
			if ((data[0] != 0) || (data[1] != 1)) {
				char msg[1024];
				sprintf(msg, "first two coefficients must always be 0 and 1, received %g%+gi and %g%+gi", crealf(data[0]), cimagf(data[0]), crealf(data[1]), cimagf(data[1]));
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(py_coeffmap);
				return -1;
			}
			break;
		}
		case DOUBLE_COMPLEX:
		{
			const dcomplex* data = (dcomplex*)PyArray_DATA(py_coeffmap);
			if ((data[0] != 0) || (data[1] != 1)) {
				char msg[1024];
				sprintf(msg, "first two coefficients must always be 0 and 1, received %g%+gi and %g%+gi", creal(data[0]), cimag(data[0]), creal(data[1]), cimag(data[1]));
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(py_coeffmap);
				return -1;
			}
			break;
		}
		default:
		{
			// unknown data type
			PyErr_SetString(PyExc_RuntimeError, "PyMPO object has an unknown internal data type");
			return -1;
		}
	}

	// actually copy the new coefficients
	memcpy(self->assembly.coeffmap, PyArray_DATA(py_coeffmap), self->assembly.num_coeffs * sizeof_numeric_type(self->assembly.dtype));
	// regenerate the MPO
	mpo_from_assembly(&self->assembly, &self->mpo);

	Py_DECREF(py_coeffmap);

	return 0;
}


static struct PyGetSetDef PyMPO_getset[] = {
	{
		.name    = "d",
		.get     = (getter)PyMPO_d,
		.set     = NULL,
		.doc     = "local physical dimension of each site",
		.closure = NULL,
	},
	{
		.name    = "qsite",
		.get     = (getter)PyMPO_qsite,
		.set     = NULL,
		.doc     = "physical quantum numbers at each site",
		.closure = NULL,
	},
	{
		.name    = "nsites",
		.get     = (getter)PyMPO_nsites,
		.set     = NULL,
		.doc     = "number of sites",
		.closure = NULL,
	},
	{
		.name    = "bond_dims",
		.get     = (getter)PyMPO_bond_dims,
		.set     = NULL,
		.doc     = "virtual bond dimensions",
		.closure = NULL,
	},
	{
		.name    = "coeffmap",
		.get     = (getter)PyMPO_coeffmap,
		.set     = (setter)PyMPO_set_coeffmap,
		.doc     = "internal coefficient map of the MPO",
		.closure = NULL,
	},
	{
		0  // sentinel
	},
};


static PyTypeObject PyMPOType = {
	.ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "chemtensor.MPO",
	.tp_doc       = PyDoc_STR("MPO object"),
	.tp_basicsize = sizeof(PyMPOObject),
	.tp_itemsize  = 0,
	.tp_flags     = Py_TPFLAGS_DEFAULT,
	.tp_new       = PyMPO_new,
	.tp_init      = NULL,
	.tp_dealloc   = (destructor)PyMPO_dealloc,
	.tp_methods   = PyMPO_methods,
	.tp_getset    = PyMPO_getset,
};


//________________________________________________________________________________________________________________________
//


static PyObject* Py_construct_random_mps(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs)
{
	// data type
	const char* dtype_string;
	// number of lattice sites
	int nsites;
	// physical quantum numbers at each site
	PyObject* py_obj_qsite;
	// quantum number sector
	qnumber qnum_sector;
	// maximum virtual bond dimension
	long max_vdim = 256;
	// random number generator seed (for filling tensor entries)
	uint64_t rng_seed = 42;
	// whether to normalize the state
	int normalize = 1;

	// parse input arguments
	char* kwlist[] = { "", "", "", "", "max_vdim", "rng_seed", "normalize", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOi|llp", kwlist,
			&dtype_string,
			&nsites,
			&py_obj_qsite,
			&qnum_sector,
			&max_vdim,
			&rng_seed,
			&normalize)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_random_mps(dtype, nsites, qsite, qnum_sector, max_vdim=256, rng_seed=42, normalize=True)");
		return NULL;
	}

	// data type
	enum numeric_type dtype;
	if ((strcmp(dtype_string, "float32") == 0)
	 || (strcmp(dtype_string, "float") == 0)) {
		dtype = SINGLE_REAL;
	}
	else if ((strcmp(dtype_string, "float64") == 0)
	      || (strcmp(dtype_string, "double") == 0)) {
		dtype = DOUBLE_REAL;
	}
	else if ((strcmp(dtype_string, "complex64") == 0) ||
	         (strcmp(dtype_string, "float complex") == 0)) {
		dtype = SINGLE_COMPLEX;
	}
	else if ((strcmp(dtype_string, "complex128") == 0)
	      || (strcmp(dtype_string, "double complex") == 0)) {
		dtype = DOUBLE_COMPLEX;
	}
	else {
		PyErr_SetString(PyExc_ValueError, "unrecognized 'dtype' argument; use \"float32\", \"float64\", \"complex64\" or \"complex128\"");
		return NULL;
	}

	if (nsites <= 0) {
		char msg[1024];
		sprintf(msg, "'nsites' must be a positive integer, received %i", nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (max_vdim <= 0) {
		char msg[1024];
		sprintf(msg, "'max_vdim' must be a positive integer, received %li", max_vdim);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	// convert 'py_obj_qsite' to NumPy array
	PyArrayObject* py_qsite = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_qsite, NPY_INT, 1, 1);
	if (py_qsite == NULL) {
		PyErr_SetString(PyExc_ValueError, "converting 'qsite' input argument to a NumPy array with integer entries failed");
		return NULL;
	}
	if (PyArray_NDIM(py_qsite) != 1) {
		PyErr_SetString(PyExc_ValueError, "expecting a one-dimensional NumPy array");
		Py_DECREF(py_qsite);
		return NULL;
	}
	const long d = PyArray_DIM(py_qsite, 0);
	if (d == 0) {
		PyErr_SetString(PyExc_ValueError, "'qsite' cannot be an empty list");
		Py_DECREF(py_qsite);
		return NULL;
	}
	const qnumber* qsite = PyArray_DATA(py_qsite);

	struct rng_state rng_state;
	seed_rng_state(rng_seed, &rng_state);

	PyMPSObject* py_psi = (PyMPSObject*)PyMPS_new(&PyMPSType, NULL, NULL);
	if (py_psi == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPS object");
		Py_DECREF(py_qsite);
		return NULL;
	}
	// actually construct the random MPS
	construct_random_mps(dtype, nsites, d, qsite, qnum_sector, max_vdim, &rng_state, &py_psi->mps);

	if (normalize)
	{
		// perform a left and right normalization sweep to avoid superfluous bonds
		double nrm = mps_orthonormalize_qr(&py_psi->mps, MPS_ORTHONORMAL_LEFT);
		if (nrm == 0) {
			// initial norm zero indicates that quantum numbers are likely incompatible
			PyErr_SetString(PyExc_RuntimeError, "cannot normalize the MPS for the provided quantum numbers");
			Py_DECREF(py_qsite);
			return NULL;
		}
		mps_orthonormalize_qr(&py_psi->mps, MPS_ORTHONORMAL_RIGHT);
	}

	Py_DECREF(py_qsite);

	return (PyObject*)py_psi;
}


static PyObject* Py_construct_ising_1d_mpo(PyObject* Py_UNUSED(self), PyObject* args)
{
	// number of lattice sites
	int nsites;
	// Hamiltonian parameters
	double J, h, g;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "iddd", &nsites, &J, &h, &g)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_ising_1d_mpo(nsites: int, J: float, h: float, g: float)");
		return NULL;
	}
	if (nsites <= 0) {
		char msg[1024];
		sprintf(msg, "'nsites' must be a positive integer, received %i", nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	PyMPOObject* py_mpo = (PyMPOObject*)PyMPO_new(&PyMPOType, NULL, NULL);
	if (py_mpo == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPO object");
		return NULL;
	}

	// actually construct the assembly and MPO
	construct_ising_1d_mpo_assembly(nsites, J, h, g, &py_mpo->assembly);
	mpo_from_assembly(&py_mpo->assembly, &py_mpo->mpo);

	return (PyObject*)py_mpo;
}


static PyObject* Py_construct_heisenberg_xxz_1d_mpo(PyObject* Py_UNUSED(self), PyObject* args)
{
	// number of lattice sites
	int nsites;
	// Hamiltonian parameters
	double J, D, h;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "iddd", &nsites, &J, &D, &h)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_heisenberg_xxz_1d_mpo(nsites: int, J: float, D: float, h: float)");
		return NULL;
	}
	if (nsites <= 0) {
		char msg[1024];
		sprintf(msg, "'nsites' must be a positive integer, received %i", nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	PyMPOObject* py_mpo = (PyMPOObject*)PyMPO_new(&PyMPOType, NULL, NULL);
	if (py_mpo == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPO object");
		return NULL;
	}

	// actually construct the assembly and MPO
	construct_heisenberg_xxz_1d_mpo_assembly(nsites, J, D, h, &py_mpo->assembly);
	mpo_from_assembly(&py_mpo->assembly, &py_mpo->mpo);

	return (PyObject*)py_mpo;
}


static PyObject* Py_construct_bose_hubbard_1d_mpo(PyObject* Py_UNUSED(self), PyObject* args)
{
	// number of lattice sites
	int nsites;
	// physical dimension per site (maximal occupancy is d - 1)
	long d;
	// Hamiltonian parameters
	double t, u, mu;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "ilddd", &nsites, &d, &t, &u, &mu)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_bose_hubbard_1d_mpo(nsites: int, d: int, t: float, u: float, mu: float)");
		return NULL;
	}
	if (nsites <= 0) {
		char msg[1024];
		sprintf(msg, "'nsites' must be a positive integer, received %i", nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	PyMPOObject* py_mpo = (PyMPOObject*)PyMPO_new(&PyMPOType, NULL, NULL);
	if (py_mpo == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPO object");
		return NULL;
	}

	// actually construct the assembly and MPO
	construct_bose_hubbard_1d_mpo_assembly(nsites, d, t, u, mu, &py_mpo->assembly);
	mpo_from_assembly(&py_mpo->assembly, &py_mpo->mpo);

	return (PyObject*)py_mpo;
}


static PyObject* Py_construct_fermi_hubbard_1d_mpo(PyObject* Py_UNUSED(self), PyObject* args)
{
	// number of lattice sites
	int nsites;
	// Hamiltonian parameters
	double t, u, mu;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "iddd", &nsites, &t, &u, &mu)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_fermi_hubbard_1d_mpo(nsites: int, t: float, u: float, mu: float)");
		return NULL;
	}
	if (nsites <= 0) {
		char msg[1024];
		sprintf(msg, "'nsites' must be a positive integer, received %i", nsites);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	PyMPOObject* py_mpo = (PyMPOObject*)PyMPO_new(&PyMPOType, NULL, NULL);
	if (py_mpo == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPO object");
		return NULL;
	}

	// actually construct the assembly and MPO
	construct_fermi_hubbard_1d_mpo_assembly(nsites, t, u, mu, &py_mpo->assembly);
	mpo_from_assembly(&py_mpo->assembly, &py_mpo->mpo);

	return (PyObject*)py_mpo;
}


static PyObject* Py_fermi_hubbard_encode_quantum_numbers(PyObject* Py_UNUSED(self), PyObject* args)
{
	// particle number quantum number
	qnumber q_pnum;
	// spin quantum number
	qnumber q_spin;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "ii", &q_pnum, &q_spin)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: fermi_hubbard_encode_quantum_numbers(q_pnum: int, q_spin: int)");
		return NULL;
	}

	// encode particle and spin quantum numbers
	qnumber q = fermi_hubbard_encode_quantum_numbers(q_pnum, q_spin);

	return PyLong_FromLong(q);
}


static PyObject* Py_fermi_hubbard_decode_quantum_numbers(PyObject* Py_UNUSED(self), PyObject* args)
{
	qnumber qnum;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "i", &qnum)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: fermi_hubbard_decode_quantum_numbers(qnum: int)");
		return NULL;
	}

	// decode quantum numbers
	qnumber q_pnum, q_spin;
	fermi_hubbard_decode_quantum_numbers(qnum, &q_pnum, &q_spin);

	return PyTuple_Pack(2, PyLong_FromLong(q_pnum), PyLong_FromLong(q_spin));
}


static PyObject* Py_construct_molecular_hamiltonian_mpo(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs)
{
	PyObject* py_obj_tkin;
	PyObject* py_obj_vint;
	int optimize = 0;

	// parse input arguments
	char* kwlist[] = { "", "", "optimize", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &py_obj_tkin, &py_obj_vint, &optimize)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: construct_molecular_hamiltonian_mpo(tkin, vint, optimize=False)");
		return NULL;
	}

	PyArrayObject* py_tkin = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_tkin, NPY_DOUBLE, 2, 2);
	if (py_tkin == NULL) {
		PyErr_SetString(PyExc_ValueError, "converting input argument 'tkin' to a NumPy array with degree 2 failed");
		return NULL;
	}
	PyArrayObject* py_vint = (PyArrayObject*)PyArray_ContiguousFromObject(py_obj_vint, NPY_DOUBLE, 4, 4);
	if (py_vint == NULL) {
		PyErr_SetString(PyExc_ValueError, "converting input argument 'vint' to a NumPy array with degree 4 failed");
		return NULL;
	}

	// argument checks
	if (PyArray_NDIM(py_tkin) != 2) {
		char msg[1024];
		sprintf(msg, "'tkin' must have degree 2, received %i", PyArray_NDIM(py_tkin));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(py_tkin, 0) != PyArray_DIM(py_tkin, 1)) {
		char msg[1024];
		sprintf(msg, "'tkin' must be a square matrix, received a %li x %li matrix", PyArray_DIM(py_tkin, 0), PyArray_DIM(py_tkin, 1));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(py_tkin, 0) == 0) {
		PyErr_SetString(PyExc_ValueError, "'tkin' cannot be an empty matrix");
		return NULL;
	}
	if (PyArray_TYPE(py_tkin) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "'tkin' must have 'double' format entries");
		return NULL;
	}
	if (!(PyArray_FLAGS(py_tkin) & NPY_ARRAY_C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_SyntaxError, "'tkin' does not have contiguous C storage format");
		return NULL;
	}
	if (PyArray_NDIM(py_vint) != 4) {
		char msg[1024];
		sprintf(msg, "'vint' must have degree 4, received %i", PyArray_NDIM(py_vint));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if ((PyArray_DIM(py_vint, 0) != PyArray_DIM(py_vint, 1)) ||
	    (PyArray_DIM(py_vint, 0) != PyArray_DIM(py_vint, 2)) ||
	    (PyArray_DIM(py_vint, 0) != PyArray_DIM(py_vint, 3))) {
		char msg[1024];
		sprintf(msg, "all dimensions of 'vint' must agree, received a %li x %li x %li x %li tensor", PyArray_DIM(py_vint, 0), PyArray_DIM(py_vint, 1), PyArray_DIM(py_vint, 2), PyArray_DIM(py_vint, 3));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_TYPE(py_vint) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "'vint' must have 'double' format entries");
		return NULL;
	}
	if (!(PyArray_FLAGS(py_vint) & NPY_ARRAY_C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_SyntaxError, "'vint' does not have contiguous C storage format");
		return NULL;
	}
	if (PyArray_DIM(py_tkin, 0) != PyArray_DIM(py_vint, 0)) {
		PyErr_SetString(PyExc_SyntaxError, "'tkin' and 'vint' must have the same axis dimensions");
		return NULL;
	}

	const long nsites = (long)PyArray_DIM(py_tkin, 0);
	long dim_tkin[2] = { nsites, nsites };
	struct dense_tensor tkin = {
		.data  = PyArray_DATA(py_tkin),
		.dim   = dim_tkin,
		.dtype = DOUBLE_REAL,
		.ndim  = 2,
	};
	long dim_vint[4] = { nsites, nsites, nsites, nsites };
	struct dense_tensor vint = {
		.data  = PyArray_DATA(py_vint),
		.dim   = dim_vint,
		.dtype = DOUBLE_REAL,
		.ndim  = 4,
	};

	PyMPOObject* py_mpo = (PyMPOObject*)PyMPO_new(&PyMPOType, NULL, NULL);
	if (py_mpo == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPO object");
		return NULL;
	}

	// actually construct the assembly and MPO
	construct_molecular_hamiltonian_mpo_assembly(&tkin, &vint, (bool)optimize, &py_mpo->assembly);
	mpo_from_assembly(&py_mpo->assembly, &py_mpo->mpo);

	Py_DECREF(py_tkin);
	Py_DECREF(py_vint);

	return (PyObject*)py_mpo;
}


//________________________________________________________________________________________________________________________
///
/// \brief Two-site DMRG algorithm.
///
static PyObject* Py_dmrg(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs)
{
	// MPO representing the Hamiltonian
	PyMPOObject* py_mpo;
	// number of DMRG sweeps
	int num_sweeps = 5;
	// maximum number of Lanczos iterations
	int maxiter_lanczos = 20;
	// SVD splitting tolerance
	double tol_split = 1e-10;
	// maximum virtual bond dimension of MPS
	long max_vdim = 256;
	// quantum number sector of the to-be optimized MPS
	qnumber qnum_sector = 0;
	// random number generator seed (for filling tensor entries of initial MPS)
	uint64_t rng_seed = 42;

	// parse input arguments
	char* kwlist[] = { "", "num_sweeps", "maxiter_lanczos", "tol_split", "max_vdim", "qnum_sector", "rng_seed", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iidlil", kwlist,
			&py_mpo,
			&num_sweeps,
			&maxiter_lanczos,
			&tol_split,
			&max_vdim,
			&qnum_sector,
			&rng_seed)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: dmrg(mpo, num_sweeps=5, maxiter_lanczos=20, tol_split=1e-10, max_vdim=256, qnum_sector=0, rng_seed=42)");
		return NULL;
	}
	if (py_mpo->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO object has not been initialized yet");
		return NULL;
	}
	if (num_sweeps <= 0) {
		char msg[1024];
		sprintf(msg, "'num_sweeps' must be a positive integer, received %i", num_sweeps);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (maxiter_lanczos <= 0) {
		char msg[1024];
		sprintf(msg, "'maxiter_lanczos' must be a positive integer, received %i", maxiter_lanczos);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (tol_split < 0 || tol_split > 1) {
		char msg[1024];
		sprintf(msg, "'tol_split' must be in the interval [0, 1], received %g", tol_split);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (max_vdim <= 0) {
		char msg[1024];
		sprintf(msg, "'max_vdim' must be a positive integer, received %li", max_vdim);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	// number of lattice sites
	const int nsites = py_mpo->mpo.nsites;
	assert(nsites > 0);

	PyMPSObject* py_psi = (PyMPSObject*)PyMPS_new(&PyMPSType, NULL, NULL);
	if (py_psi == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating PyMPS object");
		return NULL;
	}
	// initial state vector as MPS
	struct rng_state rng_state;
	seed_rng_state(rng_seed, &rng_state);
	construct_random_mps(py_mpo->mpo.a[0].dtype, py_mpo->mpo.nsites, py_mpo->mpo.d, py_mpo->mpo.qsite, qnum_sector, max_vdim, &rng_state, &py_psi->mps);
	double nrm = mps_norm(&py_psi->mps);
	if (nrm == 0) {
		PyMPS_dealloc(py_psi);
		PyErr_SetString(PyExc_RuntimeError, "initial random MPS has norm zero (possibly due to mismatching quantum numbers)");
		return NULL;
	}

	// run two-site DMRG
	double* en_sweeps = aligned_alloc(MEM_DATA_ALIGN, num_sweeps * sizeof(double));
	double* entropy   = aligned_alloc(MEM_DATA_ALIGN, (nsites - 1) * sizeof(double));
	int ret = dmrg_twosite(&py_mpo->mpo, num_sweeps, maxiter_lanczos, tol_split, max_vdim, &py_psi->mps, en_sweeps, entropy);
	if (ret < 0) {
		aligned_free(entropy);
		aligned_free(en_sweeps);
		PyMPS_dealloc(py_psi);
		PyErr_SetString(PyExc_RuntimeError, "two-site DMRG failed internally (likely since Krylov subspace method did not converge)");
		return NULL;
	}

	// store 'en_sweeps' in a Python list
	PyObject* py_en_sweeps = PyList_New(num_sweeps);
	if (py_en_sweeps == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create 'en_sweeps' return value list");
		return NULL;
	}
	for (int i = 0; i < num_sweeps; i++) {
		if (PyList_SetItem(py_en_sweeps, i, PyFloat_FromDouble(en_sweeps[i])) < 0) {
			Py_DECREF(py_en_sweeps);
			PyErr_SetString(PyExc_RuntimeError, "set 'en_sweeps' list item");
			return NULL;
		}
	}

	// store 'entropy' in a Python list
	PyObject* py_entropy = PyList_New(nsites - 1);
	if (py_entropy == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "create 'entropy' return value list");
		return NULL;
	}
	for (int i = 0; i < nsites - 1; i++) {
		if (PyList_SetItem(py_entropy, i, PyFloat_FromDouble(entropy[i])) < 0) {
			Py_DECREF(py_entropy);
			PyErr_SetString(PyExc_RuntimeError, "set 'entropy' list item");
			return NULL;
		}
	}

	// clean up
	aligned_free(entropy);
	aligned_free(en_sweeps);

	return PyTuple_Pack(3, py_psi, py_en_sweeps, py_entropy);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the value and gradient of `<chi | op | psi>` with respect to the internal MPO coefficients.
///
static PyObject* Py_operator_average_coefficient_gradient(PyObject* Py_UNUSED(self), PyObject* args)
{
	// MPO representing the operator
	PyMPOObject* py_op;
	PyMPSObject* py_psi;
	PyMPSObject* py_chi;

	// parse input arguments
	if (!PyArg_ParseTuple(args, "OOO", &py_op, &py_psi, &py_chi)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: operator_average_coefficient_gradient(op: MPO, psi: MPS: chi: MPS)");
		return NULL;
	}

	if (py_op->mpo.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPO 'op' has not been initialized yet");
		return NULL;
	}
	if ((py_op->assembly.d != py_op->mpo.d) || (py_op->assembly.graph.nsites != py_op->mpo.nsites)) {
		PyErr_SetString(PyExc_RuntimeError, "PyMPO object is internally inconsistent");
		return NULL;
	}
	if (py_psi->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS 'psi' has not been initialized yet");
		return NULL;
	}
	if (py_chi->mps.a == NULL) {
		PyErr_SetString(PyExc_ValueError, "MPS 'chi' has not been initialized yet");
		return NULL;
	}
	// compatibility checks
	if ((py_op->assembly.d != py_psi->mps.d) ||
	    (py_op->assembly.d != py_chi->mps.d)) {
		PyErr_SetString(PyExc_ValueError, "local Hilbert space dimensions of 'op', 'psi' and 'chi' do not agree");
		return NULL;
	}
	if (!qnumber_all_equal(py_op->assembly.d, py_op->assembly.qsite, py_psi->mps.qsite) ||
	    !qnumber_all_equal(py_op->assembly.d, py_op->assembly.qsite, py_chi->mps.qsite)) {
		PyErr_SetString(PyExc_ValueError, "local Hilbert space quantum numbers of 'op', 'psi' and 'chi' do not agree");
		return NULL;
	}
	if ((py_op->assembly.dtype != py_psi->mps.a[0].dtype) ||
	    (py_op->assembly.dtype != py_chi->mps.a[0].dtype)) {
		PyErr_SetString(PyExc_ValueError, "numeric data types of 'op', 'psi' and 'chi' do not agree");
		return NULL;
	}
	if ((py_op->assembly.graph.nsites != py_psi->mps.nsites) ||
	    (py_op->assembly.graph.nsites != py_chi->mps.nsites)) {
		PyErr_SetString(PyExc_ValueError, "number of lattice sites of 'op', 'psi' and 'chi' do not agree");
		return NULL;
	}

	void* avr = aligned_alloc(MEM_DATA_ALIGN, sizeof_numeric_type(py_op->assembly.dtype));
	void* dcoeff = aligned_alloc(MEM_DATA_ALIGN, py_op->assembly.num_coeffs * sizeof_numeric_type(py_op->assembly.dtype));
	operator_average_coefficient_gradient(&py_op->assembly, &py_psi->mps, &py_chi->mps, avr, dcoeff);

	PyArrayObject* py_avr = (PyArrayObject*)PyArray_SimpleNew(0, NULL, numeric_to_numpy_type(py_op->assembly.dtype));
	if (py_avr == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy scalar");
		return NULL;
	}
	memcpy(PyArray_DATA(py_avr), avr, sizeof_numeric_type(py_op->assembly.dtype));

	npy_intp dims[1] = { py_op->assembly.num_coeffs };
	PyArrayObject* py_dcoeff = (PyArrayObject*)PyArray_SimpleNew(1, dims, numeric_to_numpy_type(py_op->assembly.dtype));
	if (py_dcoeff == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating NumPy vector");
		return NULL;
	}
	memcpy(PyArray_DATA(py_dcoeff), dcoeff, py_op->assembly.num_coeffs * sizeof_numeric_type(py_op->assembly.dtype));


	aligned_free(dcoeff);
	aligned_free(avr);


	return PyTuple_Pack(2, py_avr, py_dcoeff);
}


static PyMethodDef methods[] = {
	{
		.ml_name  = "construct_random_mps",
		.ml_meth  = (PyCFunction)Py_construct_random_mps,
		.ml_flags = METH_VARARGS | METH_KEYWORDS,
		.ml_doc   = "Construct a matrix product state with random normal tensor entries, given an overall quantum number sector and maximum virtual bond dimension.\nSyntax: construct_random_mps(dtype, nsites, qsite, qnum_sector, max_vdim=256, rng_seed=42, normalize=True)",
	},
	{
		.ml_name  = "construct_ising_1d_mpo",
		.ml_meth  = Py_construct_ising_1d_mpo,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Contruct an MPO representation of the Ising Hamiltonian 'sum J Z Z + h Z + g X' on a one-dimensional lattice.\nSyntax: construct_ising_1d_mpo(nsites: int, J: float, h: float, g: float)",
	},
	{
		.ml_name  = "construct_heisenberg_xxz_1d_mpo",
		.ml_meth  = Py_construct_heisenberg_xxz_1d_mpo,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Construct an MPO representation of the XXZ Heisenberg Hamiltonian 'sum J (X X + Y Y + D Z Z) - h Z' on a one-dimensional lattice.\nSyntax: construct_heisenberg_xxz_1d_mpo(nsites: int, J: float, D: float, h: float)",
	},
	{
		.ml_name  = "construct_bose_hubbard_1d_mpo",
		.ml_meth  = Py_construct_bose_hubbard_1d_mpo,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Construct an MPO representation of the Bose-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.\nSyntax: construct_bose_hubbard_1d_mpo(nsites: int, d: int, t: float, u: float, mu: float)",
	},
	{
		.ml_name  = "construct_fermi_hubbard_1d_mpo",
		.ml_meth  = Py_construct_fermi_hubbard_1d_mpo,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Construct an MPO representation of the Fermi-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.\nSyntax: construct_fermi_hubbard_1d_mpo(nsites: int, t: float, u: float, mu: float)",
	},
	{
		.ml_name  = "fermi_hubbard_encode_quantum_numbers",
		.ml_meth  = Py_fermi_hubbard_encode_quantum_numbers,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Encode a particle number and spin quantum number for the Fermi-Hubbard Hamiltonian into a single quantum number.\nSyntax: fermi_hubbard_encode_quantum_numbers(q_pnum: int, q_spin: int)",
	},
	{
		.ml_name  = "fermi_hubbard_decode_quantum_numbers",
		.ml_meth  = Py_fermi_hubbard_decode_quantum_numbers,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Decode a quantum number of the Fermi-Hubbard Hamiltonian into separate particle number and spin quantum numbers.\nSyntax: fermi_hubbard_decode_quantum_numbers(qnum: int)",
	},
	{
		.ml_name  = "construct_molecular_hamiltonian_mpo",
		.ml_meth  = (PyCFunction)Py_construct_molecular_hamiltonian_mpo,
		.ml_flags = METH_VARARGS | METH_KEYWORDS,
		.ml_doc   = "Construct a molecular Hamiltonian as MPO, using physicists' convention for the interaction term:\nH = sum_{i,j} t_{i,j} ad_i a_j + 1/2 sum_{i,j,k,l} v_{i,j,k,l} ad_i ad_j a_l a_k\nSyntax: construct_molecular_hamiltonian_mpo(tkin, vint, optimize=False)",
	},
	{
		.ml_name  = "dmrg",
		.ml_meth  = (PyCFunction)Py_dmrg,
		.ml_flags = METH_VARARGS | METH_KEYWORDS,
		.ml_doc   = "Run the two-site DMRG algorithm for the Hamiltonian provided as MPO.\nSyntax: dmrg(mpo, num_sweeps=5, maxiter_lanczos=20, tol_split=1e-10, max_vdim=256, qnum_sector=0, rng_seed=42)",
	},
	{
		.ml_name  = "operator_average_coefficient_gradient",
		.ml_meth  = Py_operator_average_coefficient_gradient,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Compute the value and gradient of `<chi | op | psi>` with respect to the internal MPO coefficients.\nSyntax: operator_average_coefficient_gradient(op: MPO, psi: MPS: chi: MPS)",
	},
	{
		0  // sentinel
	},
};


static struct PyModuleDef module = {
	.m_base     = PyModuleDef_HEAD_INIT,
	.m_name     = "chemtensor",  // name of module
	.m_doc      = "chemtensor module for tensor network methods applied to chemical systems",  // module documentation, may be NULL
	.m_size     = -1,            // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
	.m_methods  = methods,       // module methods
	.m_slots    = NULL,          // slot definitions for multi-phase initialization
	.m_traverse = NULL,          // traversal function to call during GC traversal of the module object, or NULL if not needed
	.m_clear    = NULL,          // a clear function to call during GC clearing of the module object, or NULL if not needed
	.m_free     = NULL,          // a function to call during deallocation of the module object, or NULL if not needed
};


PyMODINIT_FUNC PyInit_chemtensor(void)
{
	// import NumPy array module (required)
	import_array();

	if (PyType_Ready(&PyMPSType) < 0) {
		return NULL;
	}
	if (PyType_Ready(&PyMPOType) < 0) {
		return NULL;
	}

	PyObject* m = PyModule_Create(&module);
	if (m == NULL) {
		return NULL;
	}

	// register MPS type
	Py_INCREF(&PyMPSType);
	if (PyModule_AddObject(m, "MPS", (PyObject*)&PyMPSType) < 0) {
		Py_DECREF(&PyMPSType);
		Py_DECREF(m);
		return NULL;
	}

	// register MPO type
	Py_INCREF(&PyMPOType);
	if (PyModule_AddObject(m, "MPO", (PyObject*)&PyMPOType) < 0) {
		Py_DECREF(&PyMPOType);
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
