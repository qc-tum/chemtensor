#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "hamiltonian.h"
#include "dmrg.h"
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


static PyObject* PyMPS_to_statevector(PyMPSObject* self, PyObject* Py_UNUSED(args))
{
	if (self->mps.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPS has not been initialized yet");
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
	if (self->mps.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPS has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mps.d);
}


static PyObject* PyMPS_qsite(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPS has not been initialized yet");
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
	if (self->mps.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPS has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mps.nsites);
}


static PyObject* PyMPS_bond_dims(PyMPSObject* self, void* Py_UNUSED(closure))
{
	if (self->mps.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPS has not been initialized yet");
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


static PyObject* PyMPO_to_matrix(PyMPOObject* self, PyObject* Py_UNUSED(args))
{
	if (self->mpo.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPO has not been initialized yet");
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
	if (self->mpo.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPO has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mpo.d);
}


static PyObject* PyMPO_qsite(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPO has not been initialized yet");
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
	if (self->mpo.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPO has not been initialized yet");
		return NULL;
	}

	return PyLong_FromLong(self->mpo.nsites);
}


static PyObject* PyMPO_bond_dims(PyMPOObject* self, void* Py_UNUSED(closure))
{
	if (self->mpo.a == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "MPO has not been initialized yet");
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


static PyMethodDef methods[] = {
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
		.ml_name  = "dmrg",
		.ml_meth  = (PyCFunction)Py_dmrg,
		.ml_flags = METH_VARARGS | METH_KEYWORDS,
		.ml_doc   = "Run the two-site DMRG algorithm for the Hamiltonian provided as MPO.\nSyntax: dmrg(mpo, num_sweeps=5, maxiter_lanczos=20, tol_split=1e-10, max_vdim=256, qnum_sector=0, rng_seed=42)",
	},
	{
		0  // sentinel
	},
};


static struct PyModuleDef module = {
	.m_base     = PyModuleDef_HEAD_INIT,
	.m_name     = "chemtensor",  // name of module
	.m_doc      = "chemtensor module for tensor network methods",  // module documentation, may be NULL
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
