ChemTensor
==========

Tensor network algorithms for chemical systems.


Building
--------
The code requires the BLAS, LAPACKE and HDF5 development libraries. These can be installed via `sudo apt install libblas-dev liblapacke-dev libhdf5-dev` (on Ubuntu Linux) or similar.

From the project directory, use `cmake` to build the project:
```
mkdir build && cd build
cmake ../
cmake --build .
````

Currently, this will compile the unit tests, which you can run via `./chemtensor_test`.


References
----------
- U. Schollwöck  
  The density-matrix renormalization group in the age of matrix product states  
  [Ann. Phys. 326, 96-192 (2011)](https://doi.org/10.1016/j.aop.2010.09.012) ([arXiv:1008.3477](https://arxiv.org/abs/1008.3477))
- J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete  
  Unifying time evolution and optimization with matrix product states  
  [Phys. Rev. B 94, 165116 (2016)](https://doi.org/10.1103/PhysRevB.94.165116) ([arXiv:1408.5056](https://arxiv.org/abs/1408.5056))
- C. Krumnow, L. Veis, Ö. Legeza, J. Eisert  
  Fermionic orbital optimization in tensor network states  
  [Phys. Rev. Lett. 117, 210402 (2016)](https://doi.org/10.1103/PhysRevLett.117.210402) ([arXiv:1504.00042](https://arxiv.org/abs/1504.00042))
- G. K.-L. Chan, A. Keselman, N. Nakatani, Z. Li, S. R. White  
  Matrix product operators, matrix product states, and ab initio density matrix renormalization group algorithms  
  [J. Chem. Phys. 145, 014102 (2016)](https://doi.org/10.1063/1.4955108) ([arXiv:1605.02611](https://arxiv.org/abs/1605.02611))
- J. Ren, W. Li, T. Jiang, Z. Shuai  
  A general automatic method for optimal construction of matrix product operators using bipartite graph theory  
  [J. Chem. Phys. 153, 084118 (2020)](https://doi.org/10.1063/5.0018149) ([arXiv:2006.02056](https://arxiv.org/abs/2006.02056))
