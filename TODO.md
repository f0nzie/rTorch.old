# TODO

## rTorch 0.0.4.x
* add Linear Algebra functions: _Gaussian Elimination_, Cholesky, LU
* add examples for Gaussian Elimination
* add unit tests for linear algebra functions
* add examples for matrix operations.
* find stable libraries for fast operations in matrices.

## rTorch 0.0.3.x
### TODO
* Update appveyor for unit tests in Windows
* Logical operations should return booleans
* Matrix for parallel Travis testing
* Add new function to `pkgdown`

### DONE
* modify `pytorch_install` to include version. DONE
* Modify `install_pytorch()` and remove `pytorch-cpu==1.1` way to specify `pytorch`. It is causing error because new PyTorch versions do not take the suffix `-cpu`. DONE
* add to `install_pytorch()` parameters for Python version, packages, and PyTorch version. DONE
* Generate documentation with `pkgdown`. DONE

## half-done
* move some functions in `tests/testthat/utils.R` to `R/utils.R`



## rTorch 0.0.3.9009
* add vignette for PyTorch installation
