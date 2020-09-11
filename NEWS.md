# rTorch 0.0.3.9008
* 20200911
* add new vignette for PyTorch installation details


# rTorch 0.0.3.9008
* 20200910. Branch `0.0.3.9008-implement-todo-items`
* add test in `test_numpy_logical.R` to check sample tensors
* add test in `test_info.R` add test to check the version three components
* regenerate pkgdown site. add `make_copy` function


# rTorch 0.0.3.9007
* 20200909. Branch `0.0.3.9007-fix-auto-load-torch`
* simplify imports in `package.R`
* provide function for help handler after change in `on_load()`
* fix function `make_copy()` to consider when an object have multiple classes. Use `any` for the logical selection
* test o Travis for macOS and R-4.0.2, R-3.6.3 with pytorch=1.4. PASSED
* test o Travis for Linux Xenial and R-4.0.2, R-3.6.3 with pytorch=1.4. PASSED



# rTorch 0.0.3.9006
* 20200830. Fix `install_pytorch()` and `parse_torch_version()`.
* Modify functions `install_pytorch()` and `parse_torch_version()`
* add new unit tests for `install_pytorch()` and `parse_torch_version()`
* add the `dry_run` option to `install_pytorch()` to use output values in unit tests
* new unit tests file `test-install_commands.R`


# rTorch 0.0.3.9005
* 20200829
* rename function in tests from `tensor_dim_` to `tensor_ndim`
* Dockerfile now using environment variables for name and version of the package insaide the script.
* export function `make_copy()` moved from unit test utilities.
* Update README. Remove mention to `torch$index` (not applicable).
* Add more installation instructions for PyTorch. 
* Clarify some examples in the README. Use `message()` instead of `print()`



# rTorch 0.0.3.9004
* 20200828
* All tests are passing in Travis on R-4.0.0, R-3.6.3, R-3.5.3 and R-3.4.3.
* Tests that are failing are in the `examples`. 
* Error is `RuntimeError: Expected object of scalar type Byte but got scalar type Long ` in `[.torch.Tensor` at generics functions.
* Example causing error is a verification of the tensor `(all(y[all_dims(), 1] == y[,,,,1]) == torch$tensor(1L))$numpy()`. In older versions of R it works. We could change the test to something like `as.logical((all(y[all_dims(), 1] == y[,,,,1]))$numpy()) == TRUE`. Tested in R-3.6.3 locally and PASSED. Will test via Travis.
* All tests in Ubuntu xenial with PyTorch 1.1 using Travis passed.
* Testing R-4.0.0 with PyTorch 1.1 generates 28 warnings `test_torch_core.R:211: warning: narrow the condition has length > 1 and only the first element will be used` but all test passed.
* integrating Docker with rTorch. The Docker container will create an equivalent Travis machine to save time during tests.
* Adding option `- if [ "$TRAVIS_OS_NAME" = "osx" ]; then conda install nomkl;fi` in *.travis.yml* to be able to get rid off an error related to **OMP**
* merging branch `003.9004-fix-examples-torch-byte-to-long` with `develop`.
* will start testing with *PyTorch 1.4* as the average version. Installing PyTorch 1.4 with
```
> rTorch:::install_conda(package="pytorch=1.4", envname="r-torch", conda="auto", conda_python_version = "3.6", pip=FALSE, channel="pytorch", extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas"))
```



# rTorch 0.0.3.9003
* 20200814
* creating branch, make active `fix-readme-add-tests`.
* Using _https://travis-ci.org/_
* combine tensor_functions.R and utils.R
* unit tests for transpose and permute
* Getting this warning during check: `checkRd: (5) rTorch.Rd:0-7: Must have a \description`. Also stops in _travis-ci.org_.
* Switching from PyTorch `1.6` to `1.1` to debug error in _rTorch.Rd_
* Fixed problem with _rTorch.Rd_. Block in package.R needed description. Added this extra line below the title: `#' PyTorch bindings for R`. The problem originated by the new R version.
* Re-install PyTorch 1.6 with `rTorch:::install_conda(package="pytorch=1.6", envname="r-torch", conda="auto", conda_python_version = "3.6", pip=FALSE, channel="pytorch", extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas"))`. Run tests. Run devtools::check(). All passed.
* Add `--run-donttest` option to check() arguments. Getting errors.
* Fix `all_dims()` examples in _generics.R_.
* Fix `logical_not()` examples in _generics.R_.
* Fix ``[.torch.Tensor`` examples in _extract.R_.
* Fix `torch_extract_opts` examples in _extract.R_.
* Travis stopping on error in `dontrun` examples that passed in the local machine. What is different is the PyTorch version specified in `.travis.yml`. Changing variable from "1.1"" to `PYTORCH_VERSION="1.6"`.
* Travis stopping on error related to suffix `pytorch-cpu==1.6` in command `'rTorch::install_pytorch(method="conda", version=Sys.getenv("PYTORCH_VERSION"), channel="pytorch", conda_python_version="3.6")'`. We need to modify function `install_pytorch()`.
* tests to be performed with `R version 4.0.0 (2020-04-24) -- "Arbor Day"`
* first, remove installation of gcc or libstdc++
* remove `rTorch::pytorch_install()`. Use instead `rTorch:::conda_install()`.
* create environment variables for PYTORCH_VERSION, PYTHON_VERSION and LD_LIBRARY_PATH.
* remove symbolic link to `libstdc++.so.6` in the Linux installation. This is confusing Python.
* export `LD_LIBRARY_PATH=${TRAVIS_HOME}/miniconda/lib`. 
* install required packages with `Rscript -e 'install.packages(c("logging", "reticulate", "jsonlite", "R6", "rstudioapi", "data.table"))`
* reduce size of tensor in `test_tensor_dim.R` because throwing error due to lack of memory.
* after careful revision no more errors in Linux. Only one NOTE: `* checking for future file timestamps ... NOTE. unable to verify current time`.
* all tests running fine with R-4.0.0. Will change version to R-3.6.3.
* all tests running fine with R-3.5.3. Multiple R versions through Travis.
* all tests running fine with R-3.4.3. 
* all tests passed in macOS with versions 3.6.3, 3.5.3 and 3.4.3.




# rTorch 0.0.3.9002
* 20200810
* creating branch `fix-elimination-cpu-suffix` to address removal of suffix by developer.
* Installed PyTorch 1.1 with `rTorch:::install_conda(package="pytorch=1.1", envname="r-torch", conda="auto", conda_python_version = "3.6", pip=FALSE, channel="pytorch", extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas"))`
* revise unit tests and fix version dependence. Two test failing since last release PyTorch 1.1. Four tests failing with PyTorch 1.6 installed. Related to versioning checks.
* All tests in README passing and running.
* fixed tests in `test_types.R`. Minor changes in `reticulate` makes it more sensitive.
* set aside check on `mnist` dataset until internal tests are resolved
* install PyTorch 1.6 on Python 3.6`. Restart RStudio.
* fix version test with `VERSIONS <- c("1.1", "1.0", "1.2", "1.3", "1.4", "1.5", "1.6")` in `test_info.R`
* With PyTorch 1.6 we are getting the warning `extract syntaxsys:1: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /opt/conda/conda-bld/pytorch_1595629417679/work/torch/csrc/utils/tensor_numpy.cpp:141.)`. 
* add test custom function `expect_all_true()` in `utils.R` that shortens a test with multiple TRUE returning from condition
* Fix overwriting warning by adding `r_to_py` to R array and then copying with `r_to_py(r_array)$copy()` before converting to tensor
* five test files giving NumPy overwrite warning test-generic-methods.R, test_generics.R, test_numpy_logical.R, test_tensor_slicing.R, test_torch_core.R
* change functions `tensor_logical_and()` and `tensor_logical_or()` in `generics.R` -which use NumPy logical functions - to make a copy before converting the numpy array to a tensor
* change `as_tensor()` function in tensor_functions.R with `torch$as_tensor()`. Use make_copy() to prevent PyTorch warning.



# rTorch 0.0.3.9001
* 20190918
* Rename tensorflow old labels to pytorch

# rTorch 0.0.3.9000
* 20190805
* Now in CRAN. 10:00
* Announce in LinkedIn and Twitter

# rTorch 0.0.3
* 20190802
* Released to CRAN at 15:20


# rTorch 0.0.2.9000
* 20190802
* Returned from CRAN with notes
* Fix single quotes in DESCRIPTION
* Change `\dontrun` by `\donttest` where applicable
* Get rid of a warning message
* Replace print/cat by message/warning
* Add `\value` to all functions with `@return`
* Added `cran-comments.md`


# rTorch 0.0.2
* July 31 2019
* Submitted to CRAN at 15:30. Received. Waiting manual inspection.
* Test with `appveyor.yml`
* Created repository [r-appveyor](https://github.com/f0nzie/r-appveyor) at Oil Gains Analytics GitHub account. `appveyor` scripts now are called from this repo. Original source is at [krlmlr/r-appveyor](https://github.com/krlmlr/r-appveyor)
* Test with `.travis.yml`
* Copy three functions from reticulate to customize it and be able to specify the conda channel. Using `pytorch` channel in `reticulate.R`.
* Specify torch-cpu and torchvision-cpu in `install.R`
* Move out vignettes to reduce testing time. Will ship separately using `rsuite`.

# rTorch 0.0.1.9013
* July 26 2019
* Vignettes temporarily moved to inst/vignettes to reduce build time of package
* Add function remainder for tensors. Equivalent to `a %% b`
* Change unit tests in `test_generics.R` to use new function `expect_true_tensor`
* Enhance functions `any` and `all`. Add examples
* Add roxygen documentation to two tensor operations
* Change download folders for MNIST datasets under project folder


# rTorch 0.0.1.9012
* July 24 2019
* Change MNIST download folder to ~/raw_data instead of inst/
* On vignette `mnist_fashion_inference.Rmd`:
* Add dropout class to reduce overfitting
* Add a training loop for the dropout class
* Added/remove experimental code to replicate the Python function to visualize the image along with the bar plot. Unsuccessful because R (image) and Python image (plt.imshow) functions use different array dimensions.

# rTorch 0.0.1.9011
* July 22 2019
* Added vignette `mnist_fashion_inference.Rmd`.
* Added vignette `simple_linear_regression.Rmd`.
* Add generic ! (logical not) 
* Fix generics any, all using as_tensor() instead of tensor()


# rTorch 0.0.1.9010
* July 22 2019
* New vignette using PyTorch builtin functions and classes. Rainfall dataset: `linear_regression_rainfall_builtins.Rmd`
* Add comments to `linear_regression_rainfall.Rmd`

# rTorch 0.0.1.9009
* July 22 2019
* Fix version numbers. Missing the number one.

# rTorch 0.0.1.9008
* July 22 2019
* Refresh pkgdown
* Export html files for pkgdown. Modify .gitignore.

# rTorch 0.0.1.9006
* July 22 2019
* Add pkgdown website

# rTorch 0.0.1.9005
* July 22 2019
* Add vignette `png_images_minist_digits.Rmd`. It uses PBG images in a local folder instead of downloading MNIST idx format images.
* Add logical operators to README.

# rTorch 0.0.1.9004
* July 22 2019
* Add vignette `idx_images_minist_digits.Rmd`

# rTorch 0.0.1.9003
* July 21 2019
* New vignette `two_layer_neural_network.Rmd`. Had some problem with the tensor types. Fixed by using shorter generic version of the tensor gradient operation.

# rTorch 0.0.1.9002
* July 21 2019
* Add two more vignettes.
* Get rid of a warning on roxygen documentation
* Remove old code from generics.R

# rTorch 0.0.1.9001
* July 21 2019
* Adding first example as a vignette.
* import Python torch with `py_run_string("import torch")`

# rTorch 0.0.1
* July 21 2019
* alpha version
* first release to Github
* package coming after publication of `rpystats-apollo11`
* still examples to be added


# rTorch 0.0.0.9000

* Added a `NEWS.md` file to track changes to the package.
