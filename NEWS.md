# 0.4.2.9000
* Add few examples for `install_pytorch()` function
* Note at CRAN: _"Result: NOTE.  Namespace in Imports field not imported from: ‘methods’. All declared Imports should be used. "_. Remove `methods` from `Imports` in DESCRIPTION.


# 0.4.2
* Release to CRAN after fix.

# 0.4.1.9000
* Returned from CRAN with note: shouldn't link using `[]()` or local link. Must use full URL if want to link to `cran-comments.md`.

# 0.4.1
* to CRAN after fixes.

# 0.4.0.9006
* Try with `Solaris: pandoc (>= 2.0), qpdf ( >= 7.0);`. Getting now two notes and a warrning.
* add `Solaris: pkgutil -y -i qpdf, pkgutil -y -i pandoc"` to SysReqs.
* use `skip_on_cran()` in *test_r_torch_share_objects.R*, *test_types.R*, and *test-install_rtorch_dryrun.R*. Causing errors in Fedora. It doesn't want to install `numpy` but now errors went away in Fedora and Solaris because is nos tested on `numpy`.
* remove line importing `numpy` at the top of test file
* Instead use `qpdf, pandoc (>= 2.7.2) on Solaris"`. 
* add `numpy ( >= 1.14.0)` for Fedora. 
* add Solaris `solaris-x86-patched` platform to rhub.
* Use a different _SystemRequirements_ in DESCRIPTION: `SystemRequirements: "conda (python=3.6 pytorch torchvision cpuonly matplotlib  pandas -c pytorch), Python (>=3.6), pytorch (>=1.6), torchvision, numpy"`. It makes no difference in Fedora.
* test with `fedora-clang-devel` in _rhub_. Still throwing error `ModuleNotFoundNo module named 'numpy'`. failed. It seems tha Fedora cannot install `numpy` as painless as in Debian or Ubuntu.
* start with one of the tests in `test_types.R`. Put quotes at the beginning of the function parentheses in Python code.
* received message from **CRAN** after version 0.4.0 been accepted. Errors in Fedora and Solaris. Will use `rhub` to debug.

# 0.4.0.9005
* add unit tests for new functions
* add functions `sign`, `abs`, `sqrt`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan` in `generics.R`. Documented.

# 0.4.0.9004
* add file `tests/testthat/rhub-tests.R` that sends rTorch for testing on three different platforms. Use it as well in addition to Travis and Appveyor. Closer to CRAN tests.
* removing `data.table` and `R6` from `Imports`. Not used yet.
* add more live tests. They work but `torch_config` does not update output after issuing a `install_pytorch()` command; they return the previous installed PyTorch info. The purpose of these test was initially see if the installation performed as
initially planned. It works outside unit tests. But we cannot enable this test for CRAN because it will take longer time and may not work due to the PyTorch installation process. The major problem I found with these tests is that the `torch_config` objects do not update after issuing a new `install_pytorch`.
* Links to tutorials added to README.
* Remove `reticulate.R` from rTorch code. Some functions were previously customized to accept `channel`. Now the `reticulate` package accepts `channel` as a function parameter.

# 0.4.0.9003
* Now CRAN WinBuilder tests are passing.
* Add conda to `SystemRequirements` because CRAN is not passing.

# 0.4.0.9002
* add `skip_if_no_python()` so CRAN doesn't throw error in unit test `test-install_rtorch_dryrun.R`
* replace function name `utils.R` by `helper_utils.R`. We also have `utils.R` under the R folder.

# 0.4.0.9001
* split test `test-install_rtorch_dryrun.R` in two files. The second one `test-install_rtorch_parse_version.R` will only perform the parsing of what is being sent to `install_pytorch()`.
* use `skip_if_no_python()` in case there is no way Python is installed at the testing point.
* using package `rhub` for testing before releasing to __CRAN__.

# 0.4.0.9000
* use `skip_if_no_torch()` in tests in cases where PyTorch cannot be installed in CRAN.
* add function `skip_if_no_python()`.
* move live torch example to a separate file.
* change `\dontest` to `\dontrun` in examples.
* create branch `0.4.0-fix_examples_problem_in_cran`.

# rTorch 0.4.0
* Modify `tests/testthat/utils.R` to include `skip_on_cran()`
* change version numbering so it is easier to renumber when back to CRAN for fixes.

# rTorch 0.0.4.9000
* Returned from CRAN because of errors. Mainly due to lack of Python and PyTorch installation.

# rTorch 0.0.4
* Release to CRAN
* Updated version of rTorch to adapt to new `PyTorch` versions 1.4, 1.5, 1.6.
* This rTorch release has been tested against Travis Linux, macOs, and Appveyor for Windows. All tests passed successfully. Furthermore, the package has been tested under `Python` 3.6, 3.7 and 3.8. A testing matrix was implemented in `Travis` and `Appveyor` to test version combinations of Python, PyTorch and R. The R versions tested were `R-3.4.3`, `R-3.5.3`, `R-3.6.3`, and `R-4.0.2`.

# rTorch 0.0.3.9013
* Fixed `travis.yml` by bringing `- PYTHON_V="3.7" PYTORCH_V="1.6"` near `env: metrix`. Maybe some space or alignment was preventing ennvronment variables being passed to Travis containers.
* Travis test passing with Python 3.8 in Linux and macOS. Environment variables are not being passed.
* add function `is_rtorch_env_name()` and ` env_name` object to ` torch_config()` to live unit test `install_pytorch()`
* add function `install_pytorch()` to vignette
* add parameter `python_version` to function `conda_install()`
* add backticks to roxygen text since now we are using Markdown in `Roxygen: list(markdown = TRUE)`
* new pkgdown section for Installation. Add two logical functions
* use markdown in roxygen help text
* Travis test PyTorch 1.5 in R-4.0.2 for Linux and macOS with variable shortened.
* Appveyor test **PyTorch 1.6** in R-4.0.2 for Windows. 
* New Appveyor test with **PyTorch 1.5** in R-4.0.2 for Windows. Failing. Definitely PyTorch 1.5 failing in most of the tests.
* Shorten the variable names `PYTORCH_VERSION` and `PYTHON_VERSION` in Travis.
* Appveyor test **PyTorch 1.6** in R-4.0.2 for Windows. Passed.
* Appveyor test PyTorch 1.5 in R-4.0.2 for Windows. Failed. 
* Travis test PyTorch 1.5 in R-4.0.2 for Linux and macOS
* change logical `and`, `or` and `not` to be boolean or uint8 as their inputs.
* do the same for `equal` and `not equal`.
* add a parameter to force to return boolean values instead of `uint8` types. Currently, AND ("!") and OR ("|") return booleans while `NOT` and others don't; they return `uint8`. We should fix this lack of consistency.
* testing on macOS in Travis.
* add condition when PyTorch is 1.1 or lower to compare against `uint8`. Newer PyTorch versions make the conversion of the comparison and return boolean values. In 1.1 they return `uint8`.
* modify functions `torch$eq()`  and `torch$ne()` to validate boolean inputs.
* modify tests for `eq()` and `ne()` in PyTorch 1.1. They return `tensor(True, dtype=torch.bool)` or `tensor(False, dtype=torch.bool)`. 
* add more examples in `generics.R` and `properties.R`


# rTorch 0.0.3.9012
* Finding a problem when using PyTorch 1.1 in logical operations.
* logical generic functions should return `uint8` types as original PyTorch functions in `generics.R`.
* new unit tests for `torch$all`, `torch$any` and some generic logicals in `test-tensor_comparison.R`.
* switch to a couple of Travis and Appveyor tests to save time.
* modify generic `!.torch.Tensor` to return boolean if input is boolean, otherwise return opriginal type. Fix tests in `test_generics.R` and `test_numpy_logical.R`.
* tests for 4 PyTorch versions in `R-4.0.2` and `Python 3.7`,


# rTorch 0.0.3.9011
* Because PyTorch 1.1 and 1.2 are failing on Python 3.8, we could install a custom pytorch with `install_pytorch(conda_python_version = "3.8", version = "1.2")`. Tests failed. But not because of PyTorch but conflict during the conda installation.
* Other custom pytorch with `install_pytorch(conda_python_version = "3.8", version = "1.4")` with tests passed.
* add `numpy` version to printout of `rtorch_config()`.
* perform rebase to get rid off wrong settings for matrix jobs in travis. Keep only those settings that worked.
* maybe a good idea to remove tests with `Python 3.8` because they fail with all PyTorch versions.
* add environment variable `PYTHON_VERSION` to conda in build_script of __appveyor__. Twelve (12) passed.
* Duplicate matrix for __Travis__ tests. Now we have tests for `Python 3.6` and `Python 3.7`, and `3.8` for only `PyTorch 1.6`, a total of 36 tests. All passed.
* Duplicate matrix for __Appveyor__ tests. Now we have tests for `Python 3.6` and `Python 3.7`, a total of 36 tests. All passed.
* Testing `develop` branch with Travis and Appveyor. All tests passed.



# rTorch 0.0.3.9010
* 20200913. branch `0.0.3.9010-fix-appveyor`
* modify __appveyor__ script. currently failing
* remove suffix `-cpu` from `pytorch` and `torchvision` packages from `appveyor.yml`. still failing because of python version is *3.6.1*.
* change python version to *3.6* in  `appveyor.yml`. __Passed__. 
* **appveyor** still failing with error  *package 'remotes' was installed before R 4.0.0: please re-install it*. repo `f0nzie/r-appveyor` requires some changes.
* updating file `appveyor-tool.ps1 ` in `r-appveyor` repo. changes related to Rtools4.
* Windows tests with appveyor have been so far with `pytorch=1.1.0`. Will change to `pytorch=1.4`.
* Testing Linux with `python=3.7` and `pytorch=1.4`. All R versions **passed**.
* clean up `DESCRIPTION`. remove ctb. will credit them in README.
* Testing Linux with `python=3.7` and `pytorch=1.6`. All R versions **passed**.
* Windows tests with `pytorch=1.4` and R-4.0.2. Passed.
* Windows tests (matrix) with `pytorch=1.2, 1.4, 1.6` and R-4.0.2. __Passed__.
* Windows tests (matrix) with two version of `R`: 4.0.2 and 3.6.3 over `pytorch`, 1.1, 1.2, 1.4, and 1.6. using appveyor variable `R_VERSION`.
* Linux tests at `python=3.8` and `pytorch=1.1` __failing__ for all `R` versions. Rest of tests __passed__.
* Windows tests (matrix) with two version of `R`: `4.0.2`, `3.6.3` and `3.5.3` over `pytorch`, 1.1, 1.2, 1.4, and 1.6. using appveyor variable `R_VERSION`.




# rTorch 0.0.3.9009
* 20200911
* add new vignette for PyTorch installation details
* clean up and imporve README
* tested on PyTorch 1.4 on macOS and Linux. All passed
* tested on PyTorch 1.6 on Linux. Passed.
* tested on PyTorch 1.2 on Linux. Passed.
* tested on PyTorch 1.2 on macOS. Passed but R-3.4.3.


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
