library(testthat)

source("tensor_functions.R")

context("R and Python share variables")


test_that("Python and R share variables", {
  expr <- py_run_string("
x = 10
y = 20
# we don't want to print
# print(x)
# print(y)
")
  # print(expr$x)
  expect_equal(py$x, 10)
  expect_equal(py$y, 20)

  # TODO: how do we get the R values in Python
  # In notebooks is easy but how do we do it in R scripts?

  A <- 100.123
  B <- 200.2

  Ap <- r_to_py(A)
  Bp <- r_to_py(B)
  # py_run_string("print(r.A)") # error:  error: object 'A' not found.

  # py_run_string("print(r['A'])") # untimeEvaluation error: object 'A' not found.
  # A <- r_to_py(A)
  # py_run_string("print(A)")
  # py_run_string("import numpy as np")
  # py_run_string("my_python_array = np.array([2,4,6,8])")
  # py_run_string("for item in my_python_array: print(item)")
  # py_eval("print(dir(r))")
  obj <- py_run_string("r.__dir__()")
  # print(class(obj))
  # print(obj$keys())

})
