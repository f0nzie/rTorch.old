skip_if_no_torch <- function() {
  if (!reticulate::py_module_available("torch"))
    skip("Torch not available for testing")
}

expect_all_true <- function(obj, ...) {
    testthat::expect_true(all(object = obj), ...)
}
