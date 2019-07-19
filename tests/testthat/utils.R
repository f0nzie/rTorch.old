skip_if_no_torch <- function() {
  if (!reticulate::py_module_available("torch"))
    skip("Torch not available for testing")
}
