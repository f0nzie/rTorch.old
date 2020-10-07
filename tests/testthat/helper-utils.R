library(reticulate)

if (reticulate::py_module_available("torch")) {
  torch       <- import("torch")
  torchvision <- import("torchvision")
  nn          <- import("torch.nn")
  transforms  <- import("torchvision.transforms")
  dsets       <- import("torchvision.datasets")
  builtins    <- import_builtins()
  np          <- import("numpy", convert = TRUE, delay_load = FALSE)
  # default setting is converting automatically to R objects

  # https://stackoverflow.com/a/59337065/5270873
  # filter_warnings <- import("warnings.filterwarnings")
  # filter_warnings("ignore")
}



tensor_logical_and <- function(x, y) {
    x <- r_to_py(x$numpy())
    y <- r_to_py(y$numpy())
    torch$BoolTensor(make_copy(np$logical_and(x, y)))
}

tensor_logical_or <- function(x, y) {
    x <- r_to_py(x$numpy())
    y <- r_to_py(y$numpy())
    torch$BoolTensor(make_copy(np$logical_or(x, y)))
}


# make_copy <- function(object, ...) {
#     if (class(object) == "torch.Tensor") {
#         obj <- object$copy_(object)
#     }
#     else if (class(object) == "numpy.ndarray") {
#         obj <- object$copy()
#     } else {
#         obj <- r_to_py(object)$copy()
#     }
#     return(obj)
# }


# as_tensor <- function(...) torch$as_tensor(...)

tensor_dot <- function(A, B) {
    torch$dot(A, B)
}

tensor_dim <- function(tensor) {
    # same as R dim() returning a vector of integers
    builtins$list(tensor$size())
}

tensor_ndim <- function(tensor) {
    # same as torch$dim()
    size <- builtins$list(tensor$size())
    length(size)
}

tensor_sum <- function(tensor) {
    tensor$sum()$item()
}

# is_tensor <- function(object) {
#     class(object) %in% c("torch.Tensor")
#     class_obj <- class(object)
#     all(class_obj[grepl("Tensor", class_obj)] %in%
#             c("torch.Tensor", "torch._C._TensorBase"))
# }

py_object_last <- function(object) {
    if (py_has_length(object)) py_len(object) - 1L
    else stop()
}

py_has_length <- function(object) {
    # ifelse(any(py_list_attributes(object) %in% c("__len__")), TRUE, FALSE)
    tryCatch({
        any(py_list_attributes(object) %in% c("__len__"))
    },
    error = function(e) {
        message("object has no __len__ attribute")
        # print(e)
        return(FALSE)
    }
    )
}



skip_if_no_torch <- function() {
  if (!reticulate::py_module_available("torch"))
    skip("Torch not available for testing")
}


skip_if_no_python <- function() {
  if (!reticulate::py_available())
    skip("Python not available for testing")
}


expect_all_true <- function(obj, ...) {
    testthat::expect_true(all(object = obj), ...)
}

expect_near <- function(..., tol = 1e-5) expect_equal(..., tolerance = tol)

expect_tensor_equal <- function(a, b) {
    # a <- make_copy(a)
    # b <- make_copy(b)
    expect_true(torch$equal(a, b))
}
