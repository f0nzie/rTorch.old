#' Torch for R
#' @import methods
#' @import R6
#' @import reticulate
#' @importFrom reticulate import dict iterate import_from_path array_reshape
#' import_builtins iter_next np_array py_run_file
#' py_iterator py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr
#' py_to_r r_to_py tuple
#' @importFrom graphics par plot points
#' @docType package
#' @name rTorch
NULL



# #' #' @keywords internal
# #' "_PACKAGE"


# ' #' Torch for R
# ' #'
# ' #' @import reticulate
# ' #' @docType package
# ' #' @name rTorch
# ' NULL

.globals <- new.env(parent = emptyenv())
.globals$torch <- NULL


.onLoad <- function(libname, pkgname) {
    torch <<- import("torch", delay_load = list(
        priority = 5,
        environment = "r-torch"

        # on_load = function() {
        #     python_path <- system.file("python", package = "rTorch")
        #     tools <- import_from_path("torchtools", path = python_path)
        # }

    ))
}


