#' Torch for R
#' @import methods
#' @import R6
#' @importFrom reticulate import dict iterate import_from_path array_reshape np_array py_run_file py_run_string py_iterator py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr py_to_r r_to_py tuple
#' @import reticulate
#' @importFrom graphics par plot points
#' @docType package
#' @name rTorch
NULL




.globals <- new.env(parent = emptyenv())
.globals$torchboard <- NULL



packageStartupMessage("loading PyTorch")

.onLoad <- function(libname, pkgname) {

    # delay load PyTorch
    torch <<- import("torch", delay_load = list(
        priority = 5,
        environment = "r-torch"       # this is a user generated environment
    ))

    torchvision <<- import("torchvision", delay_load = list(
      priority = 4,                    # decrease priority so we don't get collision with torch
      environment = "r-torchvision"    # this is a user generated environment
    ))

    np <<- import("numpy", delay_load = list(
      priority = 3,                 # decrease priority so we don't get collision with torch
      environment = "r-np"          # this is a user generated environment
    ))

  # provide a common base S3 class for tensors
  reticulate::register_class_filter(function(classes) {
      if (any(c("torch.autograd.variable.Variable",
                  "torch.tensor._TensorBase")
              %in%
              classes)) {
        c("torch.tensor", classes)      # this enables the generics + * - /
      } else {
          classes
      }
  })
}



#' Torch configuration information
#'
#' @return List with information on the current configuration of PyTorch.
#'   You can determine whether PyTorch was found using the `available`
#'   member (other members vary depending on whether `available` is `TRUE`
#'   or `FALSE`)
#'
#' @keywords internal
#' @export
torch_config <- function() {

    # first check if we found Torch
    have_torch <- py_module_available("torch")

    # get py config
    config <- py_config()

    # found it!
    if (have_torch) {

        # get version
        tfv <- strsplit(torch$"__version__", ".", fixed = TRUE)[[1]]
        version <- package_version(paste(tfv[[1]], tfv[[2]], sep = "."))

        structure(class = "pytorch_config", list(
            available = TRUE,
            version = version,
            version_str = torch$"__version__",
            location = config$required_module_path,
            python = config$python,
            python_version = config$version
        ))
        # didn't find it
    } else {
        structure(class = "pytorch_config", list(
            available = FALSE,
            python_verisons = config$python_versions,
            error_message = torch_config_error_message()
        ))
    }
}


#' @rdname torch_config
#' @keywords internal
#' @export
torch_version <- function() {
    config <- torch_config()
    if (config$available)
        as.character(config$version)
    else
        NULL
}


#' @export
print.pytorch_config <- function(x, ...) {
    if (x$available) {
        aliased <- function(path) sub(Sys.getenv("HOME"), "~", path)
        cat("PyTorch v", x$version_str, " (", aliased(x$location), ")\n", sep = "")
        cat("Python v", x$python_version, " (", aliased(x$python), ")\n", sep = "")
    } else {
        cat(x$error_message, "\n")
    }
}



# Build error message for TensorFlow configuration errors
torch_config_error_message <- function() {
    message <- "Installation of Torch not found."
    config <- py_config()
    if (!is.null(config)) {
        if (length(config$python_versions) > 0) {
            message <- paste0(message,
                              "\n\nPython environments searched for 'rTorch' package:\n")
            python_versions <- paste0(" ", normalizePath(config$python_versions, mustWork = FALSE),
                                      collapse = "\n")
            message <- paste0(message, python_versions, sep = "\n")
        }
    }
    message <- paste0(message,
                      "\nYou can install PyTorch using the install_pytorch() function.\n")
    message
}
