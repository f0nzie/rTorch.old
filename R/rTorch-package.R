#' @keywords internal
"_PACKAGE"


#' Torch for R
#'
#' @import reticulate
#' @docType package
#' @name torch
NULL

.globals <- new.env(parent = emptyenv())
.globals$torch <- NULL


.onLoad <- function(libname, pkgname) {
    torch <<- import("torch", delay_load = list(
        priority = 5,
        environment = "r-torch"

    ))
}
