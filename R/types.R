
#' Tensor shape
#'
#' @param ... Tensor dimensions
#'
#' @export
shape <- function(...) {
    values <- list(...)
    lapply(values, function(value) {
        if (!is.null(value))
            as.integer(value)
        else
            NULL
    })
}


tensor_shape_to_list <- function(x) {
    # convert the shape of a torch tensor to a list
    # a tensor shape has the form "torch.Size([2, 5])"
    l <- import_builtins()$list(x)
    if (inherits(l, "python.builtin.object")) {
        l <- reticulate::py_to_r(l)
    }
    l <- as.list(l)
}


#' @export
`[.torch.Size` <- function(x, i) {
    stopifnot(i>0)
    l <- tensor_shape_to_list(x)
    l[i]
}

#' @export
`[[.torch.Size` <- function(x, i) {
    stopifnot(i>0)
    l <- tensor_shape_to_list(x)
    l[[i]]
}
