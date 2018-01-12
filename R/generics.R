
#' @export
py_str.torch.python.ops.variables.Variable <- function(object, ...) {
    paste0("Variable(shape=", py_str(object$size()), ", ",
           "dtype=", object$dtype$name, ")\n", sep = "")
}


#' @importFrom utils str
#' @export
"print.torch.tensor" <- function(x, ...) {
    if (py_is_null_xptr(x))
        cat("<pointer: 0x0>\n")
    else {
        str(x, ...)
        if (!is.null(torch$get_num_threads())) {
            value <- tryCatch(x$eval(), error = function(e) NULL)
            if (!is.null(value))
                cat(" ", str(value), "\n", sep = "")
        }
    }
}


#' @importFrom utils .DollarNames
#' @export
.DollarNames.torch.python.platform.flags._FlagValues <- function(x, pattern = "") {

    # skip if this is a NULL xptr
    if (py_is_null_xptr(x))
        return(character())

    # get the underlying flags and return the names
    flags <- x$`__flags`
    names(flags)
}


#' @export
"+.torch.tensor" <- function(a, b) {
    if (any(class(a) == "torch._C.FloatTensorBase"))
        torch$add(a, b)
    else
        torch$add(b, a)
}


#' @export
"*.torch.tensor" <- function(a, b) {
    if (any(class(a) == "torch._C.FloatTensorBase"))
        torch$mul(a, b)
    else
        torch$mul(b, a)
}


#' @export
"/.torch.tensor" <- function(a, b) {
    if (any(class(a) == "torch._C.FloatTensorBase"))
        torch$div(a, b)
    else
        torch$div(b, a)
}


#' @export
"==.torch.tensor" <- function(a, b) {
    torch$equal(a, b)
}


#' Matrix/Tensor multiplication
#' PyTorch matmul
#' @param a Tensor 1
#' @param b Tensor 2
#' @export
`%**%` <- function(a, b) {
    torch$matmul(a, b)
}
