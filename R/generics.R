
#' @export
py_str.torch.python.ops.variables.Variable <- function(object, ...) {
    paste0("Variable(shape=", py_str(object$size()), ", ",
           "dtype=", object$dtype$name, ")\n", sep = "")
}


#' @importFrom utils str
#' @export
"print.torch.Tensor" <- function(x, ...) {
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
"dim.torch.Tensor" <- function(x) {        # change .tensor to .Tensor
    if (py_is_null_xptr(x))
        NULL
    else {
        shape <- import_builtins()$list(x$size())  # return a list
        # shape <- x$dim()   # torch has a dim() function
        if (!is.null(shape))
            shape
        else
            NULL
    }
}


#' @export
"length.torch.Tensor" <- function(x) {
    if (py_is_null_xptr(x))
        length(NULL)
    else
        Reduce(`*`, dim(x))
}




#' @export
"+.torch.Tensor" <- function(a, b) {
    if (any(class(a) == "torch.Tensor"))
        torch$add(a, b)
    else
        torch$add(b, a)
}


#' @export
"-.torch.Tensor" <- function(a, b) {
    if (missing(b)) {
        if (py_has_attr(torch, "negative"))
            torch$negative(a)
        else
            torch$neg(a)
    } else {
        if (py_has_attr(torch, "subtract"))
            torch$subtract(a, b)
        else
            torch$sub(a, b)
    }
}


#' @export
"*.torch.Tensor" <- function(a, b) {
    if (py_has_attr(torch, "multiply"))
        torch$multiply(a, b)
    else
        torch$mul(a, b)
}

#' @export
"/.torch.Tensor" <- function(a, b) {
    torch$div(a, b)
}


#' @export
`%.*%` <- function(a, b) {
    torch$dot(a, b)
}


# TODO: finish these two and tensor float ###################
#' @export
"*.torch.Tensor" <- function(a, b) {
    if (any(class(a) == "torch._C.FloatTensorBase"))
        torch$mul(a, b)
    else
        torch$mul(b, a)
}


#' @export
"/.torch.Tensor" <- function(a, b) {
    if (any(class(a) == "torch._C.FloatTensorBase"))
        torch$div(a, b)
    else
        torch$div(b, a)
}

##################################################################


#' @export
"==.torch.Tensor" <- function(a, b) {
    torch$equal(a, b)
}


#' @export
"!=.torch.Tensor" <- function(a, b) {
    # there is not not_equal function in PyTorch
    !torch$equal(a, b)
}



#' @export
"<.torch.Tensor" <- function(a, b) {
    torch$lt(a, b)
}


#' @export
"<=.torch.Tensor" <- function(a, b) {
    torch$le(a, b)
}


#' @export
">.torch.Tensor" <- function(a, b) {
    torch$gt(a, b)
}


#' @export
">=.torch.Tensor" <- function(a, b) {
    torch$ge(a, b)
}




#' Matrix/Tensor multiplication
#' PyTorch matmul
#' @param a Tensor 1
#' @param b Tensor 2
#' @export
`%**%` <- function(a, b) {
    torch$matmul(a, b)
}



#' #' @export
#' "[.torch.tensor" <- function(x, ...) {
#'
#'     call <- match.call()
#'     check_zero_based(call)
#'
#'     one_based_extract <- getOption("torch.one_based_extract", TRUE)
#'
#'     basis <- ifelse(one_based_extract, 1, 0)
#'     call_list <- as.list(call)[-1]
#'     do.call(extract_manual,
#'             c(call_list, basis = basis),
#'             envir = parent.frame())
#' }

#' @export
"^.torch.Tensor" <- function(a, b) {
    torch$pow(a, b)
}

#' @export
"exp.torch.Tensor" <- function(x) {
    torch$exp(x)
}

#' @export
"log.torch.Tensor" <- function(x, base = exp(1L)) {
    if (is_tensor(base) || base != exp(1L)) {
        base <- torch$as_tensor(base, x$dtype)
        torch$log(x) / torch$log(base)
        print("here")
    } else {
        print("not here")
        torch$log(x)
    }


}

#' @export
#' @method log2 torch.Tensor
"log2.torch.Tensor" <- function(x) {
    torch$log2(x)
}

#' @export
#' @method log10 torch.Tensor
"log10.torch.Tensor" <- function(x) {
    torch$log10(x)
}

np <- import("numpy")

tensor_logical_and <- function(x, y) {
    x <- r_to_py(x$numpy())
    y <- r_to_py(y$numpy())
    torch$BoolTensor(np$logical_and(x, y))
}

tensor_logical_or <- function(x, y) {
    x <- r_to_py(x$numpy())
    y <- r_to_py(y$numpy())
    torch$BoolTensor(np$logical_or(x, y))
}


#' @export
"&.torch.Tensor" <- function(a, b) {
    tensor_logical_and(a, b)
}


#' @export
"|.torch.Tensor" <- function(a, b) {
    tensor_logical_or(a, b)
}


