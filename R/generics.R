# np <- import("numpy")

#' @importFrom utils str
#' @export
"print.torch.Tensor" <- function(x, ...) {
    if (py_is_null_xptr(x))
        message("<pointer: 0x0>\n")
    else {
        str(x, ...)
        if (!is.null(torch$get_num_threads())) {
            value <- tryCatch(x$eval(), error = function(e) NULL)
            if (!is.null(value))
                message(" ", str(value), "\n", sep = "")
        }
    }
}


#' @export
py_str.torch.python.ops.variables.Variable <- function(object, ...) {
    paste0("Variable(shape=", py_str(object$size()), ", ",
           "dtype=", object$dtype$name, ")\n", sep = "")
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


#' One tensor operation
#'
#' @param x tensor
#'
#' @examples
#' \donttest{
#' A <- torch$ones(c(60000L, 1L, 28L, 28L))
#' dim(A)
#' }
one_tensor_op <- function(x) UseMethod("one_tensor_op")


#' @describeIn one_tensor_op Dimensions of a tensor
#' @details Get the dimensions of a tensor displaying it as a vector.
#' @return a vector of integers with the dimensions of the tensor
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

#' @describeIn one_tensor_op Length of a tensor. Eqivalent to torch$numel()
#' @return the number of elements of a tensor as an integer
#' @export
"length.torch.Tensor" <- function(x) {
    if (py_is_null_xptr(x))
        length(NULL)
    else
        Reduce(`*`, dim(x))
}


#' Remainder
#'
#' Computes the element-wise remainder of division.
#' @param a a tensor
#' @param b a scalar or a tensor
#' @export
#' @examples
#' \donttest{
#' x <- torch$Tensor(list(-3., -2, -1, 1, 2, 3))
#' y <- torch$Tensor(list(1., 2, 3, 4, 5))
#' torch$remainder(x, 2)
#' torch$remainder(y, 1.5)
#'
#' x %% 2
#' y %% 1.5
#' }
#' @return the reminder of the division between tensor by a scalar or tensor
"%%.torch.Tensor" <- function(a, b) {
    torch$remainder(a, b)
}


# all <- function(x, ...) UseMethod("all")
# any <- function(x, ...) UseMethod("any")


#' all
#'
#' Returns True if all elements in the tensor are non-zero, False otherwise.
#' @param x tensor
#' @param dim dimension to reduce
#' @param ... other parameters (yet to be developed)
#' @export
#' @examples
#' \donttest{
#' a <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
#' b <- torch$BoolTensor(list(FALSE, TRUE, TRUE, TRUE))
#' c <- torch$BoolTensor(list(TRUE, TRUE, TRUE, FALSE))
#' all(a)
#' all(b)
#' all(c)
#' d <- torch$tensor(list(list(0, 0),
#'                        list(0, 0),
#'                        list(0, 1),
#'                        list(1, 1)), dtype=torch$uint8)
#' all(d)
#' all(d, dim=0L)
#' all(d, dim=1L)
#' }
"all.torch.Tensor" <- function(x, dim, ...) {
    # quick version of torch$all
    # TODO: modify to use all arguments
    # all(dim, keepdim=False, out=None) → Tensor
    # DO NOT USE torch$tensor() to prevent warning:
    #            ... it is recommended to use sourceTensor.clone().detach()
    x <- torch$as_tensor(x, dtype = torch$uint8)
    # as.logical(torch$all(x)$numpy())
    if (missing(dim)) torch$all(x) else torch$all(x, dim=as.integer(dim))
}


#' any
#'
#' Returns True if any elements in the tensor are non-zero, False otherwise.
#' @param x tensor
#' @param dim dimension to reduce
#' @param ... other params (yet to be developed)
#' @export
#' @examples
#' \donttest{
#' a <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
#' b <- torch$BoolTensor(list(FALSE, TRUE, TRUE, TRUE))
#' c <- torch$BoolTensor(list(TRUE, TRUE, TRUE, FALSE))
#' any(a)
#' any(b)
#' any(c)
#' d <- torch$tensor(list(list(1, 0),
#'                        list(0, 0),
#'                        list(0, 1),
#'                        list(0, 0)), dtype=torch$uint8)
#' any(d)
#' any(d, dim=0L)
#' any(d, dim=1L)
#' }
"any.torch.Tensor" <- function(x, dim, ...) {
    # quick version of torch$any
    # TODO: modify to use all arguments
    # all(dim, keepdim=False, out=None) → Tensor
    # DO NOT USE torch$tensor() to prevent warning:
    #            ... it is recommended to use sourceTensor.clone().detach()
    x <- torch$as_tensor(x, dtype = torch$uint8)
    # as.logical(torch$any(x)$numpy())
    if (missing(dim)) torch$any(x) else torch$any(x, dim=as.integer(dim))
}


#' Two tensor operations
#'
#' @param a tensor
#' @param b tensor
#' @examples
#' \donttest{
#' a <- torch$Tensor(list(1, 1, 1))
#' b <- torch$Tensor(list(2, 2, 2))
#' s <- 2.0
#' a + b
#' b - a
#' a * b
#' a / s
#' a == b
#' a == a
#' a != a
#' x <- torch$Tensor(list(list(2, 2, 2), list(4, 4, 4)))
#' y <- torch$Tensor(list(list(1, 2, 1), list(3, 4, 5)))
#' x > y
#' x < y
#' x >= y
#' y <= x
#' diag <- torch$eye(3L)
#' zeros <- torch$zeros(c(3L, 3L))
#' diag & zeros
#' diag & diag
#' diag | diag
#' zeros | zeros
#' zeros & zeros
#' diag & zeros
#' diag | zeros
#' }
tensor_ops <- function(a, b) UseMethod("tensor_ops")



#' @describeIn tensor_ops Add two tensors
#' @export
"+.torch.Tensor" <- function(a, b) {
    if (any(class(a) == "torch.Tensor"))
        torch$add(a, b)
    else
        torch$add(b, a)
}


#' @describeIn tensor_ops Subtract two tensors
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
#' @describeIn tensor_ops Tensor multiplication
"*.torch.Tensor" <- function(a, b) {
    if (py_has_attr(torch, "multiply"))
        torch$multiply(a, b)
    else
        torch$mul(a, b)
}

#' @describeIn tensor_ops Divide two tensors
#' @export
"/.torch.Tensor" <- function(a, b) {
    torch$div(a, b)
}




#' @describeIn tensor_ops Compare two tensors if equal
#' @export
"==.torch.Tensor" <- function(a, b) {
    torch$as_tensor(torch$eq(a, b), dtype = torch$bool)
    # torch$BoolTensor(torch$eq(a, b))
}


tensor_not_equal <- function(x, y) {
    # there is not not_equal function in PyTorch
    x <- r_to_py(x$numpy())
    y <- r_to_py(y$numpy())
    torch$BoolTensor(np$not_equal(x, y))
}

#' @describeIn tensor_ops Compare two tensors if not equal
#' @export
"!=.torch.Tensor" <- function(a, b) {
    # there is not not_equal function in PyTorch
    # tensor_not_equal(a, b)
    # torch$ne(a, b)
    torch$as_tensor(torch$ne(a, b), dtype = torch$bool)
}

#' @describeIn one_tensor_op Logical NOT of a tensor
#' @export
"!.torch.Tensor" <- function(x) {
    # there is not logical not in torch
    # torch$BoolTensor(np$logical_not(a))
    torch$as_tensor(np$logical_not(x), dtype = torch$bool)
}


#' @describeIn tensor_ops A tensor less than another tensor
#' @export
"<.torch.Tensor" <- function(a, b) {
    # torch$lt(a, b)
    torch$as_tensor(torch$lt(a, b), dtype = torch$bool)
}


#' @describeIn tensor_ops A tensor less or equal than another tensor
#' @export
"<=.torch.Tensor" <- function(a, b) {
    # torch$le(a, b)
    torch$as_tensor(torch$le(a, b), dtype = torch$bool)
}


#' @describeIn tensor_ops A tensor greater than another tensor
#' @export
">.torch.Tensor" <- function(a, b) {
    # torch$gt(a, b)
    torch$as_tensor(torch$gt(a, b), dtype = torch$bool)
}

#' @describeIn tensor_ops A tensor greater or equal than another tensor
#' @export
">=.torch.Tensor" <- function(a, b) {
    # torch$ge(a, b)
    torch$as_tensor(torch$ge(a, b), dtype = torch$bool)
}



#' Dot product of two tensors
#' PyTorch dot function
#' @param a Tensor 1
#' @param b Tensor 2
#' @export
`%.*%` <- function(a, b) {
    torch$dot(a, b)
}


#' Matrix/Tensor multiplication of two tensors
#' PyTorch matmul
#' @param a Tensor 1
#' @param b Tensor 2
#' @export
`%**%` <- function(a, b) {
    torch$matmul(a, b)
}




#' @describeIn tensor_ops A tensor 'a' to the power of 'b'
#' @export
"^.torch.Tensor" <- function(a, b) {
    torch$pow(a, b)
}


#' @describeIn one_tensor_op Exponential of a tensor
#' @export
"exp.torch.Tensor" <- function(x) {
    torch$exp(x)
}


#' Logarithm of a tensor given the tensor and the base
#' @param x a tensor
#' @param base the base of the logarithm
#' @export
"log.torch.Tensor" <- function(x, base = exp(1L)) {
    if (is_tensor(base) || base != exp(1L)) {
        base <- torch$as_tensor(base, x$dtype)
        torch$log(x) / torch$log(base)
    } else {
        torch$log(x)
    }
}

#' Logarithm of a tensor in base 2
#' @param x a tensor
#' @export
#' @method log2 torch.Tensor
"log2.torch.Tensor" <- function(x) {
    torch$log2(x)
}

#' Logarithm of a tensor in base 10
#' @param x a tensor
#' @export
#' @method log10 torch.Tensor
"log10.torch.Tensor" <- function(x) {
    torch$log10(x)
}



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


#' @describeIn tensor_ops Logical AND of two tensors
#' @export
"&.torch.Tensor" <- function(a, b) {
    tensor_logical_and(a, b)
}


#' @describeIn tensor_ops Logical OR of two tensors
#' @export
"|.torch.Tensor" <- function(a, b) {
    tensor_logical_or(a, b)
}


