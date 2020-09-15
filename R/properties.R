
#' @title Size of a torch tensor object
#' @description Get the size of a torch tensor or of torch.size object
#'
#' @param obj a torch tensor object
#' @export
torch_size <- function(obj) {
    py <- reticulate::import_builtins()
    if (any(class(obj) %in% "torch._C._TensorBase")) {
        it <- iterate(py$enumerate(obj$size()))
    } else if (any(class(obj) %in% "torch.Size")) {
        # also take obj$size() to extract size
        it <- iterate(py$enumerate(obj))
    } else {
        stop("Not a tensor object")
    }
    return(unlist(sapply(it, `[`)[2,]))
}


#' @title Make copy of tensor, numpy array or R array
#' @description A copy of an array or tensor might be needed to prevent warnings
#' by new PyTorch versions on overwriting the numpy object
#'
#' @param object a torch tensor or numpy array or R array
#' @param ... additional parameters
#' @export
make_copy <- function(object, ...) {
    # the object could hold multiple classes
    if (any(class(object) %in% "torch.Tensor")) {
        obj <- object$copy_(object)
    }
    else if (any(class(object) %in% "numpy.ndarray")) {
        obj <- object$copy()
    } else {
        # it is an R object
        rtp_obj <- r_to_py(object)
        obj <- rtp_obj$copy()
    }
    return(obj)
}



#' @title Convert tensor to boolean type
#' @description Convert a tensor to a boolean equivalent tensor
#'
#' @param x a torch tensor
#' @export
as_boolean <- function(x) {
    torch$as_tensor(x, dtype = torch$bool)
}


#' @title Is the object a tensor
#' @description Determine if the object is a tensor by looking inheritance
#'
#' @param obj an object
#' @export
is_tensor <- function(obj) inherits(obj, "torch.Tensor")  # do not use torch.tensor

