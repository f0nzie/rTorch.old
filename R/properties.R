
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
#' @description A copy o an array or tensor might be needed to prevent warnings
#' by new PyTorch versions
#'
#' @param object a torch tensor or numpy array or R array
#' @export
make_copy <- function(object, ...) {
    if (class(object) == "torch.Tensor") {
        obj <- object$copy_(object)
    }
    else if (class(object) == "numpy.ndarray") {
        obj <- object$copy()
    } else {
        obj <- r_to_py(object)$copy()
    }
    return(obj)
}
