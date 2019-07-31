
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
