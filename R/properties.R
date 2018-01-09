
# #' @importFrom reticulate import_builtins
#' @export
torch_size <- function(obj) {
    py <- reticulate::import_builtins()
    if (any(class(obj) %in% "torch.tensor._TensorBase")) {
        it <- iterate(py$enumerate(obj$size()))
    } else if (any(class(obj) %in% "torch.Size")) {
        it <- iterate(py$enumerate(obj))
    } else {
        stop("Not a tensor object")
    }
    return(unlist(sapply(it, `[`)[2,]))
}
