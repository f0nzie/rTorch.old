

#' @export
"+.torch.tensor" <- function(a, b) {
    torch$add(a, b)
}



#' @export
"*.torch.tensor" <- function(a, b) {
    if (py_has_attr(torch, "multiply"))
        torch$multiply(a, b)
    else
        torch$mul(a, b)
}
