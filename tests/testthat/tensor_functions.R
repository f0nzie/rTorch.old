library(reticulate)


torch       <- import("torch")
torchvision <- import("torchvision")
nn          <- import("torch.nn")
transforms  <- import("torchvision.transforms")
dsets       <- import("torchvision.datasets")
builtins    <- import_builtins()
np          <- import("numpy")


tensor_dot <- function(A, B) {
    torch$dot(A, B)
}

tensor_dim <- function(tensor) {
    builtins$list(tensor$size())
}

tensor_dim_ <- function(tensor) {
    size <- builtins$list(tensor$size())
    length(size)
}

tensor_sum <- function(tensor) {
    tensor$sum()$item()
}

# is_tensor <- function(object) {
#     class(object) %in% c("torch.Tensor")
#     class_obj <- class(object)
#     all(class_obj[grepl("Tensor", class_obj)] %in%
#             c("torch.Tensor", "torch._C._TensorBase"))
# }

py_object_last <- function(object) {
    if (py_has_length(object)) py_len(object) - 1L
    else stop()
}

py_has_length <- function(object) {
    # ifelse(any(py_list_attributes(object) %in% c("__len__")), TRUE, FALSE)
    tryCatch({
        any(py_list_attributes(object) %in% c("__len__"))
    },
        error = function(e) {
            message("object has no __len__ attribute")
            # print(e)
            return(FALSE)
        }
    )
}


