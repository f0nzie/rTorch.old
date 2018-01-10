

#' MNIST database of handwritten digits
#'
#' Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
#'
#' @param ntrain number of training samples
#' @param ntest number of test samples
#' @param onehot boolean
#' @family datasets
#'
#' @export
dataset_mnist <- function(ntrain = 60000, ntest = 10000, onehot = TRUE) {
    # import data_utils module
    python_path <- system.file("python", package = "rTorch")
    tools <- import_from_path("torchtools", path = python_path)
    tools$data_util$load_mnist(ntrain, ntest, onehot)
}


