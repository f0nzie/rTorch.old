library(rTorch)
source("tests/testthat/helper_utils.R")

# library(rTorch)
# pkg <- "package:rTorch"
# detach(pkg, character.only = TRUE)
# sessionInfo()

m1 = torch$ones(3L, 5L)
m1$shape

tensor_ndim(m1)
# [1] 2
tensor_dim(m1)
# [1] 3 5

# library(magrittr)
# m1 %>%
#   as.array()

all_boolean <- function(x) {
  # convert tensor of 1s and 0s to a unique boolean
  as.logical(torch$all(x)$numpy())
}


# testing that new generic all, any for tensors work
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5
all(torch$lt(C, A))
all(C < A)
all(A < C)


all(torch$tensor(list(1, 1, 1)))
all(torch$tensor(list(1, 1, 0)))
any(torch$tensor(list(1, 1, 0)))
any(torch$tensor(list(0, 0, 0)))
all(torch$eye(3L))
any(torch$eye(3L))


# library(testthat)
# res <- install_pytorch(version = "1.6", conda_python_version = "3.7",
#                        extra_packages = "pandas",
#                        dry_run = FALSE)
# #detach("package:rTorch", unload=TRUE)
# #require(rTorch)
# # devtools::reload(pkg = ".", quiet = FALSE)
# # use torch_config for live test
# #unloadNamespace("rTorch")
# # detach("package:rTorch", unload=TRUE)
# #library(rTorch)
# res <- torch_config()
# print(res)
# expect_equal(res$available, TRUE)
# expect_equal(res$version_str, "1.6.0")
# expect_equal(res$python_version, "3.7")
# expect_equal(res$numpy_version, "1.19.1")
# expect_equal(res$env_name, "r-torch")
#
#
#
# res <- install_pytorch(version = "1.3", conda_python_version = "3.6",
#                        extra_packages = c("pandas", "matplotlib"),
#                        dry_run = FALSE)
#
# # detach("package:rTorch", unload=TRUE)
# # require(rTorch)
# # devtools::reload(pkg = ".", quiet = FALSE)
# #unloadNamespace("rTorch")
# # detach("package:rTorch", unload=TRUE)
# #library(rTorch)
# res <- torch_config()
# print(res)
# expect_equal(res$available, TRUE)
# expect_equal(res$version_str, "1.3.0")
# expect_equal(res$python_version, "3.6")
# expect_equal(res$numpy_version, "1.19.1")
# expect_equal(res$env_name, "r-torch")
