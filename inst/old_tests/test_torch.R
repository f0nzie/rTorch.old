library(testthat)
library(rTorch)
# library(tensorflow)






arr <- function (...) {
    # create an array with the specified dimensions, and fill it with consecutive
    # increasing integers
    dims <- unlist(list(...))
    array(1:prod(dims), dim = dims)
}

randn <- function (...) {
    dim <- c(...)
    array(rnorm(prod(dim)), dim = dim)
}



# set up arrays
x1_ <- arr(3)
x2_ <- arr(3L, 3L)
x3_ <- arr(3L, 3L, 3L)

# cast to Tensors
x1 <- torch$LongTensor(x1_)
x2 <- torch$LongTensor(x2_)
x3 <- torch$LongTensor(x3_)

# extract as arrays
y1_ <- x1_[1]
y2_ <- x2_[1, 2]
y3_ <- x3_[1, 2, 3]


print(torch$Size(x2))


# # extract as Tensors
# y1 <- x1[1]
# y2 <- x2[1, 2]
# y3 <- x3[1, 2, 3]


# matrixmultiply <- function(mat1, mat2) {
#     n  <-  mat1$size(0L)
#     m <- mat1$size(1L)
#     p <- mat2$size(1L)
#
#     res <- torch$zeros(n, p)
#     print(res)
#
#     for (i in seq(1, res$size(0L))) {
#         for (j in seq(1, res$size(1L))) {
#             for (k in 1:m) {
#                 res$_set_index((i, j), sum(mat1[i, k] * mat2[k, j]))
#             }
#
#         }
#     }
#     return(res)
# }

n <- 10L
m <- 10L
p <- 5L

mat1 <- torch$randn(n, m)
# print(mat1)
mat2 <-  torch$randn(m, p)
res <-  torch$mm(mat1, mat2)

# res2 = matrixmultiply(mat1, mat2)
