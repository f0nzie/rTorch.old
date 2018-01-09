library(testthat)


test_that("reads the size of a tensor ", {
    t_size <- torch$Size(c(256L, 3L, 9L, 9L, 2L))   # size of the tensor
    t      <- torch$Tensor(t_size)                  # build the tensor
    expect_equal(torch_size(t), c(256, 3, 9, 9, 2))
})


test_that("the function works for two different objects", {
    x = torch$rand(20L, 50L, 100L)   # build the tensor
    x_size <- x$size()               # get the size of a tensor
    expect_s3_class(x, "torch.FloatTensor")
    expect_s3_class(x_size, "torch.Size")
    expect_equal(torch_size(x_size), c(20, 50, 100))
})


