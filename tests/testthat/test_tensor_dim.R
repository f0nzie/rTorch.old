library(testthat)

source("utils.R")
skip_if_no_torch()


# test function tensor_dim ----
context("test function tensor_dim")

test_that("tensor dimension is 5D: 60000x3x5x28x28", {
    img <- torch$ones(60000L, 3L, 5L, 28L, 28L)
    expect_equal(tensor_dim(img), c(60000, 3, 5, 28, 28))
    expect_equal(tensor_ndim(img), 5)
})

test_that("tensor dimension is 4D: 60000x3x28x28", {
    img <- torch$ones(60000L, 3L, 28L, 28L)
    expect_equal(tensor_dim(img), c(60000, 3, 28, 28))
    expect_equal(tensor_ndim(img), 4)
})

test_that("tensor dimension is 3D: 3x28x28", {
    img <- torch$ones(3L, 28L, 28L)
    expect_equal(tensor_dim(img), c(3, 28, 28))
    expect_equal(tensor_ndim(img), 3)
})


test_that("tensor dimension is 2D: 28x28", {
    img <- torch$ones(28L, 28L)
    expect_equal(tensor_dim(img), c(28, 28))
    expect_equal(tensor_ndim(img), 2)
})


test_that("tensor dimension is 1D: 784", {
    img <- torch$ones(28L * 28L)
    # expect_equal(tensor_dim(img), c(28, 28))
    expect_equal(tensor_dim(img), 784)
    expect_equal(tensor_ndim(img), 1)
})


