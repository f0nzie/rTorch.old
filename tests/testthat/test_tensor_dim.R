library(testthat)

skip_on_cran()

source("tensor_functions.R")

# test function tensor_dim ----
context("test function tensor_dim")

test_that("tensor dimension is 5D: 60000x3x25x28x28", {
    img <- torch$ones(60000L, 3L, 25L, 28L, 28L)
    expect_equal(tensor_dim(img), c(60000, 3, 25, 28, 28))
    expect_equal(tensor_dim_(img), 5)
})

test_that("tensor dimension is 4D: 60000x3x28x28", {
    img <- torch$ones(60000L, 3L, 28L, 28L)
    expect_equal(tensor_dim(img), c(60000, 3, 28, 28))
    expect_equal(tensor_dim_(img), 4)
})

test_that("tensor dimension is 3D: 3x28x28", {
    img <- torch$ones(3L, 28L, 28L)
    expect_equal(tensor_dim(img), c(3, 28, 28))
    expect_equal(tensor_dim_(img), 3)
})


test_that("tensor dimension is 2D: 28x28", {
    img <- torch$ones(28L, 28L)
    expect_equal(tensor_dim(img), c(28, 28))
    expect_equal(tensor_dim_(img), 2)
})


test_that("tensor dimension is 1D: 784", {
    img <- torch$ones(28L * 28L)
    # expect_equal(tensor_dim(img), c(28, 28))
    expect_equal(tensor_dim(img), 784)
    expect_equal(tensor_dim_(img), 1)
})


