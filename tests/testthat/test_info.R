library(testthat)

source("tensor_functions.R")


context("PyTorch version")

test_that("PyTorch version is 1.1.0", {
    expect_equal(torch$`__version__`, "1.1.0")
})


test_that("CUDA is not available", {
    expect_equal(torch$cuda$is_available(), FALSE)
})


test_that("Number of CPUs", {
    expect_equal(torch$get_num_threads(), 4)
})



context("package config functions")

test_that("torch_version returns value", {
  expect_equal(torch_version(), "1.1")
  # print(torch_version())

})
