library(testthat)

skip_on_cran()

source("utils.R")

context("PyTorch version")

VERSIONS <- c("1.1", "1.0", "1.2", "1.3", "1.4", "1.5", "1.6")

test_that("PyTorch version ", {
    expect_true(substr(torch$`__version__`, 1, 3)  %in% VERSIONS
                  )
})


test_that("CUDA is not available", {
    expect_equal(torch$cuda$is_available(), FALSE)
})


skip_on_travis()
test_that("Number of CPUs", {
    expect_true(torch$get_num_threads() >= 1)
})



context("package config functions")

test_that("torch_version returns value", {
  expect_true(torch_version() %in% VERSIONS)
  # print(torch_version())

})
