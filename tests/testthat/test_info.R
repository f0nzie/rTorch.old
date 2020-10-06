library(testthat)

source("utils.R")

skip_if_no_torch()


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



context("package version")

test_that("torch_version returns value", {
  expect_true(torch_version() %in% VERSIONS)
  version <- torch$`__version__`
  version <- strsplit(version, ".", fixed = TRUE)[[1]]
  expect_equal(length(version), 3)

})
