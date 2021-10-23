library(testthat)

source("helper_utils.R")

skip_if_no_torch()
skip_if_no_cuda()

test_that("CUDA is available", {
    # print(torch$cuda$is_available())
    expect_equal(torch$cuda$is_available(), TRUE)
})
