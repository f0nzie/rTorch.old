library(testthat)

source("helper_utils.R")

skip_if_no_torch()
skip_if_no_cuda()

test_that("CUDA is available", {
    # print(torch$cuda$is_available())
    expect_equal(torch$cuda$is_available(), TRUE)
})

test_that("device is CUDA selected", {
    torch_object <- c("torch.device")
    two_objects  <- c("torch.device", "python.builtin.object")
    dev <- torch$device(ifelse(torch$cuda$is_available(), "cuda", "cpu"))

    expect_true(class(dev)[1] %in% torch_object)
    expect_equal(class(dev), two_objects)
})