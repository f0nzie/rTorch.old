library(testthat)

source("helper_utils.R")

skip_if_no_torch()
skip_if_no_cuda()

test_that("CUDA is available", {
    expect_equal(torch$cuda$is_available(), TRUE)
})

test_that("device is CUDA selected", {
    torch_object <- c("torch.device")
    two_objects  <- c("torch.device", "python.builtin.object")
    dev <- torch$device(ifelse(torch$cuda$is_available(), "cuda", "cpu"))

    expect_true(class(dev)[1] %in% torch_object)
    expect_equal(class(dev), two_objects)
})


test_that("objects get assigned to cuda or cpu", {
    dev <- torch$device(ifelse(torch$cuda$is_available(), "cuda", "cpu"))
    x <-  10L
    y <- 200L
    z <- 3L

    t1 <- torch$randn(x, y, z)
    t2 <- torch$randn(x, y, z)$to(dev)

    expect_false(t1$is_cuda)
    expect_true(t2$is_cuda)

    t1$to(dev)
    expect_false(t1$is_cuda)

    t1 <- t1$to(dev)
    expect_true(t1$is_cuda)

    expect_equal(t1$is_cuda, t2$is_cuda)

})


test_that("Linear Model class to CUDA", {
    prs <- py_run_string("
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)

    def forward(self, x):
        x = self.l1(x)
        return x
")
    model <- prs$M()
    expect_true(any(class(model) %in% c("torch.nn.modules.module.Module")))

    dev <- torch$device(ifelse(torch$cuda$is_available(), "cuda", "cpu"))
    model$to(dev)
    expect_true(reticulate::iter_next(model$parameters())$is_cuda)

    # rm(prs)
    # remove Python objects so they are not found later in tests
    py_run_string("del M")
    py_run_string("del nn")
    py_run_string("del torch")
    rm(prs)
})
