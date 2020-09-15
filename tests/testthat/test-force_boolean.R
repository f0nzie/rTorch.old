library(testthat)

source("utils.R")
skip_if_no_torch()


UINT8_TRUE_TENSOR  <- torch$as_tensor(1L, dtype=torch$uint8)
UINT8_FALSE_TENSOR <- torch$as_tensor(0L, dtype=torch$uint8)

BOOL_TRUE_TENSOR  <- torch$as_tensor(1L, dtype=torch$bool)
BOOL_FALSE_TENSOR <- torch$as_tensor(0L, dtype=torch$bool)

m0 = torch$zeros(3L, 5L)
m1 = torch$ones(3L, 5L)
m2 = torch$eye(3L, 5L)


context("tensor_logical_and")

test_that("logical AND of TRUE tensors", {

    expect_true(as.logical(((UINT8_TRUE_TENSOR & UINT8_TRUE_TENSOR) == UINT8_TRUE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_TRUE_TENSOR & BOOL_TRUE_TENSOR) == BOOL_TRUE_TENSOR)$numpy()))

    expect_output(print((UINT8_TRUE_TENSOR & UINT8_TRUE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_TRUE_TENSOR & BOOL_TRUE_TENSOR)$data$type()), "torch.BoolTensor")
})

test_that("logical AND of TRUE and FALSE tensors", {

    expect_true(as.logical(((UINT8_TRUE_TENSOR & UINT8_FALSE_TENSOR) == UINT8_FALSE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_TRUE_TENSOR & BOOL_FALSE_TENSOR) == BOOL_FALSE_TENSOR)$numpy()))

    expect_output(print((UINT8_TRUE_TENSOR & UINT8_FALSE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_TRUE_TENSOR & BOOL_FALSE_TENSOR)$data$type()), "torch.BoolTensor")
})

test_that("logical AND of FALSE and FALSE tensors", {

    expect_true(as.logical(((UINT8_FALSE_TENSOR & UINT8_FALSE_TENSOR) == UINT8_FALSE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_FALSE_TENSOR & BOOL_FALSE_TENSOR) == BOOL_FALSE_TENSOR)$numpy()))

    expect_output(print((UINT8_FALSE_TENSOR & UINT8_FALSE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_FALSE_TENSOR & BOOL_FALSE_TENSOR)$data$type()), "torch.BoolTensor")
})



context("tensor_logical_or")

test_that("logical OR of TRUE tensors", {

    expect_true(as.logical(((UINT8_TRUE_TENSOR | UINT8_TRUE_TENSOR) == UINT8_TRUE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_TRUE_TENSOR | BOOL_TRUE_TENSOR) == BOOL_TRUE_TENSOR)$numpy()))

    expect_output(print((UINT8_TRUE_TENSOR | UINT8_TRUE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_TRUE_TENSOR | BOOL_TRUE_TENSOR)$data$type()), "torch.BoolTensor")
})


test_that("logical OR of TRUE and FALSE tensors", {

    expect_true(as.logical(((UINT8_TRUE_TENSOR | UINT8_FALSE_TENSOR) == UINT8_TRUE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_TRUE_TENSOR | BOOL_FALSE_TENSOR) == BOOL_TRUE_TENSOR)$numpy()))

    expect_output(print((UINT8_TRUE_TENSOR | UINT8_FALSE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_TRUE_TENSOR | BOOL_FALSE_TENSOR)$data$type()), "torch.BoolTensor")
})

test_that("logical OR of FALSE and FALSE tensors", {

    expect_true(as.logical(((UINT8_FALSE_TENSOR | UINT8_FALSE_TENSOR) == UINT8_FALSE_TENSOR)$numpy()))
    expect_true(as.logical(((BOOL_FALSE_TENSOR | BOOL_FALSE_TENSOR) == BOOL_FALSE_TENSOR)$numpy()))

    expect_output(print((UINT8_FALSE_TENSOR | UINT8_FALSE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((BOOL_FALSE_TENSOR | BOOL_FALSE_TENSOR)$data$type()), "torch.BoolTensor")
})



context("tensor_logical_NOT")

test_that("logical NOT for two tensor types", {

    expect_true(as.logical(((!UINT8_TRUE_TENSOR) == UINT8_FALSE_TENSOR)$numpy()))
    expect_true(as.logical(((!UINT8_FALSE_TENSOR) == UINT8_TRUE_TENSOR)$numpy()))

    expect_true(as.logical(((!BOOL_TRUE_TENSOR) == BOOL_FALSE_TENSOR)$numpy()))
    expect_true(as.logical(((!BOOL_FALSE_TENSOR) == BOOL_TRUE_TENSOR)$numpy()))

    expect_output(print((!UINT8_TRUE_TENSOR)$data$type()), "torch.ByteTensor")
    expect_output(print((!BOOL_TRUE_TENSOR)$data$type()), "torch.BoolTensor")

})
