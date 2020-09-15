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


context("equal")

test_that("equal UINT8 tensors", {
    # these are the original conditions without converting to boolean
    if (package_version(torch_version()) <= "1.1") {
        expect_output(print((UINT8_TRUE_TENSOR == UINT8_TRUE_TENSOR)), "tensor(1, dtype=torch.uint8)", fixed = TRUE)
        expect_output(print((UINT8_TRUE_TENSOR == UINT8_FALSE_TENSOR)), "tensor(0, dtype=torch.uint8)", fixed = TRUE)
    } else {
        expect_output(print((UINT8_TRUE_TENSOR == UINT8_TRUE_TENSOR)), "tensor(True)", fixed = TRUE)
    }

})

test_that("equal BOOL tensors", {
    # these are the original conditions without converting to boolean
    if (package_version(torch_version()) <= "1.1") {
        expect_output(print((BOOL_TRUE_TENSOR == BOOL_TRUE_TENSOR)), "tensor(1, dtype=torch.uint8)", fixed = TRUE)
        expect_output(print((BOOL_TRUE_TENSOR == BOOL_FALSE_TENSOR)), "tensor(0, dtype=torch.uint8)", fixed = TRUE)
    } else {
        expect_output(print((BOOL_TRUE_TENSOR == BOOL_TRUE_TENSOR)), "tensor(True)", fixed = TRUE)
        expect_output(print((BOOL_TRUE_TENSOR == BOOL_FALSE_TENSOR)), "tensor(False)", fixed = TRUE)
    }

})


context("not equal")

test_that("not equal UINT8 tensors", {
    # these are the original conditions without converting to boolean
    if (package_version(torch_version()) <= "1.1") {
        expect_output(print((UINT8_TRUE_TENSOR != UINT8_TRUE_TENSOR)), "tensor(0, dtype=torch.uint8)", fixed = TRUE)
        expect_output(print((UINT8_TRUE_TENSOR != UINT8_FALSE_TENSOR)), "tensor(1, dtype=torch.uint8)", fixed = TRUE)
    } else {
        expect_output(print((UINT8_TRUE_TENSOR != UINT8_TRUE_TENSOR)), "tensor(False)", fixed = TRUE)
        expect_output(print((UINT8_TRUE_TENSOR != UINT8_FALSE_TENSOR)), "tensor(True)", fixed = TRUE)
    }

})

test_that("not equal BOOL tensors", {
    # these are the original conditions without converting to boolean
    if (package_version(torch_version()) <= "1.1") {
        expect_output(print((BOOL_TRUE_TENSOR != BOOL_TRUE_TENSOR)), "tensor(0, dtype=torch.uint8)", fixed = TRUE)
        expect_output(print((BOOL_TRUE_TENSOR != BOOL_FALSE_TENSOR)), "tensor(1, dtype=torch.uint8)", fixed = TRUE)
    } else {
        expect_output(print((BOOL_TRUE_TENSOR != BOOL_TRUE_TENSOR)), "tensor(False)", fixed = TRUE)
        expect_output(print((BOOL_TRUE_TENSOR != BOOL_FALSE_TENSOR)), "tensor(True)", fixed = TRUE)
    }

})
