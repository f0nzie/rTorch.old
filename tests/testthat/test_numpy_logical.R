
source("tensor_functions.R")

context("numpy logical operations")

np <- import("numpy")

as_vector <- function(...) as.vector(...)

tensor_false <- torch$BoolTensor(list(0L))
tensor_true <- torch$BoolTensor(list(1L))

tensor_logical_and <- function(x, y) {
  x <- r_to_py(x$numpy())
  y <- r_to_py(y$numpy())
  torch$BoolTensor(np$logical_and(x, y))
}

tensor_logical_or <- function(x, y) {
  x <- r_to_py(x$numpy())
  y <- r_to_py(y$numpy())
  torch$BoolTensor(np$logical_or(x, y))
}


context("AND logical operations")

test_that("numpy AND yields logical arrays", {
  expect_false(np$logical_and(1, 0))
  expect_false(np$logical_and(TRUE, FALSE))
  expect_equal(np$logical_and(c(TRUE, FALSE), c(FALSE,FALSE)), array(c(FALSE, FALSE)))
  x = np$arange(5)
  expect_equal(as_vector(np$logical_and(x>1, x<4)), c(FALSE, FALSE, TRUE, TRUE, FALSE))
})

test_that("np$logical_and() return R logical", {
  expect_equal(class(np$logical_and(TRUE, FALSE)), "logical")
})


test_that("tensor+numpy AND yields logical arrays", {
  p <- torch$BoolTensor(list(1, 0))
  q <- torch$BoolTensor(list(0, 1))
  A <- torch$BoolTensor(list(0L))
  B <- torch$BoolTensor(list(0L))
  expect_equal(tensor_logical_and(A, B), tensor_false)
  expect_equal(tensor_logical_and(p, q), torch$BoolTensor(list(FALSE, TRUE)))
})

test_that("tensor_logical_and", {
  A <- torch$BoolTensor(list(0L))
  B <- torch$BoolTensor(list(0L))
  C <- torch$BoolTensor(list(1L))
  expect_equal((A & B), tensor_false)
  expect_equal((A & C), tensor_false)
  expect_equal((A & C), tensor_true)
})



context("OR logical operations")

test_that("numpy OR yields logical arrays", {
  expect_false(np$logical_or(0, 0))
  expect_true(np$logical_or(TRUE, FALSE))
  expect_equal(np$logical_or(c(TRUE, FALSE), c(FALSE,FALSE)), array(c(TRUE, FALSE)))
  x = np$arange(5)
  # print(np$logical_or(x>1, x<4))
  expect_equal(np$logical_or(x>1, x<4), array(c(TRUE, TRUE, TRUE, TRUE, TRUE)))
})

test_that("tensor_logical_or", {
  A <- torch$BoolTensor(list(0L))
  B <- torch$BoolTensor(list(0L))
  C <- torch$BoolTensor(list(1L))
  expect_equal((A & B), tensor_false)
  expect_equal((A & C), tensor_true)
  expect_equal((A & C), tensor_true)
})

