source("utils.R")

skip_if_no_torch()

context("numpy logical operations")

np <- import("numpy")

as_vector <- function(...) as.vector(...)

tensor_true  <<- torch$BoolTensor(list(1L))
tensor_false <<- torch$BoolTensor(list(0L))

TRUE_TENSOR  <- torch$as_tensor(1L, dtype=torch$uint8)
FALSE_TENSOR <- torch$as_tensor(0L, dtype=torch$uint8)

test_that("sample tensors as logical", {
    expect_equal(tensor_true$numpy(), array(TRUE))
    expect_equal(tensor_false$numpy(), array(FALSE))

    expect_equal(TRUE_TENSOR$numpy(), array(1))
    expect_equal(FALSE_TENSOR$numpy(), array(0))
})


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
  p <- torch$BoolTensor(make_copy(list(1, 0)))
  q <- torch$BoolTensor(make_copy(list(0, 1)))
  A <- torch$BoolTensor(make_copy(list(0L)))
  B <- torch$BoolTensor(make_copy(list(0L)))

  expect_output(print(tensor_logical_and(A, B)$data$type()), "torch.BoolTensor")
  expect_output(print(tensor_true$data$type()), "torch.BoolTensor")

  expect_tensor_equal(tensor_logical_and(A, B), tensor_false)
  expect_tensor_equal(tensor_logical_and(A, B), !tensor_true)   # RuntimeExpected object of scalar type Bool but got scalar type Byte for argument #2 'other'
  expect_tensor_equal(tensor_logical_and(p, q), torch$BoolTensor(make_copy(list(FALSE, FALSE))))
})

test_that("tensor_logical_and", {
  A <- torch$BoolTensor(list(1L))
  B <- torch$BoolTensor(list(1L))
  C <- torch$BoolTensor(list(0L))
  D <- torch$BoolTensor(list(0L))

  # expect_equal((A & B), tensor_false)
  # expect_equal((A & C), tensor_false)
  # expect_equal((C & C), tensor_true)
  # expect_equal((C & C), tensor_false)

  expect_tensor_equal((A&B),  torch$tensor(list(TRUE), dtype=torch$bool))
  expect_tensor_equal((A&C),  torch$tensor(list(FALSE), dtype=torch$bool))
  expect_tensor_equal((A&D),  torch$tensor(list(FALSE), dtype=torch$bool))
  expect_tensor_equal((A&A),  torch$tensor(list(TRUE), dtype=torch$bool))
  expect_tensor_equal((D&D),  torch$tensor(list(FALSE), dtype=torch$bool))
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
  A <- torch$BoolTensor(list(0L, 0L))
  B <- torch$BoolTensor(list(0L, 1L))
  C <- torch$BoolTensor(list(1L, 0L))
  D <- torch$BoolTensor(list(1L, 1L))


  expect_tensor_equal((A|A),  torch$tensor(c(FALSE, FALSE), dtype=torch$bool))
  expect_tensor_equal((A|B),  torch$tensor(c(FALSE, TRUE), dtype=torch$bool))
  expect_tensor_equal((A|C), !torch$tensor(c(FALSE, TRUE), dtype=torch$bool)) # RuntimeExpected object of scalar type Bool but got scalar type Byte for argument #2 'other'
  expect_tensor_equal((A|D),  torch$tensor(c(TRUE, TRUE), dtype=torch$bool))
  expect_tensor_equal((B|C),  torch$tensor(c(TRUE, TRUE), dtype=torch$bool))
  expect_tensor_equal((B|D),  torch$tensor(c(TRUE, TRUE), dtype=torch$bool))
  expect_tensor_equal((D|D),  torch$tensor(c(TRUE, TRUE), dtype=torch$bool))

})

