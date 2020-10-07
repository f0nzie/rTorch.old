library(testthat)

source("helper_utils.R")

skip_if_no_torch()


TRUE_TENSOR  <- torch$as_tensor(1L, dtype=torch$uint8)
FALSE_TENSOR <- torch$as_tensor(0L, dtype=torch$uint8)

all_boolean <- function(x) {
  # convert tensor of 1s and 0s to a unique boolean
  as.logical(torch$all(x)$numpy())
}

expect_all_equal <- function(x, y) {
  all_boolean(torch$eq(x, y))
}

expect_false_tensor <- function(x) {
  expect_true(torch$equal(x, FALSE_TENSOR))
}

expect_true_tensor <- function(x) {
  expect_true(torch$equal(x, TRUE_TENSOR))
}

expect_tensor_equal <- function(a, b) {
  expect_true(torch$equal(a, b))
}


context("dim on tensor")

test_that("R dim function works as well", {
  m1 = torch$ones(3L, 5L)
  shape <- m1$dim()
  # expect_equal(ifelse(!is.null(shape), shape, NULL), 2)
  expect_equal(dim(m1), c(3, 5))
  expect_equal(m1$dim(), 2)
})


test_that("tensor dimension is 4D: 60000x3x28x28", {
  img <- torch$ones(60000L, 3L, 28L, 28L)
  # expect_equal(tensor_ndim(img), 4)
  expect_equal(dim(img), c(60000, 3, 28, 28))
  expect_equal(img$dim(), 4)
})


context("tensor arithmetic")

test_that("tensor addition as x+y", {
  x = torch$rand(5L, 4L)
  y = torch$rand(5L, 4L)
  xy_add <- torch$add(x, y)
  expect_equal(xy_add, x+y)
})

test_that("tensor subtraction as x-y", {
  x = torch$rand(5L, 4L)
  y = torch$rand(5L, 4L)
  xy_sub <- torch$sub(x, y)
  expect_equal(xy_sub, x-y)

  ny <- -y
  expect_equal(ny, torch$neg(y))
})

test_that("minus sign negates the tensor", {
  x = torch$rand(5L, 4L)
  y = torch$rand(5L, 4L)
  ny <- -y; nx <- -x
  expect_equal(nx, torch$neg(x))
  expect_equal(ny, torch$neg(y))
})



test_that("* sign multiplies", {
  result <- torch$Tensor(list(1L, 1L, 1L, 1L))
  a = torch$ones(4L, dtype=torch$float64)
  b = torch$ones(4L, dtype=torch$float64)
  expect_equal(a*b, result)
})

test_that("* sign multiplication of tensor times a scalar", {
  result <- torch$Tensor(list(2, 2, 2, 2))
  a = torch$ones(4L, dtype=torch$float64)
  b = torch$ones(4L, dtype=torch$float64)
  i <- 2L
  expect_equal(a*i, result)
})

test_that("* sign multiplication of tensor times a negative scalar", {
  result <- torch$Tensor(list(-2, -2, -2, -2))
  a = torch$ones(4L, dtype=torch$float64)
  b = torch$ones(4L, dtype=torch$float64)
  i <- -2L
  expect_equal(a*i, result)
})


test_that("length of tensor is the same as numel()", {
  a_matrix <- matrix(rnorm(100), ncol = 2)
  a_tensor <- torch$as_tensor(make_copy(a_matrix))
  expect_equal(a_tensor$numel(), 100)
  expect_equal(length(a_tensor), 100)
})


context("tensor equal ==")
test_that("tensor equal as x == y", {
  a <- torch$Tensor(list(2, 3, 4, 5))
  b <- torch$Tensor(list(2, 3, 4, 5))
  # print(a == b)
  expect_equal((a == b), torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE)))

    x = torch$rand(5L, 4L)
    y = x * 1.0
    expected <- torch$tensor(list(
      list(1, 1, 1, 1),
      list(1, 1, 1, 1),
      list(1, 1, 1, 1),
      list(1, 1, 1, 1),
      list(1, 1, 1, 1)
    ))
    # true_tensor <- torch$as_tensor(1L, dtype=torch$uint8)
    # print(true_tensor)
    expect_equal(all(torch$eq(x, y)), TRUE_TENSOR)
    expect_equal(all(x == y), TRUE_TENSOR)
    expect_equal((x == y), expected)
})


context("tensors not equal !=")

test_that("tensor equal as x == y", {
  A <- torch$ones(60000L, 3L, 28L, 28L)
  B <- torch$ones(60000L, 3L, 28L, 28L)
  expect_equal(all(A == B), TRUE_TENSOR)
  expect_equal(all(A != B), TRUE_TENSOR)
})


test_that("tensor not equal as x != y", {
  a <- torch$Tensor(list(2, 2, 2, 2))
  b <- torch$Tensor(list(2, 2, 2, 1))
  expect_equal(all(a == b), FALSE_TENSOR)
  expect_equal(any(a != b), TRUE_TENSOR)
  # expect_equal((a != b), torch$BoolTensor(list(FALSE, FALSE, FALSE, TRUE)))

  x = torch$rand(5L, 4L)
  y = torch$rand(5L, 4L)
  expect_false_tensor(all(torch$eq(x, y)))
  expect_true_tensor(all(x != y))


  A <- torch$ones(60000L, 3L, 28L, 28L)    # all ones
  B <- torch$zeros(60000L, 3L, 28L, 28L)   # all zeroes
  # expect_true_tensor(all(A != B))
  # expect_false_tensor(all(A != B))
  expect_tensor_equal(all(A != B), TRUE_TENSOR)

  A <- torch$ones(60000L, 1L, 28L, 28L)
  B <- torch$ones(60000L, 1L, 28L, 28L)

  expect_tensor_equal(all(A != B), FALSE_TENSOR)
})


test_that("numpy logical-not works as !", {
  A <- torch$ones(5L)
  expect_false_tensor(all(!A))

  Z <- torch$zeros(5L)
  expect_false_tensor(all(Z))
  expect_true_tensor(all(!Z))

  E <- torch$eye(3L)
  expect_true_tensor(all(torch$diag(E)))

  # get the diagonal
  E_diag <- E$diag()
  expect_equal(E_diag, torch$Tensor(list(1, 1, 1)))

  NE <- torch$as_tensor(!E, dtype = torch$uint8)
  NE_diag <- NE$diag()
  expect_equal(E_diag, torch$Tensor(list(0, 0, 0)))

  expect_true_tensor(all(E$diag()))
  expect_false_tensor(all(NE$diag()))

})



context("tensor comparison")

test_that("tensor is less than ", {
  A <- torch$ones(60000L, 1L, 28L, 28L)
  C <- A * 0.5
  expect_true_tensor(all(torch$lt(C, A)))
  expect_true_tensor(all(C < A))
  expect_false_tensor(all(A < C))
})


test_that("tensor is greater than ", {
  A <- torch$ones(60000L, 1L, 28L, 28L)
  D <- A * 2.0
  # print(as.logical(torch$all(torch$gt(D, A))$numpy()))
  expect_true_tensor(all(torch$gt(D, A)))
  expect_false_tensor(all(torch$gt(A, D)))
})

test_that("tensor is less than or equal", {
  A <- torch$ones(60000L, 1L, 28L, 28L)
  expect_true_tensor(all(torch$le(A, A)))
  expect_true_tensor(all(A <= A))
})

test_that("tensor is greater than or equal", {
  A <- torch$ones(60000L, 1L, 28L, 28L)
  expect_true_tensor(all(torch$ge(A, A)))
  expect_true_tensor(all(A >= A))
})



context("tensor multiplication")

test_that("dot product", {
  p <- torch$Tensor(list(2, 3))
  q <- torch$Tensor(list(2, 1))
  expect_equal(torch$dot(p, q), torch$tensor(c(7.0)))
  expect_equal((p %.*% q), torch$tensor(c(7.0)))
})


context("Matrix product of two tensors, matmul()")

test_that("matmul: vector * vector", {
  tensor1 = torch$randn(3L)
  tensor2 = torch$randn(3L)
  expect_equal(torch$matmul(tensor1, tensor2)$size(), torch$Size(list()))
})

test_that("matmul: matrix x vector", {
  tensor1 = torch$randn(3L, 4L)
  tensor2 = torch$randn(4L)
  # torch.Size([3])
  expect_equal(torch$matmul(tensor1, tensor2)$size(), torch$Size(list(3L)))
})

test_that("matmul: batched matrix x broadcasted vector", {
  tensor1 = torch$randn(10L, 3L, 4L)
  tensor2 = torch$randn(4L)
  # torch.Size([10, 3])
  expect_equal(torch$matmul(tensor1, tensor2)$size(), torch$Size(list(10L, 3L)))
})


test_that("matmul: batched matrix x batched matrix", {
  tensor1 = torch$randn(10L, 3L, 4L)
  tensor2 = torch$randn(10L, 4L, 5L)
  # torch.Size([10, 3])
  expect_equal(torch$matmul(tensor1, tensor2)$size(), torch$Size(list(10L, 3L, 5L)))
})

test_that("matmul: batched matrix x broadcasted matrix", {
  tensor1 = torch$randn(10L, 3L, 4L)
  tensor2 = torch$randn(4L, 5L)
  # torch.Size([10, 3, 5])
  expect_equal(torch$matmul(tensor1, tensor2)$size(), torch$Size(list(10L, 3L, 5L)))
})
