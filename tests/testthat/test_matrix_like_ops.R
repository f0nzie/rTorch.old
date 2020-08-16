library(testthat)


source("utils.R")
skip_on_cran()

# matrix like tensor operations ------------------------------------------------
context("matrix like tensor operations")

test_that("Dot product of 2 tensors", {
  # Dot product of 2 tensors
  # direct operation with torch
  r = torch$dot(torch$Tensor(list(4L, 2L)), torch$Tensor(list(3L, 1L)))
  result <- r$item()
  expect_equal(result, 14)
  # using an R function and list
  r <- tensor_dot(torch$Tensor(list(4L, 2L)), torch$Tensor(list(3L, 1L)))
  result <- r$item()
  expect_equal(result, 14)
  # using an R function and vector
  r <- tensor_dot(torch$Tensor(c(4L, 2L)), torch$Tensor(c(3L, 1L)))
  result <- r$item()
  expect_equal(result, 14)

  r <- tensor_dot(torch$Tensor(c(4, 2)), torch$Tensor(c(3, 1)))
  result <- r$item()
  expect_equal(result, 14)

})

test_that("Cross product", {
  # loginfo("Cross product")
  m1 = torch$ones(3L, 5L)
  m2 = torch$ones(3L, 5L)

  # Cross product
  # Size 3x5
  r = torch$cross(m1, m2)
  expect_equal(tensor_dim(r), c(3, 5))
})


test_that("multiply tensor by scalar", {
  # loginfo("\n Multiply tensor by scalar")
  tensor = torch$ones(4L, dtype=torch$float64)
  scalar = np$float64(4.321)
  # print(torch$scalar_tensor(scalar))
  prod = torch$mul(tensor, torch$scalar_tensor(scalar))
  expect_equal(prod$numpy(), array(c(4.321, 4.321, 4.321, 4.321)), tolerance = 1e-7)
  # print(class(prod$numpy()))
})
