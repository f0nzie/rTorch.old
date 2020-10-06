library(testthat)
source("utils.R")

skip_if_no_torch()

# read, write by index ---------------------------------------------------------
context("read, write tensors by index")

test_that("Read an element in tensor at index position", {
  # loginfo("replace an element at position 0, 0")
  new_tensor = torch$Tensor(list(list(1, 2), list(3, 4)))
  # print(new_tensor)
  # print(new_tensor[1, 1]$item())
  expect_equal(new_tensor[1, 1]$item(), 1)
  expect_equal(new_tensor[1, 2]$item(), 2)
  expect_equal(new_tensor[2, 1]$item(), 3)
  expect_equal(new_tensor[2, 2]$item(), 4)
})


test_that("improper way of assign a value to a tensor", {
  new_tensor = torch$Tensor(list(list(1, 2), list(3, 4)))
  expect_error(new_tensor[0] <- 5)
  expect_error(new_tensor[1, 1] <- 5)
  expect_error(new_tensor[0, 0] <- 5)
  expect_error(new_tensor[c(0, 0)] <- 5)
  expect_error(new_tensor[c(0L, 0L)] <- 5)
  expect_error(new_tensor[list(0L, 0L)] <- 5)
})

test_that("Proper way to assign a value to a tensor", {
  new_tensor = torch$Tensor(list(list(1, 2), list(3, 4)))
  new_tensor[1, 1]$fill_(5)
  new_tensor[1, 2]$fill_(6)
  new_tensor[2, 1]$fill_(7)
  new_tensor[2, 2]$fill_(8)
  expect_equal(new_tensor[1, 1]$item(), 5)
  expect_equal(new_tensor[1, 2]$item(), 6)
  expect_equal(new_tensor[2, 1]$item(), 7)
  expect_equal(new_tensor[2, 2]$item(), 8)
})

test_that("Write to tensor using Python, no global", {
  new_tensor <- torch$Tensor(list(list(1, 2), list(3, 4)))
  expect_error(py_run_string("r.new_tensor[0,0] = 7"))
  expect_equal(new_tensor[1, 1]$item(), 1)
})

test_that("Write to tensor using Python, declare global", {
  new_tensor <<- torch$Tensor(list(list(1, 2), list(3, 4)))
  py_run_string("r.new_tensor[0,0] = 7")
  expect_equal(new_tensor[1, 1]$item(), 7)
})
