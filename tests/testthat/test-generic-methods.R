skip_if_no_torch()

source("tensor_functions.R")
source("utils.R")


context("generic methods")

as_tensor <- function(...) torch$as_tensor(...)

expect_near <- function(..., tol = 1e-5) expect_equal(..., tolerance = tol)

test_that("logical operations", {
  a = torch$ByteTensor(list(0, 1, 1, 0))
  b = torch$ByteTensor(list(1, 1, 0, 0))

  # print(a & b)  # logical and
  # tensor([0, 1, 0, 0], dtype=torch.uint8)
  expect_equal((a & b), torch$BoolTensor(list(FALSE, TRUE, FALSE, FALSE)))
})



test_that("base or", {
  base <- exp(1L)
  expect_false(base != exp(1L))
  expect_false(is_tensor(base))
  expect_true(is_tensor(as_tensor(base)))

  r <- array(as.double(1:5))
  t <- torch$as_tensor(r, dtype = torch$float32)
  # print(log(t, base = 3L))
})

test_that("log with supplied base works", {

  skip_if_no_torch()

  r <- array(as.double(1:20))
  t <- as_tensor(r, dtype = torch$float32)
  n <- as_tensor(-r, dtype = torch$float32)



  expect_near(exp(log(t)), t)
#
#   # print(torch$log(t))
#   # print(exp(r))
#   # print(t)
#   # print(log(as_tensor(exp(r))))
  expect_near(log(as_tensor(exp(r))), t)
  expect_near(as_tensor(2 ^ r), torch$pow(2, as_tensor(r)))

  expect_near(log2(as_tensor(2 ^ r)), t)

  expect_near(t, log10(torch$as_tensor(10 ^ r)) )

  expect_near(t, log( exp(t)))
  expect_near(t, log2(  2 ^ t ))
  expect_near(t, log10( 10 ^ t ))

  # # log() dispatches correctly without trying to change base
  expect_near(torch$log(t), log(t))
#   # print(torch$log(t))
#   # print(log(t))
#   #
# TODO: fix this test
  # Typeas_tensor() takes 1 positional argument but 2 were given
# expect_near(log(as_tensor(r), base = 3L), log(t, base = 3L))
#   print(log(torch$as_tensor(r), base = 3L))
#
})
