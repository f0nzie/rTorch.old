source("utils.R")

context("generic methods")

test_that("logical operations", {
  skip_if_no_torch()

  a = torch$ByteTensor(list(0, 1, 1, 0))
  b = torch$ByteTensor(list(1, 1, 0, 0))

  # print(a & b)  # logical and
  # tensor([0, 1, 0, 0], dtype=torch.uint8)
  expect_equal((a & b), torch$BoolTensor(list(FALSE, TRUE, FALSE, FALSE)))
})


test_that("base or", {
  skip_if_no_torch()

  base <- exp(1L)
  expect_false(base != exp(1L))
  expect_false(is_tensor(base))
  expect_true(is_tensor(torch$as_tensor(base)))
})


context("log of number with a base")

test_that("log with base", {
  skip_if_no_torch()

  r <- array(as.double(1:5))
  t <- torch$as_tensor(make_copy(r), dtype = torch$float32)
  # print(log(t, base = torch$as_tensor(list(3L))))
  # print(torch$tensor(list(0.0000, 0.6309, 1.0000, 1.2619, 1.4650)))
  expect_equal(log(t, base = torch$as_tensor(list(3L))), torch$tensor(list(0.0000, 0.6309, 1.0000, 1.2619, 1.4650)))

})


test_that("log with supplied base works", {
  skip_if_no_torch()

  r <- array(as.double(1:20))
  t <- torch$as_tensor(make_copy(r), dtype = torch$float32)
  n <- torch$as_tensor(make_copy(-r), dtype = torch$float32)

  expect_near(exp(log(t)), t)
  expect_near(log(torch$as_tensor(make_copy(exp(r)))), t)
  expect_near(torch$as_tensor(make_copy(2 ^ r)), torch$pow(2, torch$as_tensor(make_copy(r))))
  expect_near(log2(torch$as_tensor(make_copy(2 ^ r))), t)

  expect_near(t, log10(torch$as_tensor(make_copy(10 ^ r))))
  expect_near(t, log( exp(t)))
  expect_near(t, log2(  2 ^ t ))
  expect_near(t, log10( 10 ^ t ))

  # log() dispatches correctly without trying to change base
  expect_near(torch$log(t), log(t))
  # TODO: fix this test. error: Typeas_tensor() takes 1 positional argument but 2 were given
  # FIXED: modify "log.torch.Tensor" in generics.R
  expect_near(log(torch$as_tensor(make_copy(r)), base = 3L), log(t, base = 3L))
})
