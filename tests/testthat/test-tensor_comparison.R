source("helper_utils.R")


# torch$all: Returns True if all elements in the tensor are True, False otherwise.
# torch$any: Returns True if any elements in the tensor are True, False otherwise.

context("torch all of ones and zeros")

test_that("check if torch$all of all ones returns TRUE", {
    skip_if_no_torch()

    a <- torch$ones(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch_all <- torch$all(a_uint8)
    a_r_logical <- as.logical(a_torch_all$numpy())
    expect_true(a_r_logical)
})

test_that("check if torch$all of all zeros returns FALSE", {
    skip_if_no_torch()

    a <- torch$zeros(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch_all <- torch$all(a_uint8)
    a_r_logical <- as.logical(a_torch_all$numpy())
    expect_false(a_r_logical)
})



context("torch any of ones and zeros")

test_that("check if torch$any of all ones returns TRUE", {
    skip_if_no_torch()

    a <- torch$ones(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch_all <- torch$any(a_uint8)
    a_r_logical <- as.logical(a_torch_all$numpy())
    expect_true(a_r_logical)
})

test_that("check if torch$any of all zeros returns FALSE", {
    skip_if_no_torch()

    a <- torch$zeros(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch <- torch$any(a_uint8)
    a_r_logical <- as.logical(a_torch$numpy())
    expect_false(a_r_logical)
})


context("torch all/any of 3x3 eye tensor")

test_that("check if torch$any of 3x3 eye tensor returns FALSE", {
    skip_if_no_torch()

    a <- torch$eye(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch <- torch$all(a_uint8)
    a_r_logical <- as.logical(a_torch$numpy())
    expect_false(a_r_logical)
})

test_that("check if torch$all of all zeros returns TRUE", {
    skip_if_no_torch()

    a <- torch$eye(3L, 3L)
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch <- torch$any(a_uint8)
    a_r_logical <- as.logical(a_torch$numpy())
    expect_true(a_r_logical)
})


context("torch$all on dimensions: by rows and by columns")

test_that("torch$all on rows", {
    skip_if_no_torch()

    r1 <- torch$ones(1L, 3L)
    r2 <- torch$zeros(1L, 3L)
    r3 <- torch$ones(1L, 3L)

    a <- torch$cat(list(r1, r2, r3))
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch <- torch$all(a_uint8, dim = 0L)
    a_numpy <- a_torch$numpy()
    a_r_logical <- as.logical(a_numpy)
    expect_equal(a_r_logical, c(FALSE, FALSE, FALSE))
})

test_that("torch$all on columns", {
    skip_if_no_torch()

    r1 <- torch$ones(1L, 3L)
    r2 <- torch$zeros(1L, 3L)
    r3 <- torch$ones(1L, 3L)

    a <- torch$cat(list(r1, r2, r3))
    a_uint8 <- torch$as_tensor(a, dtype = torch$uint8)
    a_torch <- torch$all(a_uint8, dim = 1L)
    a_numpy <- a_torch$numpy()
    a_r_logical <- as.logical(a_numpy)
    expect_equal(a_r_logical, c(TRUE, FALSE, TRUE))
})


context("Lower than, lt <")
skip_if_no_torch()

A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

test_that("C lower than A", {
    cond_torch <- all(torch$lt(C, A))
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})

test_that("C < A", {
    cond_torch <- all(C < A)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})

test_that("A < C", {
    cond_torch <- all(A < C)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})




context("Greater than, gt >")
skip_if_no_torch()

A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

test_that("C greater than A", {
    cond_torch <- all(torch$gt(C, A))
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})

test_that("C > A", {
    cond_torch <- all(C > A)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})

test_that("A > C", {
    cond_torch <- all(A > C)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})




context("Lower than or equal, lt <=")
skip_if_no_torch()

A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

test_that("C lower or equal than A", {
    cond_torch <- all(torch$le(C, A))
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})

test_that("C <= A", {
    cond_torch <- all(C <= A)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})

test_that("A <= C", {
    cond_torch <- all(A <= C)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})




context("Greater or equal than, ge >=")
skip_if_no_torch()

A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

test_that("C greater or equal than A", {
    cond_torch <- all(torch$ge(C, A))
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})

test_that("C >= A", {
    cond_torch <- all(C >= A)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_false(cond_r_logical)
})

test_that("A >= C", {
    cond_torch <- all(A >= C)
    cond_numpy <- cond_torch$numpy()
    cond_r_logical <- as.logical(cond_numpy)
    expect_true(cond_r_logical)
})



context("return of logical comparisons")
skip_if_no_torch()

test_that("one", {
    cond_torch_letters <- all(torch$gt(C, A))
    cond_torch_symbol <- all(C > A)

    expect_output(print(cond_torch_letters$dtype), "torch.uint8")
    expect_output(print(cond_torch_symbol$dtype), "torch.uint8")

    letters_torch <- cond_torch_letters
    symbol_torch <- cond_torch_symbol

    letters_numpy <- letters_torch$numpy()
    letters_r_logical <- as.logical(letters_numpy)

    symbol_numpy <- symbol_torch$numpy()
    symbol_r_logical <- as.logical(symbol_numpy)

    # print(letters_numpy)
    expect_false(letters_r_logical)
    expect_false(symbol_r_logical)


})
