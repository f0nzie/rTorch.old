library(testthat)

source("helper_utils.R")

# skip_if_no_python()
skip_on_cran()

context("basic tests")

test_that("Python integer is R integer", {
    builtins    <- import_builtins()
    num_epochs <- 1L
    expect_equal(builtins$int(num_epochs), 1L)
    expect_equal(builtins$int(num_epochs), 1)
    expect_equal(as.integer(num_epochs), 1)
})


test_that("Python integers can be compared", {
    builtins    <- import_builtins()
    a <- builtins$int(1)
    b <- builtins$int(2)
    expect_true(a <= 1)
    expect_true(a <= 1L)
    expect_true(b >= 2)
    expect_true(b >= 2L)
})


test_that("Dictionary items can be get / set / removed with py_item APIs", {
    d <- dict()
    one <- r_to_py(1)
    py_set_item(d, "apple", one)
    expect_equal(py_get_item(d, "apple"), one)
    py_del_item(d, "apple")
    expect_error(py_get_item(d, "apple"))
    expect_identical(py_get_item(d, "apple", silent = TRUE), NULL)
})


test_that("Multi-dimensional arrays are handled correctly", {
    a1 <- array(c(1:8), dim = c(2,2,2))
    # print(a1)
    expect_equal(-a1, np$negative(a1))
    na1 <- np$negative(a1)

    expect_equal(na1[1], -1)
    expect_equal(na1[2,2,2], -8)
    # print(na1[2,2,2])
})

test_that("shape of numpy array return in a list", {
    b1 <- np$array(list(c(1L:30L), c(1L:30L)))
    expect_equal(c(np$shape(b1)[[1]], np$shape(b1)[[2]]), c(2, 30))

})

test_that("indices in numpy array have to be integers", {
    z0 <- np$zeros(c(2L, 3L, 2L))
    expect_equal(dim(z0), c(2,3,2))
    expect_error(np$zeros(c(2, 3, 2)))
})


# This test associated and affected by global variables in another script

test_that("Python string of commands returns a dictionary", {
    # ensure other Python variables have been cleared
    # indent Python code to the left margin
    #
    # we can make prs a global variable with <<- but it will catch other objects
    prs <- py_run_string("
import numpy as np
a = np.zeros((100, 100, 3))
a[:,:,0] = 255
")
    two_objects <- c("python.builtin.dict", "python.builtin.object")
    five_objects <- c("a", "np", "r", "R", "sys")
    twelve_objects <- c("__name__", "__doc__", "__package__", "__loader__",
                        "__spec__", "__annotations__", "__builtins__", "sys",
                        "R", "r", "np", "a")
    forty_objects <- c("__contains__", "__delattr__", "__delitem__", "__dir__",
                       "__doc__", "__eq__", "__format__", "__ge__",
                       "__getattribute__", "__getitem__", "__gt__",
                       "__hash__", "__init__", "__init_subclass__", "__iter__",
                       "__le__", "__len__", "__lt__", "__ne__", "__new__",
                       "__reduce__", "__reduce_ex__", "__repr__", "__setattr__",
                       "__setitem__", "__sizeof__", "__str__", "__subclasshook__",
                       "clear", "copy", "fromkeys", "get", "items", "keys",
                       "pop", "popitem", "setdefault", "update", "values")

    expect_equal(class(prs), two_objects)
    expect_true(all(class(prs) %in% two_objects))
    expect_equal(length(names(prs)), 5-1)            # before conversion
    expect_equal(length(names(py_to_r(prs))), 12-1)  # after conversion

    expect_true(all(names(prs) %in% five_objects))
    expect_true(all(names(py_to_r(prs) %in% twelve_objects)))
    expect_true(all(names(reticulate::py_list_attributes(prs) %in% forty_objects)))

    expect_s3_class(prs$keys(), 'python.builtin.dict_keys')
    # check dimensions
    expect_equal(dim(prs$a), c(100, 100, 3))
    expect_equal(dim(prs['a']), c(100, 100, 3))
    # check we get zeros in the 3rd dim slice 2 and 3
    expect_equal(prs$a[1,1,2], 0)
    expect_equal(prs$a[1,1,3], 0)
    expect_equal(prs$a[2,2,3], 0)
    expect_equal(prs$a[3,3,3], 0)
    expect_equal(prs$a[100,100,2], 0)
    expect_equal(prs$a[100,100,3], 0)
    # check we get 255 in the 3rd dim slice 1
    expect_equal(prs$a[1,1,1], 255)
    expect_equal(prs$a[2,2,1], 255)
    expect_equal(prs$a[3,3,1], 255)
    expect_equal(prs$a[3,2,1], 255)
    expect_equal(prs$a[2,3,1], 255)
    expect_equal(prs$a[100,100,1], 255)
})


