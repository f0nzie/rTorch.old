library(testthat)


source("tensor_functions.R")

# function narrow() extracts part of a tensor ----------------------------------
context("extract parts of a Python object")

test_that("extract a slice from a list", {
    py_run_string("li = [0, 1, 2, 3, 4, 5]")
    sl <- builtins$slice(1L, 3L)
    py_li <- r_to_py(py$li)
    expect_equal(py_li$`__getitem__`(sl), r_to_py(py_eval("[1, 2]")))
    expect_equal(py$li[c(1, 3)], c(0, 2))
    expect_equal(py$li[c(2, 3)], c(1, 2))
    py_run_string("del li")  # remove variable from Python environment
})

test_that("tensor dimension is 4D: 60000x3x28x28", {
    img <<- torch$ones(60000L, 3L, 28L, 28L)
    expect_equal(tensor_dim(img), c(60000, 3, 28, 28))
    expect_equal(tensor_dim_(img), 4)
    # print(img[10:11][0:1])
    expect_equal(tensor_dim(py_eval("r.img[0:10]")), c(10, 3, 28, 28))
    expect_equal(tensor_dim(img[0:9]), c(10, 3, 28, 28))

    expect_equal(tensor_dim(py_eval("r.img[0:10, 0:1]")), c(10, 1, 28, 28))
    expect_error(tensor_dim(img[0:10, 0:1]))
    expect_error(tensor_dim(img[c(list(0:10), list(0:1))]))


    slice1D <- py_eval("[x for x in range(100)]")
    slice2D <- py_eval("[x for x in range(0)]")
    # print(tensor_dim(img[10:10][1]))
    # print(tensor_dim(img[10:10][0:0]))
    # print(tensor_dim(img[10:10][0:1]))
    # print(tensor_dim(img[10:10][1:1]))
})



