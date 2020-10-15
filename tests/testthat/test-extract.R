source("helper_utils.R")

context("extract syntax")

as_array <- function(x) {
  if (x$numel() > 1) x$numpy()
  else as.vector(x$numpy())
}


null_out_all_extract_opts <- function() {
  opts <- options()
  opts[grepl("^torch[.]extract", names(opts))] <- list(NULL)
  options(opts)
}

arr <- function (...) {
  # create an array with the specified dimensions, and fill it with consecutive
  # increasing integers
  dims <- unlist(list(...))
  array(1:prod(dims), dim = dims)
}

randn <- function (...) {
  dim <- c(...)
  array(rnorm(prod(dim)), dim = dim)
}

# check a simple (one-object) expression produces the same result when done on
# an R array, and when done on a tensor, with results ported back to R
# e.g. check_expr(a[1:3], swap = "a")
check_expr <- function (expr, name = "x") {

  call <- substitute(expr)
  r_out <- as.array(eval(expr))

  # swap the array for a constant, run, and convert back to an array
  obj <- get(name, parent.frame())
  swapsies <- list(tor$constant(obj))
  names(swapsies) <- name
  tf_out <- with(swapsies, as_array(eval(call)))

  # check it's very very similar
  expect_identical(r_out, tf_out)

}

reset_warnings <- function() {
  e <- rTorch:::warned_about
  e$negative_indices <- FALSE
  e$tensors_passed_asis <- FALSE
}



test_that("scalar indexing works", {
  skip_if_no_torch()
  oopt <- options(torch.extract.style = "R")
  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)
  x3_ <- arr(3, 3, 3)

  x1_py <- r_to_py(x1_)
  x1 <- torch$from_numpy(x1_py$copy())          # copy numpy array to fix warning

  expect_true(class(x1_) == "array")
  expect_true(class(x1_py)[1] == "numpy.ndarray")

  # expect_all_true is custom made in utils.R
  expect_all_true(class(x1) %in%
                c("torch.Tensor", "torch._C._TensorBase", "python.builtin.object"))

  x2_py <- r_to_py(x2_)
  x3_py <- r_to_py(x3_)
  # cast to Tensors
  x2 <- torch$from_numpy(x2_py$copy())                    # fix warning
  x3 <- torch$from_numpy(x3_py$copy())                    # fix warning


  # extract as arrays
  y1_ <- x1_[1]
  y2_ <- x2_[1, 2]
  y3_ <- x3_[1, 2, 3]
  expect_true(class(y2_) == "integer")
  # print(y2_)
  # print(y3_)

  # extract as Tensors
  y1 <- x1[1]
  y2 <- x2[1, 2]
  y3 <- x3[1, 2, 3]
  # print(y1)
  # print(y2)
  # print(y3)

  # # they should be equivalent
  expect_equal(y1_, torch$scalar_tensor(y1)$item())
  expect_equal(y2_, torch$scalar_tensor(y2)$item())
  expect_equal(y3_, torch$scalar_tensor(y3)$item())
  # expect_equal(y1_, y1)
  # expect_equal(y2_, y2)
  # expect_equal(y3_, y3)
  #
  # options(oopt)
})



test_that("vector indexing works", {
  skip_if_no_torch()

  oopt <- options(torch.extract.one_based = FALSE)
  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)

  # cast to Tensors
  x1 <- torch$as_tensor(r_to_py(x1_)$copy())                    # fix warning
  x2 <- torch$as_tensor(r_to_py(x2_)$copy())                    # fix warning

  # extract as arrays
  y1_ <- x1_[2:3]
  y2_ <- x2_[2:3, 1]
  # print(class(y2_))

  # extract as Tensors
  y1 <- x1[1:2]
  y2 <- x2[1:2, 0]
  # print(class(y2))

  # these should be equivalent (need to coerce R version back to arrays)
  expect_equal(y1_, y1$numpy())
  expect_equal(y2_, as.vector(y2$numpy()))
  # expect_equal(y2_, y2$numpy())
  # expect_equal(y1_, y1)
  # expect_equal(array(y2_), y2)

  options(oopt)
})


test_that("blank indices retain all elements", {
  skip_if_no_torch()

  oopt <- options(torch.extract.one_based = FALSE)

  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)
  x3_ <- arr(3, 3, 3)
  x4_ <- arr(3, 3, 3, 3)

  # cast to Tensors
  x1 <- torch$as_tensor(r_to_py(x1_)$copy())                    # fix warning
  x2 <- torch$as_tensor(r_to_py(x2_)$copy())                    # fix warning
  x3 <- torch$as_tensor(r_to_py(x3_)$copy())                    # fix warning
  x4 <- torch$as_tensor(r_to_py(x4_)$copy())                    # fix warning

  # extract as arrays
  y1_ <- x1_[]
  y2_a <- x2_[2:3, ]
  y2_b <- x2_[, 1:2]
  y3_a <- x3_[2:3, 1, ]
  y3_b <- x3_[2:3, , 1]
  y4_ <- x4_[2:3, 1, , 2:3]

  # print(y1_)

  # extract as Tensors
  y1 <- x1[]
  y2a <- x2[1:2, ]  # j missing
  y2b <- x2[, 0:1]
  y3a <- x3[1:2, 0, ]
  y3b <- x3[1:2, , 0]
  y4 <- x4[1:2, 0, , 1:2]



  # print(class(y1$numpy()))
  # print(as_array(y1))

  # # these should be equivalent
  expect_equal(y1_, as_array(y1))
  expect_equal(y2_a, as_array(y2a))
  expect_equal(y2_b, as_array(y2b))  #
  expect_equal(y3_a, as_array(y3a))
  expect_equal(y3_b, as_array(y3b))  #
  expect_equal(y4_, as_array(y4))

  options(oopt)
})

test_that("indexing works within functions", {
  skip_if_no_torch()

  # tensorflow.extract.style = "python",
  oopt <- options(torch.extract.one_based = FALSE)

  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)
  x3_ <- arr(3, 3, 3)

  # cast to Tensors
  x1 <- torch$as_tensor(r_to_py(x1_)$copy())
  x2 <- torch$as_tensor(r_to_py(x2_)$copy())
  x3 <- torch$as_tensor(r_to_py(x3_)$copy())

  # set up functions
  sub1 <- function (x, a)
    x[a - 1]
  sub2 <- function (x, a, b)
    x[a - 1, b - 1]
  sub3 <- function (x, b, c)
    x[, b - 1, c - 1]  # skip first element

  # extract as arrays
  y1_ <- x1_[1:3]
  y2_ <- x2_[, 1:2]
  y3_a <- x3_[, 1:2, ]
  y3_b <- x3_[, , 1]

  # print(y1_)

  # extract as Tensors
  y1 <- sub1(x1, 1:3)
  y2 <- sub2(x2, 1:3, 1:2)
  y3a <- sub3(x3, 1:2, 1:3)
  y3b <- sub3(x3, 1:3, 1)

  # print(y1)

  # these should be equivalent
  expect_equal(y1_, as_array(y1))
  expect_equal(y2_, as_array(y2))
  expect_equal(y3_a, as_array(y3a))
  expect_equal(y3_b, as_array(y3b))

  options(oopt)
})

test_that("indexing works with variables", {
  skip_if_no_torch()

  expect_ok <- function (expr) {
    #expect_is(expr, "torch.python.framework.ops.Tensor")
    expect_is(expr, "torch.Tensor")
  }

  # set up tensors
  x1 <- torch$as_tensor(r_to_py(arr(3))$copy())
  x2 <- torch$as_tensor(r_to_py(arr(3, 3))$copy())
  x3 <- torch$as_tensor(r_to_py(arr(3, 3, 3))$copy())

  # extract with index (these shouldn't error)
  index <- 2
  expect_ok(x1[index])  # i
  expect_ok(x2[, index])  # j
  expect_ok(x3[, , index])  # dots
  # print(class(x1[index]))
  # print(x2[, index])
})


test_that("indexing with negative sequences errors", {
  skip_if_no_torch()

  oopt <- options(torch.extract.style = "R")

  # set up Tensors
  x1 <- torch$as_tensor(r_to_py(arr(3))$copy())
  x2 <- torch$as_tensor(r_to_py(arr(3, 3))$copy())

  # extract with negative indices (where : is not the top level call)
  expect_error(x1[-(1:2)], 'positive')
  expect_error(x2[-(1:2), ], 'positive')

  options(oopt)
})


test_that("incorrect number of indices errors", {
  skip_if_no_torch()

  # set up Tensor
  x <- torch$as_tensor(r_to_py(arr(3, 3, 3))$copy())
  # options(tensorflow.extract.one_based = TRUE)
  # too many
  expect_error(x[1:2, 2, 1:2, 3],
               'Incorrect number of dimensions')
  expect_error(x[1:2, 2, 1:2, 3, , ],
               'Incorrect number of dimensions')
  expect_error(x[1:2, 2, 1:2, 3, , drop = TRUE],
               'Incorrect number of dimensions')
  # too few
  expect_warning(x[],
                 'Incorrect number of dimensions')
  expect_warning(x[1:2, ],
                 'Incorrect number of dimensions')
  expect_warning(x[1:2, 2],
                 'Incorrect number of dimensions')

})

test_that("silly indices error", {
  skip_if_no_torch()

  # set up Tensor
  x <- torch$as_tensor(r_to_py(arr(3, 3, 3))$copy())

  # these should all error and notify the user of the failing index
  expect_error(x[1:2, NA, 2], 'NA')
  expect_error(x[1:2, Inf, 2], 'Inf')
  expect_error(x[1:2, 'apple', 2], 'character')
  expect_error(x[1:2, mean, 2], 'function')
})



test_that("passing non-vector indices errors", {
  skip_if_no_torch()

  # set up Tensor
  x1 <- torch$as_tensor(r_to_py(arr(3, 3))$copy())
  x2 <- torch$as_tensor(r_to_py(arr(3, 3, 3))$copy())

  # block indices
  block_idx_1 <- rbind(c(1, 2), c(0, 1))
  block_idx_2 <- rbind(c(1, 2, 1), c(0, 1, 2))

  # indexing with matrices should fail
  expect_error(x1[block_idx_1],
               'not currently supported')
  expect_error(x2[block_idx_2],
               'not currently supported')
})



test_that("dim(), length(), nrow(), and ncol() work on tensors", {
  skip_if_no_torch()
  a_matrix <- matrix(rnorm(100), ncol = 2)
  a_tensor <- torch$as_tensor(r_to_py(a_matrix)$copy())

  expect_equal(dim(a_matrix), dim(a_tensor))
  expect_equal(length(a_matrix), length(a_tensor))
  expect_equal(nrow(a_matrix), nrow(a_tensor))
  expect_equal(ncol(a_matrix), ncol(a_tensor))
})



test_that("all_dims()", {
  skip_if_no_torch()
  x1.r <- arr(3)
  x2.r <- arr(3, 3)
  x3.r <- arr(3, 3, 3)
  x4.r <- arr(3, 3, 3, 3)

  x1.t <- torch$as_tensor(r_to_py(x1.r)$copy())
  x2.t <- torch$as_tensor(r_to_py(x2.r)$copy())
  x3.t <- torch$as_tensor(r_to_py(x3.r)$copy())
  x4.t <- torch$as_tensor(r_to_py(x4.r)$copy())

  expect_equal(as_array(x1.t[all_dims()]), x1.r[])
  # print(x1.t[all_dims()])
  # print(x1.r[])

  options(tensorflow.extract.one_based = TRUE)
  # TODO: review the following statement:
  # as.array() because torch returns 1d arrays, not bare atomic vectors
  expect_equal(as_array(x2.t[all_dims()]), as.array( x2.r[,]  ))
  expect_equal(as_array(x2.t[1, all_dims()]), as.array( x2.r[1,] ))
  expect_equal(as_array( x2.t[ all_dims(), 1] ), as.array( x2.r[,1] ))

  expect_equal(as_array( x3.t[all_dims()]       ), as.array( x3.r[,,]   ))
  expect_equal(as_array( x3.t[1, all_dims()]    ), as.array( x3.r[1,,]  ))
  expect_equal(as_array( x3.t[1, 1, all_dims()] ), as.array( x3.r[1,1,] ))
  expect_equal(as_array( x3.t[1, all_dims(), 1] ), as.array( x3.r[1,,1] ))
  expect_equal(as_array( x3.t[all_dims(), 1]    ), as.array( x3.r[,,1]  ))
  expect_equal(as_array( x3.t[all_dims(), 1, 1] ), as.array( x3.r[,1,1] ))

  expect_equal(as_array( x4.t[all_dims()]       ), as.array( x4.r[,,,]   ))
  expect_equal(as_array( x4.t[1, all_dims()]    ), as.array( x4.r[1,,,]  ))
  expect_equal(as_array( x4.t[1, 1, all_dims()] ), as.array( x4.r[1,1,,] ))
  expect_equal(as_array( x4.t[1, all_dims(), 1] ), as.array( x4.r[1,,,1] ))
  expect_equal(as_array( x4.t[all_dims(), 1]    ), as.array( x4.r[,,,1]  ))
  expect_equal(as_array( x4.t[all_dims(), 1, 1] ), as.array( x4.r[,,1,1] ))

})




test_that("negative-integers work python style", {

  skip_if_no_torch()
  options(torch.extract.warn_negatives_pythonic = FALSE)
  # options(tensorflow.warn_negative_extract_is_python_style = FALSE)

  x1.r <- arr(4)
  x2.r <- arr(4, 4)

  x1.t <- torch$as_tensor(r_to_py(x1.r)$copy())
  x2.t <- torch$as_tensor(r_to_py(x2.r)$copy())

  options(torch.extract.one_based = TRUE)
  # expect_equal(as_array( x1.t[-1] ),     x1.r[4]    )
  # print(x1.t[-1])
  # print(x1.r[4])
  expect_equal(as_array( x1.t[-2] ),     x1.r[3]    )
  expect_equal(as_array( x2.t[-2, -2] ), x2.r[3, 3] )
  expect_equal(as_array( x2.t[-1, ] ), as.array( x2.r[4,] ))

  options(tensorflow.extract.one_based = FALSE)
  # same as above
  expect_equal(as_array( x1.t[-1] ),     x1.r[4]    )
  expect_equal(as_array( x1.t[-2] ),     x1.r[3]    )

  expect_equal(as_array( x1.t[NULL:-2] ), x1.r[1:3] )
  expect_equal(as_array( x1.t[NULL:-1] ), x1.r[] )

  expect_equal(as_array( x2.t[-2, -2] ), x2.r[3, 3] )
  expect_equal(as_array( x2.t[-1, ] ), as.array( x2.r[4,] ))

  null_out_all_extract_opts()
})


context("switch extract style R to Python")

test_that("python-style strided slice", {
  skip_if_no_torch()
  oopts <- options()
  options(torch.extract.warn_negatives_pythonic = FALSE)

  x.r <- arr(20, 2) # 2nd dim to keep R from dropping (since tf always returns 1d array)
  x.t <- torch$as_tensor(r_to_py(x.r)$copy())            # fix overwrite warning

  options(torch.extract.style = "R")

  expect_equal(as_array( x.t[ `5:`          ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[ `5:NULL`      ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[  5:NULL       ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[ `5:NULL:`     ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[  5:NULL:NULL  ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[ `5:NULL:NULL` ,] ), x.r[ 5:20,])

  expect_equal(as_array( x.t[ `5::` ,] ), x.r[ 5:20,])
  expect_equal(as_array( x.t[ `:5:` ,] ), x.r[ 1:5,])
  expect_equal(as_array( x.t[ `:5`  ,] ), x.r[ 1:5,])
  expect_equal(as_array( x.t[ `2:5` ,] ), x.r[ 2:5,])
  expect_equal(as_array( x.t[ 2:5   ,] ), x.r[ 2:5,])

  expect_equal(as_array( x.t[ `::2` ,]       ), x.r[ seq.int(1, 20, by = 2) ,])
  expect_equal(as_array( x.t[ NULL:NULL:2 ,] ), x.r[ seq.int(1, 20, by = 2) ,])

  # non syntantic names or function calls can work too
  `_idx` <- 1
  expect_equal(as_array( x.t[ `_idx`:(identity(5)+1L),]), x.r[ 1:6, ] )


  expect_equal(as_array( x.t[ `2:6:2`,]), x.r[ seq.int(2, 6, 2) ,])
  expect_equal(as_array( x.t[  2:6:2 ,]), x.r[ seq.int(2, 6, 2) ,])

  # TODO: these two test give error
  # decreasing indexes work
  # expect_equal(as_array( x.t[ `6:2:-2`,]), x.r[ seq.int(6, 2, -2) ,]) # Valuestep must be greater than zero
  # expect_equal(as_array( x.t[  6:2:-2 ,]), x.r[ seq.int(6, 2, -2) ,]) # Valuestep must be greater than zero

  # TODO: error Valuenegative step not yet supported
  # sign of step gets automatically inverted on decreasing indexes
  # expect_equal(as_array( x.t[ `6:2:2` ,]), x.r[ seq.int(6, 2, -2) ,]) # Valuestep must be greater than zero
  # expect_equal(as_array( x.t[  6:2:2  ,]), x.r[ seq.int(6, 2, -2) ,]) # same error for the rest
  # expect_equal(as_array( x.t[  6:2    ,]),   x.r[ 6:2 ,])
  # expect_equal(as_array( x.t[  6:2:1 ,]),   x.r[ 6:2  ,])
  # expect_equal(as_array( x.t[  6:2:-1 ,]),   x.r[ 6:2 ,])


  options(torch.extract.style = "python")
  # options set to match python
  # helper function to actually test in python
  test_in_python <- (function() {
    # main <- reticulate::import_main()
    reticulate::py_run_string(paste(
      "import numpy as np",
      "x = np.array(range(1, 41))",
      "x.shape = (2, 20)",
      "x = x.transpose()", sep = "\n"))
    function(chr) {
      reticulate::py_eval(chr)
    }
  })()


  expect_equal(as_array( x.t[ 2:5,] ), test_in_python("x[2:5,]"))
  # print(x.t[ 2:5,])
  # print(test_in_python("x[2:5,]"))
  expect_equal(as_array( x.t[ 2:-5 ,] ), test_in_python("x[ 2:-5 ,]"))
  # expect_equal(as_array( x.t[ 2:5:2 ,] ), test_in_python("x[ 2:5:2 ,]"))

  # TODO: Valuenegative step not yet supported
  # expect_equal(as_array( x.t[ -2:-5:-1 ,] ), test_in_python("x[ -2:-5:-1 ,]"))
  # expect_equal(as_array( x.t[ 5:2:-1 ,] ), test_in_python("x[ 5:2:-1 ,]"))
  # expect_equal(as_array( x.t[ 5:2:-2 ,] ), test_in_python("x[ 5:2:-2 ,]"))


  # indexing with tensors
  expect_equal(as_array( x.t[torch$as_tensor(2L),] ), as.array(x.r[3,]))
  expect_equal(as_array( x.t[torch$as_tensor(2L):torch$as_tensor(5L),] ), x.r[3:5,])

  # expect warning that no translation on tensors performed
  null_out_all_extract_opts()
  # expect_warning(as_array( x.t[torch$as_tensor(2L),] ), "ignored")
  expect_equal(as_array( x.t[torch$as_tensor(2L),]), array(c(3,23)))

  # warn only once
  expect_silent(as_array( x.t[torch$as_tensor(2L),] ))

  # warn in slice syntax too
  reset_warnings()
  null_out_all_extract_opts()
  # TODO:  did not produce any warnings.
  # expect_warning(as_array( x.t[torch$as_tensor(2L):torch$as_tensor(5L),] ), "ignored")
  expect_equal(x.t[torch$as_tensor(2L):torch$as_tensor(5L),],
  torch$tensor(list(
               list(3, 23),
               list(4, 24),
               list(5, 25)), dtype=torch$int32))

  reset_warnings()
  options(torch.extract.warn_tensors_passed_asis = FALSE)
  expect_silent(as_array( x.t[torch$as_tensor(2L):torch$as_tensor(5L),] ))

  null_out_all_extract_opts()
})

