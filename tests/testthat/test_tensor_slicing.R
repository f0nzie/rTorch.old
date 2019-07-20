library(testthat)


source("tensor_functions.R")


# test slicing with chunk() and select_index() ---------------------------------
context("test slicing with chunk() and select_index()")

test_that("test tensor has 3 dimensions", {
    builtins    <- import_builtins()
    img <- torch$ones(3L, 28L, 28L)
    result <- builtins$list(img$size())
    # print(result)
    expect_equal(result, c(3, 28, 28))
    expect_equal(tensor_dim_(img), 3)
})


# split tensor with function chunk() -------------------------------------------
context("split tensor with function chunk()")

test_that("tensor can be split in chunks", {
    img <- torch$ones(3L, 28L, 28L)

    img_chunks <- torch$chunk(img, chunks = 3L, dim = 0L)
    result <- length(img_chunks)
    # print(result)
    expect_equal(result, 3)

    # loginfo("1st chunk member")
    img_chunk_1 <- img_chunks[[1]]
    result <- img_chunk_1$size()
    result <- tensor_dim(img_chunk_1)
    expect_equal(result, c(1, 28, 28))
    expect_equal(tensor_sum(img_chunk_1), 784)

    # loginfo("2nd chunk member")
    img_chunk_2 <- img_chunks[[2]]
    result <- img_chunk_2$size()
    result <- tensor_dim(img_chunk_2)
    expect_equal(result, c(1, 28, 28))
    expect_equal(tensor_sum(img_chunk_2), 784)

    # loginfo("3rd chunk member")
    img_chunk_3 <- img_chunks[[3]]
    result <- img_chunk_3$size()
    result <- tensor_dim(img_chunk_3)
    expect_equal(result, c(1, 28, 28))
    expect_equal(tensor_sum(img_chunk_3), 784)

    # there is no 4th member because the tensor dim=1 has only three
    expect_error(img_chunks[[4]])
})


# tensor extraction of 3D tensor with index_select() ---------------------------
context("tensor extraction of 3D tensor with index_select()")

test_that("select_index() can also select a tensor layer", {
    img <- torch$ones(3L, 28L, 28L)        # 3D image tensor

    # loginfo("index_select. get layer 1")
    indices = torch$tensor(c(0L))
    img_mod <- torch$index_select(img, dim = 0L, index = indices)
    expect_equal(tensor_dim(img_mod), c(1, 28, 28))
    expect_equal(tensor_sum(img_mod), 784)

    # loginfo("index_select. get layer 2")
    indices = torch$tensor(c(1L))
    img_mod <- torch$index_select(img, dim = 0L, index = indices)
    expect_equal(tensor_dim(img_mod), c(1, 28, 28))
    expect_equal(tensor_sum(img_mod), 784)

    # loginfo("index_select. get layer 3")
    indices = torch$tensor(c(2L))
    img_mod <- torch$index_select(img, dim = 0L, index = indices)
    expect_equal(tensor_dim(img_mod), c(1, 28, 28))
    expect_equal(tensor_sum(img_mod), 784)
})


# tensor extraction of 4D tensor with index_select() ---------------------------
context("tensor extraction of 4D tensor with index_select()")

test_that("get first element in a tensor", {
    # https://discuss.pytorch.org/t/is-there-anyway-to-get-the-first-element-of-a-tensor-as-a-scalar/2097
    img <- torch$ones(60000L, 3L, 28L, 28L)        # 4D image tensor

    # ERROR with new slicing
    # print(tensor_dim(img$data[1,,,]))
    expect_equal(tensor_dim(img$data[1,,,]), c(3, 28, 28)) # 3D tensor
    expect_equal(tensor_dim(img$data[1, 1,,]), c(28, 28)) # 2D tensor
    expect_equal(tensor_dim(img$data[1, 1, 1, ]), c(28))  # 1D tensor
})

test_that("select_index() can also select a tensor layer", {
    # 4D image tensor
    img <- torch$ones(60000L, 3L, 28L, 28L)

    # first block in dim=0
    indices = torch$tensor(c(0L))      # get block 1 on dim=0
    expect_equal(tensor_dim_(img), 4)
    # expect_equal(img[0]$numel(), 60000)
    img_mod <- torch$index_select(img, dim = 0L, index = indices)
    expect_equal(tensor_dim(img_mod), c(1, 3, 28, 28))
    # print(tensor_sum(img_mod))
    expect_equal(tensor_sum(img_mod), 2352)

    # last block in dim=0
    indices = torch$tensor(c(59999L))
    img_mod <- torch$index_select(img, dim = 0L, index = indices)
    expect_equal(tensor_dim(img_mod), c(1, 3, 28, 28))
    expect_equal(tensor_sum(img_mod), 2352)

    # first layer in dim=1
    indices = torch$tensor(c(0L))      # get layer 1 on dim=1
    expect_equal(tensor_dim_(img), 4)
    img_mod <- torch$index_select(img, dim = 1L, index = indices)
    expect_equal(tensor_dim(img_mod), c(60000, 1, 28, 28))
    expect_equal(tensor_sum(img_mod), 60000*28*28)

    # last layer in dim=1
    indices = torch$tensor(c(2L))      # get layer 1 on dim=1
    expect_equal(tensor_dim_(img), 4)
    img_mod <- torch$index_select(img, dim = 1L, index = indices)
    expect_equal(tensor_dim(img_mod), c(60000, 1, 28, 28))
    expect_equal(tensor_sum(img_mod), 60000*28*28)

    # first square in dim=2
    indices = torch$tensor(c(0L))      # get layer 1 on dim=1
    expect_equal(tensor_dim_(img), 4)
    img_mod <- torch$index_select(img, dim = 2L, index = indices)
    expect_equal(tensor_dim(img_mod), c(60000, 3, 1, 28))
    expect_equal(tensor_sum(img_mod), 60000*3*1*28)

    # last square in dim=2
    indices = torch$tensor(c(27L))      # get layer 1 on dim=1
    expect_equal(tensor_dim_(img), 4)
    img_mod <- torch$index_select(img, dim = 2L, index = indices)
    expect_equal(tensor_dim(img_mod), c(60000, 3, 1, 28))
    expect_equal(tensor_sum(img_mod), 60000*3*1*28)
    # overflowing index on dim=2
    indices = torch$tensor(c(28L))     # this will produce an error
    expect_error(torch$index_select(img, dim = 2L, index = indices))

    # last square in dim=3
    indices = torch$tensor(c(27L))
    expect_equal(tensor_dim_(img), 4)
    img_mod <- torch$index_select(img, dim = 3L, index = indices)
    expect_equal(tensor_dim(img_mod), c(60000, 3, 28, 1))
    expect_equal(tensor_sum(img_mod), 60000*3*28*1)
    # overflowing index on dim=2
    indices = torch$tensor(c(28L))     # this will produce an error
    expect_error(torch$index_select(img, dim = 3L, index = indices))
})


# function narrow() extracts part of a tensor ----------------------------------
context("function narrow() extracts part of a tensor")

test_that("narrow on rows to extract part of a tensor", {
    x = torch$tensor(list(list(1, 2, 3), list(4, 5, 6), list(7, 8, 9)))
    expect_equal(x$narrow(0L, 0L, 2L),
                 torch$tensor(list(list(1, 2, 3), list(4, 5, 6)))
                 )
})

test_that("narrow on columns to extract part of a tensor", {
    x = torch$tensor(list(list(1, 2, 3), list(4, 5, 6), list(7, 8, 9)))
    expect_equal(x$narrow(1L, 0L, 2L),
                 torch$tensor(list(list(1, 4, 7), list(2, 5, 8)))
    )
})


