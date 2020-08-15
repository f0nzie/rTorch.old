# using function make_copy() to get rid off PyTorch warnings

source("tensor_functions.R")

skip_on_cran()


context("core PyTorch functions")

context("_select")

test_that("masked_select", {

    torch$manual_seed(42L)
    x <- torch$randn(3L, 4L)
    mask <- x$ge(0.5)
    mselect <- torch$masked_select(x, mask)

    mask_expect_r <- rbind(
        c(0,  0,  0,  0),
        c(0,  0,  1,  0),
        c(0,  0,  1,  1)
    )

    mask_expect <- torch$ByteTensor(make_copy(mask_expect_r))

    mselect_expect <- torch$FloatTensor(c(
                        -0.9105,
                        1.2812,
                        -0.9752))

    expect_equal(mask, mask_expect)
    expect_equal(mselect, mselect_expect)
    expect_equal(x$size(), torch$Size(c(3L, 4L)))
})


test_that("index_select", {
    torch$manual_seed(42L)
    x <- torch$randn(3L, 4L)
    src <- "
     0.3367  0.1288  0.2345  0.2303
    -1.1229 -0.1863  2.2082 -0.6380
     0.4617  0.2674  0.5349  0.8094
    "
    # select indices by rows, dim = 0L
    indices = torch$LongTensor(c(0L, 2L))
    x0 <- torch$index_select(x, 0L, indices)
    src <- "
     0.3367  0.1288  0.2345  0.2303
     0.4617  0.2674  0.5349  0.8094
    "
    mx <- as.matrix(read.table(text = src))
    x0_expect <- torch$FloatTensor(make_copy(mx))
    expect_equal(x0, x0_expect)

    # select indices by columns, dim = 1L
    x1 <- torch$index_select(x, 1L, indices)
    src <- "
     0.3367  0.2345
    -1.1229  2.2082
     0.4617  0.5349
    "
    mx <- as.matrix(read.table(text = src))
    x1_expect <- torch$FloatTensor(make_copy(mx))
    expect_equal(x1, x1_expect)
})


context("cat, split")

test_that("cat", {
    #  concatenates by rows and columns
    # Concatenates the given sequence of seq tensors in the given dimension. All
    # tensors must either have the same shape (except in the cat dimension) or
    # be empty.
    # torch.cat(seq, dim=0, out=None) -> Tensor
    torch$manual_seed(42L)
    x <- torch$randn(2L, 3L)

    # concatenate along rows, dim = 0
    x30 <- torch$cat(c(x, x, x), 0L)
    src <- "
        0.3367  0.1288  0.2345
        0.2303 -1.1229 -0.1863
        0.3367  0.1288  0.2345
        0.2303 -1.1229 -0.1863
        0.3367  0.1288  0.2345
        0.2303 -1.1229 -0.1863
    "

    m_r <- as.matrix(read.table(text = src))
    x30_expect <- torch$FloatTensor(make_copy(m_r))
    expect_equal(x30, x30_expect)

    # concatenate along columns, dim = 1
    vec <- c(x, x, x)
    x31 <- torch$cat(make_copy(vec), 1L)
    src <- "
    0.3367  0.1288  0.2345  0.3367  0.1288  0.2345  0.3367  0.1288  0.2345
    0.2303 -1.1229 -0.1863  0.2303 -1.1229 -0.1863  0.2303 -1.1229 -0.1863
    "
    mx <- as.matrix(read.table(text = src))
    x31_expect <- torch$FloatTensor(make_copy(mx))
    expect_equal(x31, x31_expect)

})


test_that("split", {
    # Splits the tensor into chunks.
    # Last chunk will be smaller if tensor size along dimension is not divisible
    # torch.split(tensor, split_size_or_sections, dim=0)
    #
    #  0.3367  0.1288  0.2345  0.2303
    # -1.1229 -0.1863  2.2082 -0.6380
    #  0.4617  0.2674  0.5349  0.8094

    torch$manual_seed(42L)
    x <- torch$randn(3L, 4L)

    # split by rows. no-divisible
    x0 <- torch$split(x, 2L, dim = 0L)
    src <- "
     0.3367  0.1288  0.2345  0.2303
    -1.1229 -0.1863  2.2082 -0.6380
    "
    mx <- as.matrix(read.table(text = src))
    x01_expect <- torch$FloatTensor(make_copy(mx))
    expect_equal(x0[[1]], x01_expect)
    src <- "
     0.4617  0.2674  0.5349  0.8094
    "
    x02_expect <- torch$FloatTensor(make_copy(as.matrix(read.table(text = src))))
    expect_equal(x0[[2]], x02_expect)

    # split by columns. divisible
    x1 <- torch$split(x, 2L, dim = 1L)
    src <- "
     0.3367  0.1288
    -1.1229 -0.1863
     0.4617  0.2674
    "
    x11_expect <- torch$FloatTensor(make_copy(as.matrix(read.table(text = src))))
    expect_equal(x1[[1]], x11_expect)
    src <- "
    0.2345  0.2303
    2.2082 -0.6380
    0.5349  0.8094
    "
    x12_expect <- torch$FloatTensor(make_copy(as.matrix(read.table(text = src))))
    expect_equal(x1[[2]], x12_expect)
})


context("squeeze, take, narrow")

test_that("squeeze", {
    # removes dimensions of size 1
    x = torch$zeros(2L, 1L, 2L, 1L, 2L)
    expect_equal(x$size(), torch$Size(c(2L, 1L, 2L, 1L, 2L)))

    y = torch$squeeze(x)
    expect_equal(y$size(), torch$Size(c(2L, 2L, 2L)))

    # squeeze specified dimension at index=0. if size not 1, skip
    y0 = torch$squeeze(x, 0L)
    # print(y0$size())
    expect_equal(y0$size(), torch$Size(c(2L, 1L, 2L, 1L, 2L)))

    # squeeze specified dimension at index=1; removed because it is 1
    y1 = torch$squeeze(x, 1L)
    expect_equal(y1$size(), torch$Size(c(2L, 2L, 1L, 2L)))

})

test_that("take", {
    # treats the tensor like a vector and extracts elements in the order
    # specified by the 2nd argument
    src <- torch$Tensor(make_copy(rbind(c(4, 3, 5),
                              c(6, 7, 8))))
    res <- torch$take(src, torch$LongTensor(c(0L, 2L, 5L)))
    expect_equal(res, torch$FloatTensor(c(4, 5, 8)))
})


test_that("narrow", {
    # Returns a new tensor that is a narrowed version of self tensor. The
    # dimension dim is narrowed from start to start + length.

    # narrow # 1
    x_t <- "
    1  2  3
    4  5  6
    7  8  9
    "
    x_ <- as.matrix(read.table(text = x_t))
    x  <- torch$Tensor(make_copy(x_))

    r_t <- "
    1  2  3
    4  5  6
    "
    r <- torch$Tensor(make_copy(as.matrix(read.table(text = r_t))))
    expect_equal(x$narrow(0L, 0L , 2L), r)

    # narrow #2
    r_t <- "
    2  3
    5  6
    8  9
    "
    r <- torch$Tensor(make_copy(as.matrix(read.table(text = r_t))))
    expect_equal(x$narrow(1L, 1L , 2L), r)

})


context("transpose")

test_that("numpy transpose", {
    # two dimensions: 2x2
    x = r_to_py(np$arange(4L))
    x = x$reshape(c(2L,2L))
    t = np$transpose(x)
    expected <- (matrix(c(0,1,2,3), nrow = 2))
    expect_equal(t, expected)

    # two dimensions: 3x3
    x = r_to_py(np$arange(9L))
    x = x$reshape(c(3L, 3L))
    t = np$transpose(x)
    expected <- (matrix(seq(0,8), nrow = 3))
    expect_equal(t, expected)

    # two dimensions: 5x5
    x = r_to_py(np$arange(25L))
    x = x$reshape(c(5L, 5L))
    t = np$transpose(x)
    expected <- (matrix(seq(0,24), nrow = 5))
    expect_equal(t, expected)

    # three dimensions: 1x2x3
    x = np$ones(c(1L, 2L, 3L))
    x = r_to_py(x)
    t <- np$transpose(x, c(1L, 0L, 2L))
    result <- r_to_py(t)$shape
    result <- unlist(py_to_r(result))
    expect_equal(result, c(2, 1, 3))

})

test_that("torch transpose", {
    # two dimensions: 2x2
    x <- torch$arange(4L)
    x <- x$view(c(2L, 2L))
    t = torch$transpose(x, 0L, 1L)
    mat <- matrix(c(0,1,2,3), nrow = 2)
    expected <- torch$as_tensor(r_to_py(mat)$copy())
    expect_equal(t, expected)

    # two dimensions: 3x3
    x <- torch$arange(9L)
    x <- x$view(c(3L, 3L))
    t = torch$transpose(x, 0L, 1L)
    mat <- matrix(seq(0,8), nrow = 3)
    expected <- torch$as_tensor(r_to_py(mat)$copy())
    expect_equal(t, expected)


    # two dimensions: 5x5
    x <- torch$arange(25L)
    x <- x$view(c(5L, 5L))
    t = torch$transpose(x, 0L, 1L)
    mat <- matrix(seq(0, 24), nrow = 5)
    expected <- torch$as_tensor(r_to_py(mat)$copy())
    expect_equal(t, expected)

    # three dimensions: 1x2x3
    x <- torch$ones(c(1L, 2L, 3L))
    t <- torch$transpose(x, 1L, 0L)
    result <- torch$as_tensor(t$shape)
    expected <- torch$tensor(c(2L, 1L, 3L))
    expect_equal(result, expected)

})

context("permute")

test_that("permute in 2D", {
    x <- torch$tensor(list(list(list(1,2)), list(list(3,4)), list(list(5,6))))
    result <- torch$as_tensor(x$shape)
    expected <- torch$tensor(c(3L, 1L, 2L))
    expect_equal(result, expected)  # test original tensor

    permuted <- x$permute(c(1L, 2L, 0L))
    result <- torch$as_tensor(permuted$shape)
    expected <- torch$tensor(c(1L, 2L, 3L))
    expect_equal(result, expected)  # test permuted tensor

})

test_that("permute in 3D", {
    x <- torch$randn(10L, 480L, 640L, 3L)
    result <- torch$as_tensor(x$shape)
    expected <- torch$tensor(c(10L, 480L, 640L, 3L))
    expect_equal(result, expected)   # test original tensor

    p <- x$permute(0L, 3L, 1L, 2L)
    result <- torch$as_tensor(p$size())
    expected <- torch$tensor(c(10L, 3L, 480L, 640L))
    expect_equal(result, expected)  # test permuted tensor
})

