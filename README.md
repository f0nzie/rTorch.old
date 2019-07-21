
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rTorch

The goal of rTorch is providing an R wrapper to
\[PyTorch\](<https://pytorch.org/>. We have borrowed ideas and code used
in R [tensorflow](https://github.com/rstudio/tensorflow) to implement
`rTorch`.

Besides the module `torch`, which provides `PyTorch` methods, classes
and functions, we are also providing numpy as a method `np`, and
`torchvision` as well. The dollar sign after the module will provide you
access to their objects.

## Installation

`rTorch` is available in GitHub only at this moment.

Install rTorch with: `devtools::install_github("f0nzie/rTorch")`

Before start running `rTorch`, install a Python Anaconda environment
first.

1.  Create a conda environment with `conda create -n myenv python=3.7`

2.  Activate the new environment with `conda activate myenv`

3.  Install PyTorch related packages with:

`conda install python=3.6.6 pytorch-cpu torchvision-cpu matplotlib
pandas -c pytorch`

Now, you can load `rTorch` in R or RStudio.

The automatic installation, like in `rtensorflow`, may be available
later.

## Matrices and Linear Algebra

There are five major type of Tensors in PyTorch

``` r
library(rTorch)

bt <- torch$ByteTensor(3L, 3L)
ft <- torch$FloatTensor(3L, 3L)
dt <- torch$DoubleTensor(3L, 3L)
lt <- torch$LongTensor(3L, 3L)
Bt <- torch$BoolTensor(5L, 5L)

ft
#> tensor([[1.2456e-11, 1.0779e-08, 1.4585e-19],
#>         [2.5348e-09, 2.6033e-12, 4.0058e-11],
#>         [2.6257e-06, 2.5193e-09, 2.5930e-09]])
dt
#> tensor([[6.9466e-310, 6.9466e-310, 1.4706e-296],
#>         [ 5.6872e-13, 1.4043e-309, 2.4844e-296],
#>         [4.8524e-273,  6.8754e+11, 2.4509e-296]], dtype=torch.float64)
Bt
#> tensor([[ True, False, False, False, False],
#>         [False, False, False,  True,  True],
#>         [ True,  True,  True,  True, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```

A 4D tensor like in MNIST hand-written digits recognition dataset:

``` r
mnist_4d <- torch$FloatTensor(60000L, 3L, 28L, 28L)

# size
mnist_4d$size()
#> torch.Size([60000, 3, 28, 28])

# length
length(mnist_4d)
#> [1] 141120000

# shape, like in numpy
mnist_4d$shape
#> torch.Size([60000, 3, 28, 28])

# number of elements
mnist_4d$numel()
#> [1] 141120000
```

A 3D tensor:

``` r
ft3d <- torch$FloatTensor(4L, 3L, 2L)
ft3d
#> tensor([[[0., 0.],
#>          [0., 0.],
#>          [0., 0.]],
#> 
#>         [[0., 0.],
#>          [0., 0.],
#>          [0., 0.]],
#> 
#>         [[0., 0.],
#>          [0., 0.],
#>          [0., 0.]],
#> 
#>         [[0., 0.],
#>          [0., 0.],
#>          [0., 0.]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(0.)
```

``` r
bt
#> tensor([[ 99, 111, 110],
#>         [116,  97, 105],
#>         [110, 105, 110]], dtype=torch.uint8)
# [torch.ByteTensor of size 3x3]
```

``` r
ft
#> tensor([[1.2456e-11, 1.0779e-08, 1.4585e-19],
#>         [2.5348e-09, 2.6033e-12, 4.0058e-11],
#>         [2.6257e-06, 2.5193e-09, 2.5930e-09]])
# [torch.FloatTensor of size 3x3]
```

``` r
# create a tensor with a value
torch$full(list(2L, 3L), 3.141592)
#> tensor([[3.1416, 3.1416, 3.1416],
#>         [3.1416, 3.1416, 3.1416]])
```

## Basic Tensor Operations

### Uniform matrix

``` r
library(rTorch)

# 3x5 matrix uniformly distributed between 0 and 1
mat0 <- torch$FloatTensor(3L, 5L)$uniform_(0L, 1L)

# fill a 3x5 matrix with 0.1
mat1 <- torch$FloatTensor(3L, 5L)$uniform_(0.1, 0.1)

# a vector with all ones
mat2 <- torch$FloatTensor(5L)$uniform_(1, 1)

mat0
#> tensor([[0.3646, 0.1149, 0.0186, 0.1285, 0.1759],
#>         [0.5926, 0.3924, 0.4522, 0.8252, 0.0833],
#>         [0.7572, 0.4890, 0.2568, 0.1539, 0.8257]])
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[0.4646, 0.2149, 0.1186, 0.2285, 0.2759],
#>         [0.6926, 0.4924, 0.5522, 0.9252, 0.1833],
#>         [0.8572, 0.5890, 0.3568, 0.2539, 0.9257]])
```

> The expression `tensor.index(m)` is equivalent to `tensor[m]`.

``` r
# add an element of tensor to a tensor
mat1[1, 1] + mat2
#> tensor([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
```

``` r
# add two tensors
mat1 + mat0
#> tensor([[0.4646, 0.2149, 0.1186, 0.2285, 0.2759],
#>         [0.6926, 0.4924, 0.5522, 0.9252, 0.1833],
#>         [0.8572, 0.5890, 0.3568, 0.2539, 0.9257]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[0.6879, 1.1870, 0.9821, 0.3622],
#>         [1.2884, 1.0730, 0.6290, 1.2204],
#>         [0.3553, 1.6722, 1.6823, 0.6222],
#>         [1.2933, 1.1620, 1.8817, 1.0638],
#>         [1.8549, 0.4704, 1.0517, 0.5450]])
print(x + y)
#> tensor([[0.6879, 1.1870, 0.9821, 0.3622],
#>         [1.2884, 1.0730, 0.6290, 1.2204],
#>         [0.3553, 1.6722, 1.6823, 0.6222],
#>         [1.2933, 1.1620, 1.8817, 1.0638],
#>         [1.8549, 0.4704, 1.0517, 0.5450]])
```

## NumPy and PyTorch

`numpy` has been made available as a module in `rTorch`. We can call
functions from `numpy` refrerring to it as `np$_a_function`. Examples:

``` r
# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.8134918 0.1794718 0.9986543 0.6661731 0.1580702
#> [2,] 0.7945867 0.1378126 0.8414559 0.0172430 0.3302996
#> [3,] 0.8564783 0.8156700 0.1301964 0.7678505 0.3488201
```

``` r
# numpy arrays of zeros
syn1 <- np$zeros(c(5L, 10L))
syn1
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    0    0    0    0    0    0    0    0    0     0
#> [2,]    0    0    0    0    0    0    0    0    0     0
#> [3,]    0    0    0    0    0    0    0    0    0     0
#> [4,]    0    0    0    0    0    0    0    0    0     0
#> [5,]    0    0    0    0    0    0    0    0    0     0
```

``` r
# add a scalar to a numpy array
syn1 = syn1 + 0.1
syn1
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [2,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [3,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [4,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [5,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
```

``` r
# in numpy a multidimensional array needs to be defined with a tuple
# in R we do it with a vector
l1 <- np$ones(c(5L, 5L))
l1
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    1    1    1    1
#> [2,]    1    1    1    1    1
#> [3,]    1    1    1    1    1
#> [4,]    1    1    1    1    1
#> [5,]    1    1    1    1    1
```

``` r
# vector-matrix multiplication
np$dot(syn0, syn1)
#>           [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
#> [1,] 0.2815861 0.2815861 0.2815861 0.2815861 0.2815861 0.2815861 0.2815861
#> [2,] 0.2121398 0.2121398 0.2121398 0.2121398 0.2121398 0.2121398 0.2121398
#> [3,] 0.2919015 0.2919015 0.2919015 0.2919015 0.2919015 0.2919015 0.2919015
#>           [,8]      [,9]     [,10]
#> [1,] 0.2815861 0.2815861 0.2815861
#> [2,] 0.2121398 0.2121398 0.2121398
#> [3,] 0.2919015 0.2919015 0.2919015
```

``` r
# build a numpy array from three R vectors
X <- np$array(rbind(c(1,2,3), c(4,5,6), c(7,8,9)))
X
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
#> [3,]    7    8    9
```

``` r
# transpose the array
np$transpose(X)
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
#> [2,]    2    5    8
#> [3,]    3    6    9
```

``` r
# as_tensor. Modifying tensor modifies numpy object as well
a = np$array(list(1, 2, 3))
t = torch$as_tensor(a)
print(t)
#> tensor([1., 2., 3.], dtype=torch.float64)

torch$tensor(list( 1,  2,  3))
#> tensor([1., 2., 3.])
t[1L]$fill_(-1)
#> tensor(-1., dtype=torch.float64)
print(a)
#> [1] -1  2  3
```

## Create tensors

``` r
# a random 1D tensor
ft1 <- torch$FloatTensor(np$random$rand(5L))
ft1
#> tensor([0.9864, 0.2828, 0.4524, 0.0970, 0.1566])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.2310, 0.5811, 0.9835, 0.4856, 0.5834], dtype=torch.float64)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 7.5472e+02,  4.5873e-41, -1.1091e-05,  3.0823e-41,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -5.5023e-05,  3.0823e-41,
#>          -5.5023e-05,  3.0823e-41],
#>         [ 0.0000e+00,  0.0000e+00, -4.1762e-06,  3.0823e-41, -4.1762e-06,
#>           3.0823e-41, -4.1763e-06],
#>         [ 3.0823e-41,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           2.8026e-45,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.1763e-06,
#>           3.0823e-41, -4.1763e-06]])
```

Initialize a tensor randomized with a normal distribution with mean=0,
var=1:

``` r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[-0.3761, -1.2147,  0.2504,  0.3677, -0.9134, -0.3468,  0.6346],
#>         [ 1.0701, -0.2813,  1.3289,  1.1079,  0.4144,  0.0041,  0.1783],
#>         [-0.7237, -0.9832, -0.6981, -0.6932, -0.2432, -0.1765, -0.6008],
#>         [-0.0474, -2.1465, -2.3475,  0.0399,  0.5026, -0.4936,  2.9485],
#>         [-0.1864, -0.3741, -0.3795, -0.2777,  0.1697, -0.1951,  0.6729]])
print(a$size())
#> torch.Size([5, 7])
```

**Binomial distribution**

``` r
Binomial <- torch$distributions$binomial$Binomial

m = Binomial(100, torch$tensor(list(0 , .2, .8, 1)))
(x = m$sample())
#> tensor([  0.,  22.,  82., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[2., 4.],
#>         [6., 6.]])
```

**Exponential distribution**

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([1.6325])
```

## Tensor data types

``` r
# Default data type
torch$tensor(list(1.2, 3))$dtype  # default for floating point is torch.float32
#> torch.float32
```

``` r
# change default data type to float64
torch$set_default_dtype(torch$float64)
torch$tensor(list(1.2, 3))$dtype         # a new floating point tensor
#> torch.float64
```

## Inplace / Out-of-place

``` r
a$fill_(3.5)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000]],
#>        dtype=torch.float32)
# a has now been filled with the value 3.5

# add a scalar to a tensor
b <- a$add(4.0)

# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5

print(a)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000]],
#>        dtype=torch.float32)
print(b)
#> tensor([[7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000]],
#>        dtype=torch.float32)
```

``` r
# this will throw an error because we don't still have a function for assignment
a[1, 1] <- 7.7
print(a)
# Error in a[1, 1] <- 7.7 : object of type 'environment' is not subsettable
```

Some operations like`narrow` do not have in-place versions, and hence,
`.narrow_` does not exist. Similarly, some operations like `fill_` do
not have an out-of-place version, so `.fill` does not exist.

``` r
# a[[0L, 3L]]
a[1, 4]
#> tensor(3.5000, dtype=torch.float32)
```

``` r
x <- torch$ones(5L, 5L)
print(x)
#> tensor([[1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.]])
```

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]      [,2]      [,3]      [,4]
#> [1,] 0.76938084 0.8158050 0.5349001 0.3508918
#> [2,] 0.58291696 0.9298291 0.2427363 0.6125396
#> [3,] 0.83426901 0.2194458 0.1380399 0.5342753
#> [4,] 0.09678209 0.6761211 0.2564105 0.9233270
#> [5,] 0.60133614 0.3870983 0.5166955 0.5922905
```

``` r
# convert a numpy array to a tensor
np_a = np$array(c(c(3, 4), c(3, 6)))
t_a = torch$from_numpy(np_a)
print(t_a)
#> tensor([3., 4., 3., 6.])
```

## Tensor resizing

``` r
x = torch$randn(2L, 3L)            # Size 2x3
y = x$view(6L)                    # Resize x to size 6
z = x$view(-1L, 2L)                # Size 3x2
print(y)
#> tensor([-1.0349, -0.5885,  0.8776,  0.4848,  1.1738,  1.6462])
print(z)
#> tensor([[-1.0349, -0.5885],
#>         [ 0.8776,  0.4848],
#>         [ 1.1738,  1.6462]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[-0.9414, -0.7279, -1.4132],
#>         [-1.8480, -0.4937,  0.7674]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[-0.9414, -0.7279, -1.4132],
#>         [-1.8480, -0.4937,  0.7674],
#>         [-0.9414, -0.7279, -1.4132],
#>         [-1.8480, -0.4937,  0.7674],
#>         [-0.9414, -0.7279, -1.4132],
#>         [-1.8480, -0.4937,  0.7674]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[-0.9414, -0.7279, -1.4132, -0.9414, -0.7279, -1.4132, -0.9414, -0.7279,
#>          -1.4132],
#>         [-1.8480, -0.4937,  0.7674, -1.8480, -0.4937,  0.7674, -1.8480, -0.4937,
#>           0.7674]])
```

``` r
# 0 1 2
# 3 4 5
# 6 7 8
v = torch$arange(9L)
(v = v$view(3L, 3L))
#> tensor([[0, 1, 2],
#>         [3, 4, 5],
#>         [6, 7, 8]])
```

### Reshape tensors

``` r
# ----- Reshape tensors -----
img <- torch$ones(3L, 28L, 28L)
print(img$size())
#> torch.Size([3, 28, 28])

img_chunks <- torch$chunk(img, chunks = 3L, dim = 0L)
print(length(img_chunks))
#> [1] 3

# 1st chunk member
img_chunk_1 <- img_chunks[[1]]
print(img_chunk_1$size())
#> torch.Size([1, 28, 28])
print(img_chunk_1$sum())
#> tensor(784.)

# 2nd chunk member
img_chunk_1 <- img_chunks[[2]]
print(img_chunk_1$size())
#> torch.Size([1, 28, 28])
print(img_chunk_1$sum())
#> tensor(784.)


# index_select. get layer 1
indices = torch$tensor(c(0L))
img2 <- torch$index_select(img, dim = 0L, index = indices)
print(img2$size())
#> torch.Size([1, 28, 28])
print(img2$sum())
#> tensor(784.)

# index_select. get layer 2
indices = torch$tensor(c(1L))
img2 <- torch$index_select(img, dim = 0L, index = indices)
print(img2$size())
#> torch.Size([1, 28, 28])
print(img2$sum())
#> tensor(784.)

# index_select. get layer 3
indices = torch$tensor(c(2L))
img2 <- torch$index_select(img, dim = 0L, index = indices)
print(img2$size())
#> torch.Size([1, 28, 28])
print(img2$sum())
#> tensor(784.)
```

## Special tensors

### Identity matrix

``` r
# identity matrix
eye = torch$eye(3L)              # Create an identity 3x3 tensor
print(eye)
#> tensor([[1., 0., 0.],
#>         [0., 1., 0.],
#>         [0., 0., 1.]])
```

### Ones

``` r
(v = torch$ones(10L))              # A tensor of size 10 containing all ones
#> tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
(v = torch$ones(2L, 1L, 2L, 1L))      # Size 2x1x2x1
#> tensor([[[[1.],
#>           [1.]]],
#> 
#> 
#>         [[[1.],
#>           [1.]]]])
```

``` r
v = torch$ones_like(eye)        # A tensor with same shape as eye. Fill it with 1.
v
#> tensor([[1., 1., 1.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
```

### Zeros

``` r
(z = torch$zeros(10L))             # A tensor of size 10 containing all zeros
#> tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

## Tensor fill

``` r
(v = torch$ones(3L, 3L))
#> tensor([[1., 1., 1.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
v[1L, ]$fill_(2L)         # fill row 1 with 2s
#> tensor([2., 2., 2.])
v[2L, ]$fill_(3L)         # fill row 2 with 3s
#> tensor([3., 3., 3.])
print(v)
#> tensor([[2., 2., 2.],
#>         [3., 3., 3.],
#>         [1., 1., 1.]])
```

``` r
# Initialize Tensor with a range of value
v = torch$arange(10L)             # similar to range(5) but creating a Tensor
(v = torch$arange(0L, 10L, step = 1L))  # Size 5. Similar to range(0, 5, 1)
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### Initialize a linear or log scale Tensor

``` r
# Initialize a linear or log scale Tensor

# Create a Tensor with 10 linear points for (1, 10) inclusively
(v = torch$linspace(1L, 10L, steps = 10L)) 
#> tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

# Size 5: 1.0e-10 1.0e-05 1.0e+00, 1.0e+05, 1.0e+10
(v = torch$logspace(start=-10L, end = 10L, steps = 5L)) 
#> tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
```

## Access to tensor elements

``` r
# replace an element at position 0, 0
(new_tensor = torch$Tensor(list(list(1, 2), list(3, 4))))
#> tensor([[1., 2.],
#>         [3., 4.]])

print(new_tensor[1L, 1L])
#> tensor(1.)
new_tensor[1L, 1L]$fill_(5)
#> tensor(5.)
print(new_tensor)   # tensor([[ 5.,  2.],[ 3.,  4.]])
#> tensor([[5., 2.],
#>         [3., 4.]])
```

``` r
# access an element at position 1, 0
print(new_tensor[2L, 1L])           # tensor([ 3.])
#> tensor(3.)
print(new_tensor[2L, 1L]$item())    # 3.
#> [1] 3
```

``` r
# Select indices
x = torch$randn(3L, 4L)
print(x)
#> tensor([[-1.3521, -0.8520,  0.0200,  0.0108],
#>         [-0.6645,  1.4388,  1.0765, -0.6990],
#>         [ 0.1909,  0.5000,  0.5379, -0.8054]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[-1.3521, -0.8520,  0.0200,  0.0108],
#>         [ 0.1909,  0.5000,  0.5379, -0.8054]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[-1.3521,  0.0200],
#>         [-0.6645,  1.0765],
#>         [ 0.1909,  0.5379]])
```

``` r
# Take by indices
src = torch$tensor(list(list(4, 3, 5),
                        list(6, 7, 8)) )
print(src)
#> tensor([[4., 3., 5.],
#>         [6., 7., 8.]])
print( torch$take(src, torch$tensor(list(0L, 2L, 5L))) )
#> tensor([4., 5., 8.])
```

## Tensor operations

### cross product

``` r
m1 = torch$ones(3L, 5L)
m2 = torch$ones(3L, 5L)
v1 = torch$ones(3L)
# Cross product
# Size 3x5
(r = torch$cross(m1, m2))
#> tensor([[0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.]])
```

### Dot product

``` r
# Dot product of 2 tensors
# Dot product of 2 tensors

p <- torch$Tensor(list(4L, 2L))
q <- torch$Tensor(list(3L, 1L))                   

(r = torch$dot(p, q)) # 14
#> tensor(14.)
(r <- p %.*% q)
#> tensor(14.)
```
