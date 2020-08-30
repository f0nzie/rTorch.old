
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![Travis build
status](https://travis-ci.org/f0nzie/rTorch.svg?branch=master)](https://travis-ci.org/f0nzie/rTorch)
[![AppVeyor build
status](https://ci.appveyor.com/api/projects/status/github/f0nzie/rTorch?branch=master&svg=true)](https://ci.appveyor.com/project/f0nzie/rTorch)
<!-- badges: end -->

# rTorch

The goal of `rTorch` is providing an R wrapper to
[PyTorch](https://pytorch.org/). We have borrowed ideas and code used in
R [tensorflow](https://github.com/rstudio/tensorflow) to implement
`rTorch`.

Besides the module `torch`, which directly provides `PyTorch` methods,
classes and functions, the package also provides `numpy` as a method
called `np`, and `torchvision`, as well. The dollar sign `$` after the
module will provide you access to those objects.

## Installation

### From CRAN

`rTorch` is available in via CRAN and GitHub.

Install from CRAN using `install.packages("rTorch")` from the R console,
or from *RStudio* using `Tools`, `Install Packages` from the menu.

### From GitHub

From GitHub, install `rTorch` with:

`devtools::install_github("f0nzie/rTorch")`

Installing from GitHub gives you the flexibility of experimenting with
the latest development version of `rTorch`. For instance, to install
`rTorch` from the `develop` branch:

`devtools::install_github("f0nzie/rTorch", ref="develop")`

### Installing Python and PyTorch

Although, Python and PyTorch can be installed directly from the R
console, before start running `rTorch`, I would recommend installing a
Python or Python-Anaconda environment and testing it working alright
first. There are `rTorch` has functions that can install PyTorch from
the R console.

If you prefer do it manually, use this example:

1.  Create a conda environment with `conda create -n my-torch python=3.7
    -y`

2.  Activate the new environment with `conda activate my-torch`

3.  Install PyTorch related packages with:

`conda install python=3.6 pytorch torchvision matplotlib pandas -c
pytorch`

> Note: If you you don’t specify a version `conda` will install the
> latest `PyTorch`. As of this writing (August-September 2020), the
> latest PyTorch version is 1.6.

Alternatively, you could create and install a *conda* environment a
specific PyTorch version with:

`conda create -n my-torch python=3.6 pytorch=1.3 torchvision matplotlib
pandas -c pytorch -y`

`conda` will resolve the dependencies and versions of the other packages
automatically, or let you know your options.

**Note.** `matplotlib` and `pandas` are not really necessary, but I was
asked if `matplotlib` or `pandas` would work in PyTorch. Then, I decided
to put them for testing and experimentation. They both work.

## Matrices and Linear Algebra

There are five major type of Tensors in PyTorch: \* Byte \* Float \*
Double \* Long \* Bool

``` r
library(rTorch)

byte_tensor   <- torch$ByteTensor(3L, 3L)
float_tensor  <- torch$FloatTensor(3L, 3L)
double_tensor <- torch$DoubleTensor(3L, 3L)
long_tensor   <- torch$LongTensor(3L, 3L)
bool_tensor   <- torch$BoolTensor(5L, 5L)

byte_tensor  
#> tensor([[ 95, 108, 101],
#>         [ 97, 102,  10],
#>         [ 32,  32,  32]], dtype=torch.uint8)
float_tensor  
#> tensor([[4.8565e+33, 2.0618e-19, 4.7351e+22],
#>         [1.7749e+28, 2.9965e+32, 2.8231e+23],
#>         [1.6953e+25, 7.7783e+31, 8.2996e-33]])
double_tensor 
#> tensor([[4.6734e-310,  0.0000e+00, 4.2440e-314],
#>         [4.6734e-310, 4.6734e-310, 1.4822e-323],
#>         [4.2440e-314, 4.9407e-324,  0.0000e+00]], dtype=torch.float64)
long_tensor   
#> tensor([[             0, 94590233951984,              0],
#>         [             0,              0,              0],
#>         [             0,              0,              0]])
bool_tensor   
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [ True, False, False, False,  True]])
```

A `4D` tensor like in MNIST hand-written digits recognition dataset:

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

A `3D` tensor:

``` r
ft3d <- torch$FloatTensor(4L, 3L, 2L)
ft3d
#> tensor([[[nan, 0.],
#>          [0., 0.],
#>          [nan, 0.]],
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
#>          [nan, 0.],
#>          [nan, 0.]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(nan)
```

``` r
# create a tensor with a value
torch$full(list(2L, 3L), 3.141592)
#> tensor([[3.1416, 3.1416, 3.1416],
#>         [3.1416, 3.1416, 3.1416]])
```

## Basic Tensor Operations

### Add tensors

``` r
# 3x5 matrix uniformly distributed between 0 and 1
mat0 <- torch$FloatTensor(3L, 5L)$uniform_(0L, 1L)

# fill a 3x5 matrix with 0.1
mat1 <- torch$FloatTensor(3L, 5L)$uniform_(0.1, 0.1)

# a vector with all ones
mat2 <- torch$FloatTensor(5L)$uniform_(1, 1)
```

``` r
# add two tensors
mat0 + mat1
#> tensor([[1.0114, 0.3326, 0.8534, 0.1205, 0.2666],
#>         [0.3495, 0.7643, 0.5192, 0.7974, 0.7843],
#>         [0.9583, 0.8032, 0.6909, 0.6786, 0.6683]])
```

``` r
# add three tensors
mat0 + mat1 + mat2
#> tensor([[2.0114, 1.3326, 1.8534, 1.1205, 1.2666],
#>         [1.3495, 1.7643, 1.5192, 1.7974, 1.7843],
#>         [1.9583, 1.8032, 1.6909, 1.6786, 1.6683]])
```

``` r
# PyTorch add two tensors using add() function
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.2212, 0.6688, 1.4701, 0.8517],
#>         [0.9130, 0.6675, 0.7644, 1.0153],
#>         [0.7147, 1.1099, 1.3089, 1.1761],
#>         [1.5271, 0.5811, 0.9276, 0.5468],
#>         [0.6809, 0.4957, 1.1375, 1.6721]])
print(x + y)
#> tensor([[1.2212, 0.6688, 1.4701, 0.8517],
#>         [0.9130, 0.6675, 0.7644, 1.0153],
#>         [0.7147, 1.1099, 1.3089, 1.1761],
#>         [1.5271, 0.5811, 0.9276, 0.5468],
#>         [0.6809, 0.4957, 1.1375, 1.6721]])
```

### Add tensor element to another tensor

``` r
# add an element of tensor to a tensor
mat1[1, 1] + mat2
#> tensor([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
```

> The expression `tensor.index(m)` is equivalent to `tensor[m]`.

### Add a scalar to a tensor

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[1.0114, 0.3326, 0.8534, 0.1205, 0.2666],
#>         [0.3495, 0.7643, 0.5192, 0.7974, 0.7843],
#>         [0.9583, 0.8032, 0.6909, 0.6786, 0.6683]])
```

### Multiply tensor by scalar

``` r
# Multiply tensor by scalar
tensor = torch$ones(4L, dtype=torch$float64)
scalar = np$float64(4.321)
print(scalar)
#> [1] 4.321
print(torch$scalar_tensor(scalar))
#> tensor(4.3210)
(prod = torch$mul(tensor, torch$scalar_tensor(scalar)))
#> tensor([4.3210, 4.3210, 4.3210, 4.3210], dtype=torch.float64)
```

``` r
# short version using generics
(prod = tensor * scalar)
#> tensor([4.3210, 4.3210, 4.3210, 4.3210], dtype=torch.float64)
```

## NumPy and PyTorch

`numpy` has been made available as a module in `rTorch`. We can call
functions from `numpy` refrerring to it as `np$_a_function`. Examples:

``` r
# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]       [,2]       [,3]      [,4]        [,5]
#> [1,] 0.6956638 0.22032669 0.89292283 0.9992851 0.559075415
#> [2,] 0.1491096 0.96521941 0.07834506 0.7266076 0.005110647
#> [3,] 0.4414605 0.07670523 0.91312116 0.7413845 0.612391253
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
#> [1,] 0.3367274 0.3367274 0.3367274 0.3367274 0.3367274 0.3367274 0.3367274
#> [2,] 0.1924392 0.1924392 0.1924392 0.1924392 0.1924392 0.1924392 0.1924392
#> [3,] 0.2785063 0.2785063 0.2785063 0.2785063 0.2785063 0.2785063 0.2785063
#>           [,8]      [,9]     [,10]
#> [1,] 0.3367274 0.3367274 0.3367274
#> [2,] 0.1924392 0.1924392 0.1924392
#> [3,] 0.2785063 0.2785063 0.2785063
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

## With newer PyTorch versions we should work with NumPy array copies

There have been minor changes in the latest version of PyTorch that
prevent a direct use of a NumPy array. You will get this warning:

    sys:1: UserWarning: The given NumPy array is not writeable, and PyTorch does 
    not support non-writeable tensors. This means you can write to the underlying
    (supposedly non-writeable) NumPy array using the tensor. You may want to copy 
    the array to protect its data or make it writeable before converting it to a 
    tensor. This type of warning will be suppressed for the rest of this program.

For instance, this code will produce the warning:

``` r
# as_tensor. Modifying tensor modifies numpy object as well
a = np$array(list(1, 2, 3))
t = torch$as_tensor(a)
print(t)

torch$tensor(list( 1,  2,  3))
t[1L]$fill_(-1)
print(a)
```

while this other one -with some extra code- will not:

``` r
a = np$array(list(1, 2, 3))
a_copy = r_to_py(a)$copy()             # we make a copy of the numpy array first

t = torch$as_tensor(a_copy)
print(t)
#> tensor([1., 2., 3.], dtype=torch.float64)

torch$tensor(list( 1,  2,  3))
#> tensor([1., 2., 3.])
t[1L]$fill_(-1)
#> tensor(-1., dtype=torch.float64)
print(a)
#> [1] 1 2 3
```

## Create tensors

``` r
# a random 1D tensor
np_arr <- np$random$rand(5L)
ft1 <- torch$FloatTensor(r_to_py(np_arr)$copy())    # make a copy of numpy array
ft1
#> tensor([0.0538, 0.9541, 0.2209, 0.5765, 0.0162])
```

``` r
# tensor as a float of 64-bits
np_copy <- r_to_py(np$random$rand(5L))$copy()       # make a copy of numpy array
ft2 <- torch$as_tensor(np_copy, dtype= torch$float64)
ft2
#> tensor([0.7705, 0.7934, 0.4773, 0.3574, 0.9290], dtype=torch.float64)
```

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.7705, 0.7935, 0.4773, 0.3574, 0.9292], dtype=torch.float16)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[7.0065e-45,        nan,        nan,        nan,        nan,        nan,
#>                 nan],
#>         [2.3822e-44,        nan, 8.4078e-45,        nan,        nan,        nan,
#>                 nan],
#>         [4.2039e-45,        nan,        nan,        nan,        nan,        nan,
#>                 nan],
#>         [2.1019e-44,        nan,        nan, 1.4013e-44,        nan, 3.2230e-44,
#>          1.2612e-44],
#>         [5.6052e-45, 1.6816e-44, 9.8091e-45, 3.5032e-44, 3.7835e-44, 4.3440e-44,
#>                 nan]])
```

``` r
# using arange to create tensor. starts from 0
v = torch$arange(9L)
(v = v$view(3L, 3L))
#> tensor([[0, 1, 2],
#>         [3, 4, 5],
#>         [6, 7, 8]])
```

## Distributions

Initialize a tensor randomized with a normal distribution with mean=0,
var=1:

``` r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[ 0.4737,  0.7715,  0.9471, -1.8549,  0.4789,  0.0449,  1.4284],
#>         [ 0.3292,  0.5692,  1.1076, -2.0157,  0.7210, -0.0093, -0.3312],
#>         [-0.1380,  0.1812,  0.2463, -1.5742, -0.1053, -0.7697,  1.9284],
#>         [ 0.6771,  0.8697, -0.2000, -1.4217,  0.7356,  0.6454, -0.7652],
#>         [ 0.7970,  1.0912, -0.4489, -0.8858,  0.6262,  0.4520, -0.3133]])
print(a$size())
#> torch.Size([5, 7])
```

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
#> tensor([[0.4378, 0.5468, 0.6614, 0.4305, 0.4551],
#>         [0.9417, 0.3102, 0.4765, 0.4853, 0.1902],
#>         [0.7802, 0.8512, 0.8183, 0.3596, 0.6716]])
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

### Binomial distribution

``` r
Binomial <- torch$distributions$binomial$Binomial

m = Binomial(100, torch$tensor(list(0 , .2, .8, 1)))
(x = m$sample())
#> tensor([  0.,  21.,  79., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[4., 5.],
#>         [5., 8.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([0.4981])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([1.1475])
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

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]      [,2]      [,3]      [,4]
#> [1,] 0.23579969 0.4523109 0.9423700 0.2726818
#> [2,] 0.25595743 0.7565628 0.1146501 0.0970648
#> [3,] 0.03392114 0.9432601 0.2841513 0.1108227
#> [4,] 0.56212726 0.4620988 0.1675381 0.3381970
#> [5,] 0.21887906 0.4831720 0.3071367 0.3036845
```

``` r
# convert a numpy array to a tensor
np_a = np$array(c(c(3, 4), c(3, 6)))
t_a = torch$from_numpy(r_to_py(np_a)$copy())
print(t_a)
#> tensor([3., 4., 3., 6.])
```

## Tensor resizing

``` r
x = torch$randn(2L, 3L)            # Size 2x3
y = x$view(6L)                    # Resize x to size 6
z = x$view(-1L, 2L)                # Size 3x2
print(y)
#> tensor([-0.3592, -0.8903, -1.8535,  0.1179, -1.7773,  0.1235])
print(z)
#> tensor([[-0.3592, -0.8903],
#>         [-1.8535,  0.1179],
#>         [-1.7773,  0.1235]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[-0.1110, -1.0733, -1.2695],
#>         [ 1.1514,  1.2974, -0.8074]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[-0.1110, -1.0733, -1.2695],
#>         [ 1.1514,  1.2974, -0.8074],
#>         [-0.1110, -1.0733, -1.2695],
#>         [ 1.1514,  1.2974, -0.8074],
#>         [-0.1110, -1.0733, -1.2695],
#>         [ 1.1514,  1.2974, -0.8074]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[-0.1110, -1.0733, -1.2695, -0.1110, -1.0733, -1.2695, -0.1110, -1.0733,
#>          -1.2695],
#>         [ 1.1514,  1.2974, -0.8074,  1.1514,  1.2974, -0.8074,  1.1514,  1.2974,
#>          -0.8074]])
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
v = torch$ones_like(eye)     # A tensor with same shape as eye. Fill it with 1.
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

### Inplace / Out-of-place

``` r
a$fill_(3.5)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000]])
# a has now been filled with the value 3.5

# add a scalar to a tensor
b <- a$add(4.0)

# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5

print(a)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000]])
print(b)
#> tensor([[7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000]])
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
#> tensor(3.5000)
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
#> tensor([[ 0.4188, -1.3779, -1.2275,  1.5040],
#>         [ 1.5311, -1.2907, -0.1234,  0.7252],
#>         [ 0.1620,  2.3497, -0.1018,  1.6483]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 0.4188, -1.3779, -1.2275,  1.5040],
#>         [ 0.1620,  2.3497, -0.1018,  1.6483]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[ 0.4188, -1.2275],
#>         [ 1.5311, -0.1234],
#>         [ 0.1620, -0.1018]])
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

## Transpose

``` r
# two dimensions: 3x3
x <- torch$arange(9L)
x <- x$view(c(3L, 3L))
t <- torch$transpose(x, 0L, 1L)

x   # "Original tensor"
#> tensor([[0, 1, 2],
#>         [3, 4, 5],
#>         [6, 7, 8]])

t    # "Transposed"
#> tensor([[0, 3, 6],
#>         [1, 4, 7],
#>         [2, 5, 8]])
```

``` r
# three dimensions: 1x2x3
x <- torch$ones(c(1L, 2L, 3L))
t <- torch$transpose(x, 1L, 0L)

print(x)     # original tensor
#> tensor([[[1., 1., 1.],
#>          [1., 1., 1.]]])

print(t)     # transposed
#> tensor([[[1., 1., 1.]],
#> 
#>         [[1., 1., 1.]]])


print(x$shape)    # original tensor
#> torch.Size([1, 2, 3])
print(t$shape)    # transposed
#> torch.Size([2, 1, 3])
```

## Permutation

### permute a 2D tensor

``` r
x   <- torch$tensor(list(list(list(1,2)), list(list(3,4)), list(list(5,6))))
xs  <- torch$as_tensor(x$shape)
xp  <- x$permute(c(1L, 2L, 0L))
xps <- torch$as_tensor(xp$shape)

print(x)     # original tensor
#> tensor([[[1., 2.]],
#> 
#>         [[3., 4.]],
#> 
#>         [[5., 6.]]])

print(xp)    # permuted tensor
#> tensor([[[1., 3., 5.],
#>          [2., 4., 6.]]])

print(xs)     # shape original tensor
#> tensor([3, 1, 2])

print(xps)    # shape permuted tensor
#> tensor([1, 2, 3])
```

### permute a 3D tensor

``` r
x <- torch$randn(10L, 480L, 640L, 3L)
xs <- torch$as_tensor(x$size())     # torch$tensor(c(10L, 480L, 640L, 3L))
xp <- x$permute(0L, 3L, 1L, 2L)     # specify dimensions order
xps <- torch$as_tensor(xp$size())   # torch$tensor(c(10L, 3L, 480L, 640L))

print(xs)      # original tensor size
#> tensor([ 10, 480, 640,   3])

print(xps)     # permuted tensor size
#> tensor([ 10,   3, 480, 640])
```

## Logical operations

``` r
m0 = torch$zeros(3L, 5L)
m1 = torch$ones(3L, 5L)
m2 = torch$eye(3L, 5L)

print(m1 == m0)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

``` r
print(m1 != m1)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

``` r
print(m2 == m2)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

``` r
# AND
m1 & m1
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

``` r
# OR
m0 | m2
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False]])
```

``` r
# OR
m1 | m2
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

``` r
# all_boolean <- function(x) {
#   # convert tensor of 1s and 0s to a unique boolean
#   as.logical(torch$all(x)$numpy())
# }

# tensor is less than
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

# is C < A
all(torch$lt(C, A))
#> tensor(1, dtype=torch.uint8)
all(C < A)
#> tensor(1, dtype=torch.uint8)
# is A < C
all(A < C)
#> tensor(0, dtype=torch.uint8)
```

``` r
# tensor is greater than
A <- torch$ones(60000L, 1L, 28L, 28L)
D <- A * 2.0
all(torch$gt(D, A))
#> tensor(1, dtype=torch.uint8)
all(torch$gt(A, D))
#> tensor(0, dtype=torch.uint8)
```

``` r
# tensor is less than or equal
A1 <- torch$ones(60000L, 1L, 28L, 28L)
all(torch$le(A1, A1))
#> tensor(1, dtype=torch.uint8)
all(A1 <= A1)
#> tensor(1, dtype=torch.uint8)

# tensor is greater than or equal
A0 <- torch$zeros(60000L, 1L, 28L, 28L)
all(torch$ge(A0, A0))
#> tensor(1, dtype=torch.uint8)
all(A0 >= A0)
#> tensor(1, dtype=torch.uint8)

all(A1 >= A0)
#> tensor(1, dtype=torch.uint8)
all(A1 <= A0)
#> tensor(0, dtype=torch.uint8)
```

### Logical NOT

``` r
all_true <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
all_true
#> tensor([True, True, True, True])

# logical NOT
not_all_true <- !all_true
not_all_true
#> tensor([False, False, False, False])
```

``` r
diag <- torch$eye(5L)
diag
#> tensor([[1., 0., 0., 0., 0.],
#>         [0., 1., 0., 0., 0.],
#>         [0., 0., 1., 0., 0.],
#>         [0., 0., 0., 1., 0.],
#>         [0., 0., 0., 0., 1.]])

# logical NOT
not_diag <- !diag

# convert to integer
not_diag$to(dtype=torch$uint8)
#> tensor([[0, 1, 1, 1, 1],
#>         [1, 0, 1, 1, 1],
#>         [1, 1, 0, 1, 1],
#>         [1, 1, 1, 0, 1],
#>         [1, 1, 1, 1, 0]], dtype=torch.uint8)
```
