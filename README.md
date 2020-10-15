
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![Travis build
status](https://travis-ci.org/f0nzie/rTorch.svg?branch=master)](https://travis-ci.org/f0nzie/rTorch)
[![AppVeyor build
status](https://ci.appveyor.com/api/projects/status/github/f0nzie/rTorch?branch=master&svg=true)](https://ci.appveyor.com/project/f0nzie/rTorch)
<!-- badges: end -->

# rTorch

The goal of `rTorch` is providing an R wrapper to
[PyTorch](https://pytorch.org/). `rTorch` provides all the functionality
of PyTorch plus all the features that R provides. We have borrowed some
ideas and code used in R
[tensorflow](https://github.com/rstudio/tensorflow) to implement
`rTorch`.

Besides the module `torch`, which directly provides `PyTorch` methods,
classes and functions, the package also provides the modules `numpy` as
a method called `np`, and `torchvision`, as well. The dollar sign `$`
after the module will provide you access to all their sub-objects.
Example:

``` r
tv <- rTorch::torchvision
tv
#> Module(torchvision)

np <- rTorch::np
np
#> Module(numpy)

torch_module <- rTorch::torch
torch_module
#> Module(torch)
```

## rTorch Examples

To lighten up the time in building this `rTorch` package, we moved the
examples that use *tensor operations* and *neural networks* to separate
repositories. There are two sets of examples:

  - *The rTorch Minimal Book* at
    <https://f0nzie.github.io/rtorch-minimal-book/>
  - *rTorch Advanced Examples* at
    <https://github.com/f0nzie/rTorch.examples>

## rTorch installation

`rTorch` is available via CRAN and from GitHub.

### From CRAN

Install from CRAN using `install.packages("rTorch")` from the R console,
or from *RStudio* using `Tools`, `Install Packages` from the menu.

### From GitHub

For the latest version install from GitHub. Install `rTorch` with:

`devtools::install_github("f0nzie/rTorch")`

Installing from GitHub gives you the flexibility of experimenting with
the latest development version of `rTorch`. For instance, to install
`rTorch` from the `develop` branch:

`devtools::install_github("f0nzie/rTorch", ref="develop")`

or clone with Git with:

    git clone https://github.com/f0nzie/rTorch.git

# Getting Started

## Tensor types

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
#> tensor([[58,  0,  0],
#>         [ 0,  0,  0],
#>         [ 0,  0,  0]], dtype=torch.uint8)
float_tensor  
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.],
#>         [0., 0., 0.]])
double_tensor 
#> tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
#>         [1.2095e-312, 4.6469e-310, 2.4703e-323]], dtype=torch.float64)
long_tensor   
#> tensor([[             0,              0,              0],
#>         [             0,              0, 94055027694160],
#>         [             0, 94055027588736,              0]])
bool_tensor   
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
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
#> tensor([[0.3499, 0.6426, 0.8035, 1.0417, 0.7170],
#>         [0.5918, 0.4796, 0.9119, 0.9198, 0.2290],
#>         [0.1658, 1.0311, 0.3840, 0.5588, 1.0310]])
```

``` r
# add three tensors
mat0 + mat1 + mat2
#> tensor([[1.3499, 1.6426, 1.8035, 2.0417, 1.7170],
#>         [1.5918, 1.4796, 1.9119, 1.9198, 1.2290],
#>         [1.1658, 2.0311, 1.3840, 1.5588, 2.0310]])
```

``` r
# PyTorch add two tensors using add() function
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[0.8426, 0.6859, 0.6112, 1.2692],
#>         [1.3672, 0.5447, 0.7486, 0.7841],
#>         [1.2332, 0.7599, 0.1869, 0.7936],
#>         [0.2844, 1.4652, 0.2128, 1.4776],
#>         [0.9509, 1.0828, 0.8195, 1.0859]])
print(x + y)
#> tensor([[0.8426, 0.6859, 0.6112, 1.2692],
#>         [1.3672, 0.5447, 0.7486, 0.7841],
#>         [1.2332, 0.7599, 0.1869, 0.7936],
#>         [0.2844, 1.4652, 0.2128, 1.4776],
#>         [0.9509, 1.0828, 0.8195, 1.0859]])
```

### Add a tensor element to another tensor

``` r
# add an element of a tensor to another tensor
mat1[1, 1] + mat2
#> tensor([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
```

``` r
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

``` r
# extract part of the tensor
indices <- torch$tensor(c(0L, 3L))
torch$index_select(mat1, 1L, indices)   # rows = 0; columns = 1
#> tensor([[0.1000, 0.1000],
#>         [0.1000, 0.1000],
#>         [0.1000, 0.1000]])
```

### Add a scalar to a tensor

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[0.3499, 0.6426, 0.8035, 1.0417, 0.7170],
#>         [0.5918, 0.4796, 0.9119, 0.9198, 0.2290],
#>         [0.1658, 1.0311, 0.3840, 0.5588, 1.0310]])
```

### Multiply a tensor by a scalar

``` r
# Multiply tensor by scalar
tensor = torch$ones(4L, dtype=torch$float64)
scalar = np$float64(4.321)
message("a numpy scalar: ", scalar)
#> a numpy scalar: 4.321
message("a PyTorch scalar: ", torch$scalar_tensor(scalar))
#> a PyTorch scalar: tensor(4.3210)
message("\nResult")
#> 
#> Result
(prod = torch$mul(tensor, torch$scalar_tensor(scalar)))
#> tensor([4.3210, 4.3210, 4.3210, 4.3210], dtype=torch.float64)
```

``` r
# short version using generics
(prod = tensor * scalar)
#> tensor([4.3210, 4.3210, 4.3210, 4.3210], dtype=torch.float64)
```

### Multiply two 1D tensors

``` r
t1 = torch$tensor(c(1, 2))
t2 = torch$tensor(c(3, 2))

t1
#> tensor([1., 2.])
t2
#> tensor([3., 2.])
```

``` r
t1 * t2
#> tensor([3., 4.])
```

``` r
t1 = torch$tensor(list(
    c(1, 2, 3),
    c(1, 2, 3)
))

t2 = torch$tensor(list(
    c(1, 2),
    c(1, 2),
    c(1, 2)
))

t1
#> tensor([[1., 2., 3.],
#>         [1., 2., 3.]])
t2
#> tensor([[1., 2.],
#>         [1., 2.],
#>         [1., 2.]])
```

``` r
torch$mm(t1, t2)
#> tensor([[ 6., 12.],
#>         [ 6., 12.]])
```

### Dot product for 1D tensors (vectors)

``` r
t1 = torch$tensor(c(1, 2))
t2 = torch$tensor(c(3, 2))

t1
#> tensor([1., 2.])
t2
#> tensor([3., 2.])
```

``` r
# dot product of two vectors
torch$dot(t1, t2)
#> tensor(7.)
```

``` r
# Dot product of 1D tensors is a scalar

p <- torch$Tensor(list(4L, 2L))
q <- torch$Tensor(list(3L, 1L))                   

(r = torch$dot(p, q)) # 14
#> tensor(14.)
(r <- p %.*% q)
#> tensor(14.)
```

``` r
# torch$dot product will work for vectors not matrices
t1 = torch$tensor(list(
    c(1, 2, 3),
    c(1, 2, 3)
))

t2 = torch$tensor(list(
    c(1, 2),
    c(1, 2),
    c(1, 2)
))

t1$shape
#> torch.Size([2, 3])
t2$shape
#> torch.Size([3, 2])
```

``` r
# RuntimeError: 1D tensors expected, got 2D, 2D tensors
torch$dot(t1, t2)
```

### Dot product for 2D tensors (matrices)

The number of columns of the first matrix must be equal to the number of
rows of the second matrix.

``` r
# for the dot product of nD tensors we use torch$mm()
t1 = torch$tensor(list(
    c(1, 2, 3),
    c(1, 2, 3)
))

t2 = torch$tensor(list(
    c(1, 2),
    c(1, 2),
    c(1, 2)
))

torch$mm(t1, t2)
#> tensor([[ 6., 12.],
#>         [ 6., 12.]])
```

``` r
torch$mm(t2, t1)
#> tensor([[3., 6., 9.],
#>         [3., 6., 9.],
#>         [3., 6., 9.]])
```

``` r
# for the dot product of 2D tensors we use torch$mm()
t1 = torch$arange(1, 11)$view(c(2L,5L))
t2 = torch$arange(11, 21)$view(c(5L,2L))

t1
#> tensor([[ 1.,  2.,  3.,  4.,  5.],
#>         [ 6.,  7.,  8.,  9., 10.]])
t2
#> tensor([[11., 12.],
#>         [13., 14.],
#>         [15., 16.],
#>         [17., 18.],
#>         [19., 20.]])
```

``` r
# result
torch$mm(t1, t2)
#> tensor([[245., 260.],
#>         [620., 660.]])
```

### Multiplication for nD tensors

``` r
# 1D tensor
t1 = torch$tensor(c(1, 2))
t2 = torch$tensor(c(3, 2))

torch$matmul(t1, t2)
#> tensor(7.)
```

``` r
# 2D tensor
t1 = torch$tensor(list(
    c(1, 2, 3),
    c(1, 2, 3)
))

t2 = torch$tensor(list(
    c(1, 2),
    c(1, 2),
    c(1, 2)
))

torch$matmul(t1, t2)
#> tensor([[ 6., 12.],
#>         [ 6., 12.]])
```

``` r
# for the dot product of 3D tensors we use torch$matmul()
t1 = torch$arange(1, 13)$view(c(2L, 2L, 3L))   # number of columns = 2
t2 = torch$arange(0, 18)$view(c(2L, 3L, 3L))   # number of rows = 2

t1
#> tensor([[[ 1.,  2.,  3.],
#>          [ 4.,  5.,  6.]],
#> 
#>         [[ 7.,  8.,  9.],
#>          [10., 11., 12.]]])
t2
#> tensor([[[ 0.,  1.,  2.],
#>          [ 3.,  4.,  5.],
#>          [ 6.,  7.,  8.]],
#> 
#>         [[ 9., 10., 11.],
#>          [12., 13., 14.],
#>          [15., 16., 17.]]])

message("result")
#> result
torch$matmul(t1, t2)
#> tensor([[[ 24.,  30.,  36.],
#>          [ 51.,  66.,  81.]],
#> 
#>         [[294., 318., 342.],
#>          [402., 435., 468.]]])
```

``` r
t1 = torch$arange(1, 13)$view(c(3L, 2L, 2L))   # number of columns = 3
t2 = torch$arange(0, 12)$view(c(3L, 2L, 2L))   # number of rows = 3

t1
#> tensor([[[ 1.,  2.],
#>          [ 3.,  4.]],
#> 
#>         [[ 5.,  6.],
#>          [ 7.,  8.]],
#> 
#>         [[ 9., 10.],
#>          [11., 12.]]])
t2
#> tensor([[[ 0.,  1.],
#>          [ 2.,  3.]],
#> 
#>         [[ 4.,  5.],
#>          [ 6.,  7.]],
#> 
#>         [[ 8.,  9.],
#>          [10., 11.]]])

message("result")
#> result
torch$matmul(t1, t2)
#> tensor([[[  4.,   7.],
#>          [  8.,  15.]],
#> 
#>         [[ 56.,  67.],
#>          [ 76.,  91.]],
#> 
#>         [[172., 191.],
#>          [208., 231.]]])
```

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

## NumPy and PyTorch

`numpy` has been made available as a module inside `rTorch`. We could
call functions from `numpy` refrerring to it as `np$any_function`.
Examples:

``` r
# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.9290388 0.2534306 0.0200905 0.7868779 0.4900428
#> [2,] 0.5507188 0.6299202 0.7060720 0.5930674 0.5016310
#> [3,] 0.6975174 0.7638627 0.2108130 0.1136493 0.8267176
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
# From R we use a vector to refer to a tuple in Python
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
#> [1,] 0.2479481 0.2479481 0.2479481 0.2479481 0.2479481 0.2479481 0.2479481
#> [2,] 0.2981409 0.2981409 0.2981409 0.2981409 0.2981409 0.2981409 0.2981409
#> [3,] 0.2612560 0.2612560 0.2612560 0.2612560 0.2612560 0.2612560 0.2612560
#>           [,8]      [,9]     [,10]
#> [1,] 0.2479481 0.2479481 0.2479481
#> [2,] 0.2981409 0.2981409 0.2981409
#> [3,] 0.2612560 0.2612560 0.2612560
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

### Copying a numpy object

With newer PyTorch versions we should work with NumPy array copies There
have been minor changes in the latest versions of PyTorch that prevents
a direct use of a NumPy array. You will get this warning:

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

### Function `make_copy()`

To make easier to copy an object in `rTorch` we implemented the function
`make_copy`, which makes a safe copy regardless if it is a torch, numpy
or an R type object.

``` r
a = np$array(list(1, 2, 3, 4, 5))

a_copy <- make_copy(a)
t <- torch$as_tensor(a_copy)
t
#> tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
```

### Convert a numpy array to a tensor

``` r
# convert a numpy array to a tensor
np_a = np$array(c(c(3, 4), c(3, 6)))
t_a = torch$from_numpy(r_to_py(np_a)$copy())
print(t_a)
#> tensor([3., 4., 3., 6.], dtype=torch.float64)
```

## Creating tensors

### Random tensor

``` r
# a random 1D tensor
np_arr <- np$random$rand(5L)
ft1 <- torch$FloatTensor(r_to_py(np_arr)$copy())    # make a copy of numpy array
ft1
#> tensor([0.0719, 0.7019, 0.7769, 0.8034, 0.4814])
```

``` r
# tensor as a float of 64-bits
np_copy <- r_to_py(np$random$rand(5L))$copy()       # make a copy of numpy array
ft2 <- torch$as_tensor(np_copy, dtype= torch$float64)
ft2
#> tensor([0.7496, 0.3676, 0.7673, 0.4852, 0.0815], dtype=torch.float64)
```

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]      [,2]       [,3]      [,4]
#> [1,] 0.09395343 0.5595527 0.95875531 0.5781289
#> [2,] 0.08485174 0.8512743 0.09576946 0.5076690
#> [3,] 0.36763144 0.5652164 0.44276500 0.8933945
#> [4,] 0.18297040 0.1315807 0.50351322 0.3735911
#> [5,] 0.82712889 0.5211104 0.32848728 0.9897768
```

### Change the type of a tensor

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.7495, 0.3674, 0.7671, 0.4851, 0.0815], dtype=torch.float16)
```

### Create an uninitialized tensor

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[7.5670e-44, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          1.4013e-45]])
```

### Create a tensor and then change its shape

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
#> tensor([[-1.2573, -0.9536,  0.6686, -1.6667, -1.4893,  0.1584, -0.0630],
#>         [ 0.2651,  2.0210, -0.7101,  0.3994,  1.1177, -0.8158,  0.6742],
#>         [ 0.3766,  0.0532,  1.2202,  0.3820, -0.4392,  0.2881,  0.7138],
#>         [ 0.1537, -1.2810, -0.5439, -0.4608,  0.9275,  0.6893,  0.2966],
#>         [-0.8121, -0.3847, -0.0489,  0.4341,  0.1797,  0.8832, -0.1888]])
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
#> tensor([[0.3815, 0.9766, 0.7075, 0.1034, 0.5313],
#>         [0.3810, 0.2989, 0.4528, 0.3165, 0.7326],
#>         [0.8210, 0.7691, 0.0505, 0.9630, 0.8254]])
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
#> tensor([  0.,  21.,  83., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[ 4.,  4.],
#>         [ 6., 10.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([0.9798])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([0.4497])
```

## Tensor default data types

Only floating-point types are supported as the default type.

### float32

``` r
# Default data type
torch$tensor(list(1.2, 3))$dtype  # default for floating point is torch.float32
#> torch.float32
```

### float64

``` r
# change default data type to float64
torch$set_default_dtype(torch$float64)
torch$tensor(list(1.2, 3))$dtype         # a new floating point tensor
#> torch.float64
```

### double

``` r
torch$set_default_dtype(torch$double)
torch$tensor(list(1.2, 3))$dtype
#> torch.float64
```

## Tensor resizing

### Using *view*

``` r
x = torch$randn(2L, 3L)            # Size 2x3
y = x$view(6L)                    # Resize x to size 6
z = x$view(-1L, 2L)                # Size 3x2
print(y)
#> tensor([ 2.6756,  0.0601,  0.1541, -0.0576, -0.6315,  2.1067])
print(z)
#> tensor([[ 2.6756,  0.0601],
#>         [ 0.1541, -0.0576],
#>         [-0.6315,  2.1067]])
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

### Concatenating tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[ 1.4248, -0.2922,  0.2605],
#>         [-1.6589, -0.8634, -1.8625]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 1.4248, -0.2922,  0.2605],
#>         [-1.6589, -0.8634, -1.8625],
#>         [ 1.4248, -0.2922,  0.2605],
#>         [-1.6589, -0.8634, -1.8625],
#>         [ 1.4248, -0.2922,  0.2605],
#>         [-1.6589, -0.8634, -1.8625]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 1.4248, -0.2922,  0.2605,  1.4248, -0.2922,  0.2605,  1.4248, -0.2922,
#>           0.2605],
#>         [-1.6589, -0.8634, -1.8625, -1.6589, -0.8634, -1.8625, -1.6589, -0.8634,
#>          -1.8625]])
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

### Eye

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

### Fill with a unique value

``` r
# a tensor filled with ones
(v = torch$ones(3L, 3L))
#> tensor([[1., 1., 1.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
```

### Change the tensor values by rows

``` r
# change two rows in the tensor
# we are using 1-based index
v[2L, ]$fill_(2L)         # fill row 1 with 2s
#> tensor([2., 2., 2.])
v[3L, ]$fill_(3L)         # fill row 2 with 3s
#> tensor([3., 3., 3.])
```

``` r
print(v)
#> tensor([[1., 1., 1.],
#>         [2., 2., 2.],
#>         [3., 3., 3.]])
```

### Fill a tensor with a set increment

``` r
# Initialize Tensor with a range of values
(v = torch$arange(10L))             # similar to range(5) but creating a Tensor
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

``` r
(v = torch$arange(0L, 10L, step = 1L))  # Size 5. Similar to range(0, 5, 1)
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### With decimal increments

``` r
u <- torch$arange(0, 10, step = 0.5)
u
#> tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000,
#>         4.5000, 5.0000, 5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000,
#>         9.0000, 9.5000])
```

### Including the ending value

``` r
# range of values with increments including the end value
start <- 0
end   <- 10
step  <- 0.25

w <- torch$arange(start, end+step, step)
w
#> tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,
#>          2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,
#>          4.0000,  4.2500,  4.5000,  4.7500,  5.0000,  5.2500,  5.5000,  5.7500,
#>          6.0000,  6.2500,  6.5000,  6.7500,  7.0000,  7.2500,  7.5000,  7.7500,
#>          8.0000,  8.2500,  8.5000,  8.7500,  9.0000,  9.2500,  9.5000,  9.7500,
#>         10.0000])
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

### In-place / Not-in-place

``` r
a = torch$rand(5L, 4L)
print(class(a))
#> [1] "torch.Tensor"          "torch._C._TensorBase"  "python.builtin.object"
```

``` r
# converting the tensor to a numpy array, R automatically converts it
b = a$numpy()
print(class(b))
#> [1] "matrix"
```

``` r
a$fill_(3.5)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000]])
# a has now been filled with the value 3.5

# add a scalar to a tensor. 
# notice that was auto-converted from an array to a tensor
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

### Tensor element assigment not implemented yet

``` r
# this will throw an error because we don't still have a function for assignment
a[1, 1] <- 7.7
print(a)
# Error in a[1, 1] <- 7.7 : object of type 'environment' is not subsettable
```

``` r
# This would be the right wayy to assign a value to a tensor element
a[1, 1]$fill_(7.7)
#> tensor(7.7000)
```

``` r
# we can see that the first element has been changed
a
#> tensor([[7.7000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000]])
```

> Some operations like`narrow` do not have in-place versions, and hence,
> `.narrow_` does not exist. Similarly, some operations like `fill_` do
> not have an out-of-place version, so `.fill` does not exist.

``` r
# a[[0L, 3L]]
a[1, 4]
#> tensor(3.5000)
```

## Access to tensor elements

### Change a tensor element given its index

``` r
# replace an element at position 0, 0
(new_tensor = torch$Tensor(list(list(1, 2), list(3, 4))))
#> tensor([[1., 2.],
#>         [3., 4.]])
```

``` r
# first row, firt column
print(new_tensor[1L, 1L])
#> tensor(1.)
```

``` r
# change row 1, col 1 with value of 5
new_tensor[1L, 1L]$fill_(5)
#> tensor(5.)
```

``` r
# which is the same as doing this
new_tensor[1, 1]$fill_(5)
#> tensor(5.)
```

> Notice that the element was changed in-place because of `fill_`.

### In R the index is 1-based

``` r
print(new_tensor)   # tensor([[ 5.,  2.],[ 3.,  4.]])
#> tensor([[5., 2.],
#>         [3., 4.]])
```

``` r
# access an element at position (1, 0), 0-based index
print(new_tensor[2L, 1L])           # tensor([ 3.])
#> tensor(3.)
```

``` r
# convert it to a scalar value
print(new_tensor[2L, 1L]$item())    # 3.
#> [1] 3
```

``` r
# which is the same as
print(new_tensor[2, 1])
#> tensor(3.)
```

``` r
# and the scalar
print(new_tensor[2, 1]$item()) 
#> [1] 3
```

### Extract part of a tensor

``` r
# Select indices
x = torch$randn(3L, 4L)
print(x)
#> tensor([[ 2.0866, -1.5293, -0.0176, -0.0173],
#>         [ 0.4589,  0.4754, -0.4929,  0.8043],
#>         [ 0.0169,  0.4875,  0.5488,  0.2263]])
```

``` r
# extract first and third row
# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 2.0866, -1.5293, -0.0176, -0.0173],
#>         [ 0.0169,  0.4875,  0.5488,  0.2263]])
```

``` r
# extract first and third column
# Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[ 2.0866, -0.0176],
#>         [ 0.4589, -0.4929],
#>         [ 0.0169,  0.5488]])
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

### Transpose

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

### Permutation

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
torch$manual_seed(1234)
#> <torch._C.Generator>

x <- torch$randn(10L, 480L, 640L, 3L)
x[1:3, 1:2, 1:3, 1:2]
#> tensor([[[[-0.0883,  0.3420],
#>           [ 1.0051, -0.1117],
#>           [-0.0982, -0.3511]],
#> 
#>          [[-0.1465,  0.3960],
#>           [-1.6878,  0.5720],
#>           [ 0.9426,  2.1187]]],
#> 
#> 
#>         [[[ 0.8107,  0.9289],
#>           [ 0.4210, -1.5109],
#>           [-1.8483, -0.4636]],
#> 
#>          [[-1.8324, -1.9304],
#>           [-2.7020,  0.3491],
#>           [ 0.9180, -1.9872]]],
#> 
#> 
#>         [[[ 1.6555, -0.3531],
#>           [ 0.4763,  0.8037],
#>           [-0.2171, -0.0839]],
#> 
#>          [[-0.0886, -1.3389],
#>           [ 0.7163, -0.9050],
#>           [-0.8144, -1.4922]]]])
```

``` r
xs <- torch$as_tensor(x$size())     # torch$tensor(c(10L, 480L, 640L, 3L))
xp <- x$permute(0L, 3L, 1L, 2L)     # specify dimensions order
xps <- torch$as_tensor(xp$size())   # torch$tensor(c(10L, 3L, 480L, 640L))

print(xs)      # original tensor size
#> tensor([ 10, 480, 640,   3])

print(xps)     # permuted tensor size
#> tensor([ 10,   3, 480, 640])
```

``` r
xp[1:3, 1:2, 1:3, 1:2]
#> tensor([[[[-0.0883,  1.0051],
#>           [-0.1465, -1.6878],
#>           [-0.6429,  0.5577]],
#> 
#>          [[ 0.3420, -0.1117],
#>           [ 0.3960,  0.5720],
#>           [ 0.3014,  0.7813]]],
#> 
#> 
#>         [[[ 0.8107,  0.4210],
#>           [-1.8324, -2.7020],
#>           [ 1.1724,  0.4434]],
#> 
#>          [[ 0.9289, -1.5109],
#>           [-1.9304,  0.3491],
#>           [ 0.9901, -1.3630]]],
#> 
#> 
#>         [[[ 1.6555,  0.4763],
#>           [-0.0886,  0.7163],
#>           [-0.7774, -0.6281]],
#> 
#>          [[-0.3531,  0.8037],
#>           [-1.3389, -0.9050],
#>           [-0.7920,  1.3634]]]])
```

## Logical operations

### is it equal

``` r
(m0 = torch$zeros(3L, 5L))
#> tensor([[0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.]])
```

``` r
(m1 = torch$ones(3L, 5L))
#> tensor([[1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.],
#>         [1., 1., 1., 1., 1.]])
```

``` r
(m2 = torch$eye(3L, 5L))
#> tensor([[1., 0., 0., 0., 0.],
#>         [0., 1., 0., 0., 0.],
#>         [0., 0., 1., 0., 0.]])
```

``` r
# is m1 equal to m0
print(m1 == m0)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

``` r
print(as_boolean(m1 == m0))
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

### is it not equal

``` r
# is it not equal
print(m1 != m1)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

``` r
# are both equal
print(m2 == m2)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

``` r
print(as_boolean(m2 == m2))
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

``` r
# some are equal, others don't
m1 != m2
#> tensor([[False,  True,  True,  True,  True],
#>         [ True, False,  True,  True,  True],
#>         [ True,  True, False,  True,  True]])
```

``` r
# some are equal, others don't
m0 != m2
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False]])
```

``` r
as_boolean(m0 != m2)
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False]])
```

### AND

``` r
# AND
m1 & m1
#> tensor([[1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1]], dtype=torch.uint8)
```

``` r
as_boolean(m1 & m1)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

### OR

``` r
# OR
m0 | m2
#> tensor([[1, 0, 0, 0, 0],
#>         [0, 1, 0, 0, 0],
#>         [0, 0, 1, 0, 0]], dtype=torch.uint8)
```

``` r
# OR
m1 | m2
#> tensor([[1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1]], dtype=torch.uint8)
```

``` r
as_boolean(m1 | m2)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

### Extract only one logical result with *all*

``` r
# tensor is less than
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

# is C < A = TRUE
all(torch$lt(C, A)) 
#> tensor(1, dtype=torch.uint8)
all(C < A) 
#> tensor(1, dtype=torch.uint8)
# is A < C = FALSE
all(A < C)
#> tensor(0, dtype=torch.uint8)
```

### greater than

``` r
# tensor is greater than
A <- torch$ones(60000L, 1L, 28L, 28L)
D <- A * 2.0
all(torch$gt(D, A))
#> tensor(1, dtype=torch.uint8)
all(torch$gt(A, D))
#> tensor(0, dtype=torch.uint8)
```

### lower than

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

### As a R logical value

``` r
# we implement this little function
all_as_boolean <- function(x) {
  # convert tensor of 1s and 0s to a unique boolean
  as.logical(torch$all(x)$numpy())
}
```

``` r
all_as_boolean(torch$gt(D, A))
#> [1] TRUE
all_as_boolean(torch$gt(A, D))
#> [1] FALSE
all_as_boolean(A1 <= A1)
#> [1] TRUE
all_as_boolean(A1 >= A0)
#> [1] TRUE
all_as_boolean(A1 <= A0)
#> [1] FALSE
```

### Logical NOT

``` r
# vector of booleans
all_true <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
all_true
#> tensor([True, True, True, True])
```

``` r
# logical NOT
# negate vector with "!"
not_all_true <- !all_true
not_all_true
#> tensor([False, False, False, False])
```

``` r
# a diagonal matrix
diag <- torch$eye(5L)
diag <- diag$to(dtype=torch$uint8)   # convert to unsigned integer
diag
#> tensor([[1, 0, 0, 0, 0],
#>         [0, 1, 0, 0, 0],
#>         [0, 0, 1, 0, 0],
#>         [0, 0, 0, 1, 0],
#>         [0, 0, 0, 0, 1]], dtype=torch.uint8)
```

``` r
as_boolean(diag)
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False],
#>         [False, False, False,  True, False],
#>         [False, False, False, False,  True]])
```

``` r
# logical NOT
not_diag <- !diag
not_diag
#> tensor([[0, 1, 1, 1, 1],
#>         [1, 0, 1, 1, 1],
#>         [1, 1, 0, 1, 1],
#>         [1, 1, 1, 0, 1],
#>         [1, 1, 1, 1, 0]], dtype=torch.uint8)
```

``` r
# and the negation
!not_diag
#> tensor([[1, 0, 0, 0, 0],
#>         [0, 1, 0, 0, 0],
#>         [0, 0, 1, 0, 0],
#>         [0, 0, 0, 1, 0],
#>         [0, 0, 0, 0, 1]], dtype=torch.uint8)
```

``` r
as_boolean(!not_diag)
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False],
#>         [False, False, False,  True, False],
#>         [False, False, False, False,  True]])
```
