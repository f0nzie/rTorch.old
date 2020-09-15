
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
classes and functions, the package also provides the modules `numpy` as
a method called `np`, and `torchvision`, as well. The dollar sign `$`
after the module will provide you access to all their sub-objects.

## rTorch installation

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
#> tensor([[114,  32, 105],
#>         [115,  32,  99],
#>         [114, 101, 193]], dtype=torch.uint8)
float_tensor  
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.],
#>         [0., 0., 0.]])
double_tensor 
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.],
#>         [0., 0., 0.]], dtype=torch.float64)
long_tensor   
#> tensor([[139900215718160, 139900215718160,  94325669576688],
#>         [ 94325669576688,               0,               0],
#>         [              0,               0,               0]])
bool_tensor   
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
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
#> tensor([[0.5746, 0.7640, 0.7387, 0.9518, 0.4869],
#>         [0.2469, 0.4920, 0.9856, 0.6682, 0.7042],
#>         [0.4145, 1.0038, 0.2208, 0.2450, 0.3713]])
```

``` r
# add three tensors
mat0 + mat1 + mat2
#> tensor([[1.5746, 1.7640, 1.7387, 1.9518, 1.4869],
#>         [1.2469, 1.4920, 1.9856, 1.6682, 1.7042],
#>         [1.4145, 2.0038, 1.2208, 1.2450, 1.3713]])
```

``` r
# PyTorch add two tensors using add() function
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.0109, 1.4988, 1.3009, 0.8333],
#>         [0.4340, 0.3702, 0.7475, 0.1130],
#>         [0.7191, 1.2372, 1.1599, 1.0963],
#>         [0.5287, 1.2010, 1.3933, 0.9388],
#>         [1.6711, 1.0761, 1.5275, 1.8138]])
print(x + y)
#> tensor([[1.0109, 1.4988, 1.3009, 0.8333],
#>         [0.4340, 0.3702, 0.7475, 0.1130],
#>         [0.7191, 1.2372, 1.1599, 1.0963],
#>         [0.5287, 1.2010, 1.3933, 0.9388],
#>         [1.6711, 1.0761, 1.5275, 1.8138]])
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
#> tensor([[0.5746, 0.7640, 0.7387, 0.9518, 0.4869],
#>         [0.2469, 0.4920, 0.9856, 0.6682, 0.7042],
#>         [0.4145, 1.0038, 0.2208, 0.2450, 0.3713]])
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
#> [1,] 0.2076541 0.1214194 0.8808115 0.3676121 0.0593298
#> [2,] 0.3174220 0.2168312 0.4620522 0.3991046 0.4284798
#> [3,] 0.5659232 0.6665505 0.3673082 0.2586406 0.4848122
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
#> [1,] 0.1636827 0.1636827 0.1636827 0.1636827 0.1636827 0.1636827 0.1636827
#> [2,] 0.1823890 0.1823890 0.1823890 0.1823890 0.1823890 0.1823890 0.1823890
#> [3,] 0.2343235 0.2343235 0.2343235 0.2343235 0.2343235 0.2343235 0.2343235
#>           [,8]      [,9]     [,10]
#> [1,] 0.1636827 0.1636827 0.1636827
#> [2,] 0.1823890 0.1823890 0.1823890
#> [3,] 0.2343235 0.2343235 0.2343235
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
#> tensor([0.5942, 0.1451, 0.0420, 0.5183, 0.3151])
```

``` r
# tensor as a float of 64-bits
np_copy <- r_to_py(np$random$rand(5L))$copy()       # make a copy of numpy array
ft2 <- torch$as_tensor(np_copy, dtype= torch$float64)
ft2
#> tensor([0.3140, 0.9342, 0.0047, 0.1691, 0.8316], dtype=torch.float64)
```

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]       [,2]       [,3]      [,4]
#> [1,] 0.79575664 0.53712749 0.48124319 0.2021467
#> [2,] 0.06835997 0.99231666 0.13017726 0.5740867
#> [3,] 0.31663889 0.78170615 0.02760822 0.9861709
#> [4,] 0.13180590 0.07526958 0.12124765 0.7047765
#> [5,] 0.82200909 0.69837981 0.90379471 0.6348743
```

### Change the type of a tensor

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.3140, 0.9341, 0.0047, 0.1692, 0.8315], dtype=torch.float16)
```

### Create an uninitialized tensor

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 1.4013e-45,  0.0000e+00, -1.8382e+22,  3.0774e-41,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00]])
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
#> tensor([[-0.3887,  0.1765,  0.7712, -0.3002,  0.5599,  0.1540, -0.1114],
#>         [-0.2067,  2.3662, -0.8973,  1.9099,  0.0487,  0.3764, -0.1185],
#>         [-0.8155, -1.4124,  0.1771,  1.1652,  0.5650,  0.3808,  0.1084],
#>         [-0.0330, -1.8995, -0.1402, -1.1151, -0.0740,  0.8217,  0.0933],
#>         [-1.5162,  1.6975,  0.4799, -0.4325,  0.6193, -1.0418,  0.3048]])
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
#> tensor([[0.6003, 0.9577, 0.2927, 0.5873, 0.2687],
#>         [0.1030, 0.5315, 0.2114, 0.0168, 0.7232],
#>         [0.6642, 0.1408, 0.6290, 0.2166, 0.9445]])
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
#> tensor([  0.,  14.,  80., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[2., 4.],
#>         [6., 8.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([0.7829])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([1.0214])
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
#> tensor([ 2.1882,  0.3418, -0.9996,  0.2132, -0.7871,  0.1086])
print(z)
#> tensor([[ 2.1882,  0.3418],
#>         [-0.9996,  0.2132],
#>         [-0.7871,  0.1086]])
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
#> tensor([[ 1.1650, -0.6736,  0.4869],
#>         [ 0.7100, -0.2433,  1.2539]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 1.1650, -0.6736,  0.4869],
#>         [ 0.7100, -0.2433,  1.2539],
#>         [ 1.1650, -0.6736,  0.4869],
#>         [ 0.7100, -0.2433,  1.2539],
#>         [ 1.1650, -0.6736,  0.4869],
#>         [ 0.7100, -0.2433,  1.2539]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 1.1650, -0.6736,  0.4869,  1.1650, -0.6736,  0.4869,  1.1650, -0.6736,
#>           0.4869],
#>         [ 0.7100, -0.2433,  1.2539,  0.7100, -0.2433,  1.2539,  0.7100, -0.2433,
#>           1.2539]])
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
#> tensor([[0.6361, 0.9141, 0.0404, 0.9962],
#>         [0.6814, 1.5302, 2.1698, 0.2040],
#>         [1.7583, 0.9915, 0.0707, 0.0562]])
```

``` r
# extract first and third row
# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[0.6361, 0.9141, 0.0404, 0.9962],
#>         [1.7583, 0.9915, 0.0707, 0.0562]])
```

``` r
# extract first and third column
# Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[0.6361, 0.0404],
#>         [0.6814, 2.1698],
#>         [1.7583, 0.0707]])
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
#> tensor([[0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0]], dtype=torch.uint8)
```

### is it not equal

``` r
print(m1 != m1)
#> tensor([[0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0]], dtype=torch.uint8)
```

``` r
# are both equal
print(m2 == m2)
#> tensor([[1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1],
#>         [1, 1, 1, 1, 1]], dtype=torch.uint8)
```

``` r
# some are equal, others don't
m1 != m2
#> tensor([[0, 1, 1, 1, 1],
#>         [1, 0, 1, 1, 1],
#>         [1, 1, 0, 1, 1]], dtype=torch.uint8)
```

``` r
# some are equal, others don't
m0 != m2
#> tensor([[1, 0, 0, 0, 0],
#>         [0, 1, 0, 0, 0],
#>         [0, 0, 1, 0, 0]], dtype=torch.uint8)
```

### AND

``` r
# AND
m1 & m1
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```

### OR

``` r
# OR
m0 | m2
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False]], dtype=torch.bool)
```

``` r
# OR
m1 | m2
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
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
#> tensor([True, True, True, True], dtype=torch.bool)
```

``` r
# logical NOT
# negate vector with "!"
not_all_true <- !all_true
not_all_true
#> tensor([False, False, False, False], dtype=torch.bool)
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
