
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
#> tensor([[  5,   0,  44],
#>         [ 32,  42,  44],
#>         [ 32,  68, 101]], dtype=torch.uint8)
float_tensor  
#> tensor([[ 0.0000e+00,  4.6566e-10,  0.0000e+00],
#>         [ 4.6566e-10, -4.2796e+22,  4.5915e-41],
#>         [ 7.9050e+31,  5.4214e-11,  1.1289e+27]])
double_tensor 
#> tensor([[ 1.7272e-77,  1.7272e-77, 6.9531e-310],
#>         [ 4.8958e-85,  4.8921e-85, 8.4084e-315],
#>         [ 4.8958e-85,  4.8921e-85, 6.3226e-310]], dtype=torch.float64)
long_tensor   
#> tensor([[              0,               0,               0],
#>         [              0,               0,               0],
#>         [121398885512202,               0,               0]])
bool_tensor   
#> tensor([[False, False, False, False, False],
#>         [False, False,  True, False, False],
#>         [False, False, False, False, False],
#>         [ True,  True, False,  True,  True],
#>         [ True,  True, False, False,  True]])
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
#> tensor([[[4.9045e-44, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 4.5918e-39],
#>          [0.0000e+00, 4.6566e-10],
#>          [0.0000e+00, 4.6566e-10]],
#> 
#>         [[4.4842e-44, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(4.9045e-44)
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
#> tensor([[0.3664, 1.0968, 0.8337, 1.0143, 0.8043],
#>         [1.0517, 0.6268, 0.7897, 1.0059, 0.6248],
#>         [0.9122, 0.4131, 0.4224, 0.7292, 0.6625]])
```

``` r
# add three tensors
mat0 + mat1 + mat2
#> tensor([[1.3664, 2.0968, 1.8337, 2.0143, 1.8043],
#>         [2.0517, 1.6268, 1.7897, 2.0059, 1.6248],
#>         [1.9122, 1.4131, 1.4224, 1.7292, 1.6625]])
```

``` r
# PyTorch add two tensors using add() function
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[0.7121, 0.2187, 0.9743, 1.4073],
#>         [0.6882, 1.1412, 0.8857, 0.6805],
#>         [0.8604, 0.4032, 1.2354, 0.4280],
#>         [0.4615, 0.8107, 0.5310, 0.6479],
#>         [1.3691, 1.5906, 0.9496, 1.4169]])
print(x + y)
#> tensor([[0.7121, 0.2187, 0.9743, 1.4073],
#>         [0.6882, 1.1412, 0.8857, 0.6805],
#>         [0.8604, 0.4032, 1.2354, 0.4280],
#>         [0.4615, 0.8107, 0.5310, 0.6479],
#>         [1.3691, 1.5906, 0.9496, 1.4169]])
```

### Add tensor element to another tensor

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
#> tensor([[0.3664, 1.0968, 0.8337, 1.0143, 0.8043],
#>         [1.0517, 0.6268, 0.7897, 1.0059, 0.6248],
#>         [0.9122, 0.4131, 0.4224, 0.7292, 0.6625]])
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

## NumPy and PyTorch

`numpy` has been made available as a module inside `rTorch`. We could
call functions from `numpy` refrerring to it as `np$any_function`.
Examples:

``` r
# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]       [,2]      [,3]      [,4]      [,5]
#> [1,] 0.5807750 0.44946209 0.6881341 0.3810754 0.7320024
#> [2,] 0.7046284 0.09338938 0.6660603 0.4260538 0.4196759
#> [3,] 0.8039982 0.18632835 0.4121821 0.7750014 0.1014839
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
#> [1,] 0.2831449 0.2831449 0.2831449 0.2831449 0.2831449 0.2831449 0.2831449
#> [2,] 0.2309808 0.2309808 0.2309808 0.2309808 0.2309808 0.2309808 0.2309808
#> [3,] 0.2278994 0.2278994 0.2278994 0.2278994 0.2278994 0.2278994 0.2278994
#>           [,8]      [,9]     [,10]
#> [1,] 0.2831449 0.2831449 0.2831449
#> [2,] 0.2309808 0.2309808 0.2309808
#> [3,] 0.2278994 0.2278994 0.2278994
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

## Copying a numpy object

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

## Function `make_copy()`

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
#> tensor([0.3151, 0.5259, 0.9153, 0.0388, 0.9810])
```

``` r
# tensor as a float of 64-bits
np_copy <- r_to_py(np$random$rand(5L))$copy()       # make a copy of numpy array
ft2 <- torch$as_tensor(np_copy, dtype= torch$float64)
ft2
#> tensor([0.2048, 0.5739, 0.6036, 0.1178, 0.2643], dtype=torch.float64)
```

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]      [,2]      [,3]      [,4]
#> [1,] 0.08230877 0.5072403 0.9124174 0.6441675
#> [2,] 0.55823284 0.8172520 0.1416471 0.9792743
#> [3,] 0.58075166 0.1236397 0.6648014 0.4343626
#> [4,] 0.23225826 0.6618602 0.2034529 0.7483020
#> [5,] 0.98649752 0.4407772 0.9488737 0.9278077
```

### Change the type of a tensor

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.2048, 0.5737, 0.6035, 0.1178, 0.2644], dtype=torch.float16)
```

### Create an uninitialized tensor

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[0.0000e+00, 4.6566e-10, 0.0000e+00, 4.6566e-10, 1.4569e-19, 1.4584e-19,
#>          3.1432e-12],
#>         [7.9309e+34, 6.0022e+31, 4.2964e+24, 7.3162e+28, 1.6992e-07, 4.4199e-35,
#>          4.5849e-41],
#>         [0.0000e+00, 0.0000e+00, 1.4013e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 1.1473e-36, 4.5849e-41, 0.0000e+00, 0.0000e+00, 4.2298e-35,
#>          4.5849e-41],
#>         [0.0000e+00, 0.0000e+00, 1.4013e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#>          0.0000e+00]])
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
#> tensor([[-0.6252,  1.4206,  1.9980,  0.6017, -0.5984, -0.2370, -1.1494],
#>         [-1.1018, -1.3244, -1.3818, -1.5446, -0.5809, -0.7988,  0.7746],
#>         [ 0.4823, -2.3145, -0.7568, -0.4968, -0.9169, -0.8212,  1.5908],
#>         [-0.8044,  1.3315,  0.7065,  0.2724,  0.7976, -0.1997, -1.2344],
#>         [ 2.0187,  1.7635, -2.7846,  0.5588, -0.0895, -0.3347, -0.5778]])
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
#> tensor([[0.0982, 0.3751, 0.5477, 0.0490, 0.5363],
#>         [0.1151, 0.3961, 0.2104, 0.8873, 0.8115],
#>         [0.0163, 0.5628, 0.1268, 0.7364, 0.1732]])
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
#> tensor([  0.,  20.,  73., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[1., 3.],
#>         [5., 9.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([1.4582])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([1.3564])
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
#> tensor([-0.5665,  0.1773, -1.9708,  0.0297,  0.5782,  1.3566])
print(z)
#> tensor([[-0.5665,  0.1773],
#>         [-1.9708,  0.0297],
#>         [ 0.5782,  1.3566]])
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
#> tensor([[-0.1384,  0.5156, -3.3412],
#>         [ 1.4359, -1.1714, -0.4222]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[-0.1384,  0.5156, -3.3412],
#>         [ 1.4359, -1.1714, -0.4222],
#>         [-0.1384,  0.5156, -3.3412],
#>         [ 1.4359, -1.1714, -0.4222],
#>         [-0.1384,  0.5156, -3.3412],
#>         [ 1.4359, -1.1714, -0.4222]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[-0.1384,  0.5156, -3.3412, -0.1384,  0.5156, -3.3412, -0.1384,  0.5156,
#>          -3.3412],
#>         [ 1.4359, -1.1714, -0.4222,  1.4359, -1.1714, -0.4222,  1.4359, -1.1714,
#>          -0.4222]])
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
# Initialize Tensor with a range of values
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

### In-place / Not-in-place

``` r
a = torch$rand(5L, 4L)
b = a$numpy()

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
#> tensor([[-1.0042,  0.6887,  0.9738,  2.5175],
#>         [ 0.1917, -0.5764,  0.0656,  0.1924],
#>         [-1.0886,  1.0305,  1.5860,  1.2071]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[-1.0042,  0.6887,  0.9738,  2.5175],
#>         [-1.0886,  1.0305,  1.5860,  1.2071]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[-1.0042,  0.9738],
#>         [ 0.1917,  0.0656],
#>         [-1.0886,  1.5860]])
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

### is it equal

``` r
m0 = torch$zeros(3L, 5L)
m1 = torch$ones(3L, 5L)
m2 = torch$eye(3L, 5L)

print(m1 == m0)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]])
```

### is it not equal

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

### AND

``` r
# AND
m1 & m1
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]])
```

### OR

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

### One logical result with *all*

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
