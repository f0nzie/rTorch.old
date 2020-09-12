
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
#> tensor([[0, 0, 0],
#>         [0, 0, 0],
#>         [0, 0, 0]], dtype=torch.uint8)
float_tensor  
#> tensor([[1.1117e+11, 4.5866e-41, 0.0000e+00],
#>         [0.0000e+00,        nan, 1.4013e-45],
#>         [1.1079e+11, 4.5866e-41, 1.1115e+11]])
double_tensor 
#> tensor([[1.6304e-322, 1.3834e-322, 6.9456e-310],
#>         [6.1678e+223, 3.1711e+180, 5.2420e-320],
#>         [2.4209e-322, 2.2233e-322, 6.9456e-310]], dtype=torch.float64)
long_tensor   
#> tensor([[7812726245348745316, 7575164960144188271, 2314885530818447982],
#>         [7597608191290253344, 7882826708672671342, 8245905238468948591],
#>         [2340020702962283371, 2329856631471501158, 3347144771069961588]])
bool_tensor   
#> tensor([[False, False, False, False, False],
#>         [False, False,  True,  True,  True],
#>         [ True,  True,  True,  True, False],
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
#> tensor([[[1.4013e-44, 0.0000e+00],
#>          [9.8964e+05, 4.5866e-41],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [9.3327e+05, 4.5866e-41],
#>          [9.8964e+05, 4.5866e-41]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [9.8964e+05, 4.5866e-41],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [5.6052e-45, 0.0000e+00],
#>          [9.8963e+05, 4.5866e-41]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(1.4013e-44)
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
#> tensor([[0.1124, 0.6488, 0.9673, 0.7103, 0.9605],
#>         [0.7316, 0.7476, 0.7593, 0.1021, 0.4390],
#>         [0.4760, 0.6099, 0.5688, 0.3839, 0.8399]])
```

``` r
# add three tensors
mat0 + mat1 + mat2
#> tensor([[1.1124, 1.6488, 1.9673, 1.7103, 1.9605],
#>         [1.7316, 1.7476, 1.7593, 1.1021, 1.4390],
#>         [1.4760, 1.6099, 1.5688, 1.3839, 1.8399]])
```

``` r
# PyTorch add two tensors using add() function
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.0658, 0.8422, 1.6794, 0.9163],
#>         [0.2763, 0.6545, 0.4278, 0.2479],
#>         [1.5085, 1.4466, 1.6079, 0.6820],
#>         [1.3347, 0.4433, 0.5862, 0.9288],
#>         [0.3750, 0.6231, 1.1827, 1.0419]])
print(x + y)
#> tensor([[1.0658, 0.8422, 1.6794, 0.9163],
#>         [0.2763, 0.6545, 0.4278, 0.2479],
#>         [1.5085, 1.4466, 1.6079, 0.6820],
#>         [1.3347, 0.4433, 0.5862, 0.9288],
#>         [0.3750, 0.6231, 1.1827, 1.0419]])
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
#> tensor([[0.1124, 0.6488, 0.9673, 0.7103, 0.9605],
#>         [0.7316, 0.7476, 0.7593, 0.1021, 0.4390],
#>         [0.4760, 0.6099, 0.5688, 0.3839, 0.8399]])
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
#>           [,1]      [,2]        [,3]      [,4]      [,5]
#> [1,] 0.1044802 0.8447997 0.438620955 0.3189977 0.1029247
#> [2,] 0.0250070 0.5746845 0.320162157 0.7344718 0.1604845
#> [3,] 0.6127711 0.2020533 0.006989334 0.6688043 0.4583932
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
#> [1,] 0.1809823 0.1809823 0.1809823 0.1809823 0.1809823 0.1809823 0.1809823
#> [2,] 0.1814810 0.1814810 0.1814810 0.1814810 0.1814810 0.1814810 0.1814810
#> [3,] 0.1949011 0.1949011 0.1949011 0.1949011 0.1949011 0.1949011 0.1949011
#>           [,8]      [,9]     [,10]
#> [1,] 0.1809823 0.1809823 0.1809823
#> [2,] 0.1814810 0.1814810 0.1814810
#> [3,] 0.1949011 0.1949011 0.1949011
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
#> tensor([0.0795, 0.1967, 0.5089, 0.8433, 0.6677])
```

``` r
# tensor as a float of 64-bits
np_copy <- r_to_py(np$random$rand(5L))$copy()       # make a copy of numpy array
ft2 <- torch$as_tensor(np_copy, dtype= torch$float64)
ft2
#> tensor([0.8669, 0.5868, 0.5255, 0.1008, 0.1381], dtype=torch.float64)
```

This is a very common operation in machine learning:

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>            [,1]      [,2]      [,3]      [,4]
#> [1,] 0.96578276 0.6540611 0.1716326 0.4408271
#> [2,] 0.02305788 0.9562197 0.2664854 0.0752635
#> [3,] 0.20605791 0.3758263 0.6878928 0.8542705
#> [4,] 0.76719004 0.5470078 0.3737034 0.5113918
#> [5,] 0.67690694 0.3024936 0.1514014 0.4893618
```

### Change the type of a tensor

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.8667, 0.5869, 0.5254, 0.1008, 0.1381], dtype=torch.float16)
```

### Create an uninitialized tensor

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 0.0000e+00, -8.5899e+09,  0.0000e+00, -8.5899e+09,  4.4842e-44,
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
#> tensor([[-0.1685,  0.9596,  1.9220, -2.2827,  0.5819, -0.3996, -0.6784],
#>         [-0.2226,  0.9768,  0.1292,  0.6981,  1.2947,  1.0933, -0.6504],
#>         [-1.4755,  1.5880,  0.5880, -0.1356, -0.0509, -1.0395,  1.6771],
#>         [ 0.6305, -1.3544, -0.3731,  0.5021,  2.0973, -1.0560, -1.2293],
#>         [ 0.4103,  0.3055, -0.8569,  0.1605, -1.0973, -0.0048, -1.6648]])
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
#> tensor([[0.2722, 0.9828, 0.5230, 0.8556, 0.7624],
#>         [0.0271, 0.1318, 0.0186, 0.6693, 0.3144],
#>         [0.3765, 0.5742, 0.6530, 0.4964, 0.5760]])
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
#> tensor([  0.,  15.,  85., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[2., 4.],
#>         [4., 7.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([1.8233])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([0.2733])
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
#> tensor([-0.6285,  1.3181, -1.0566, -0.2009, -1.2667, -0.7597])
print(z)
#> tensor([[-0.6285,  1.3181],
#>         [-1.0566, -0.2009],
#>         [-1.2667, -0.7597]])
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
#> tensor([[ 0.4921,  0.4395, -1.2656],
#>         [-0.0320,  1.3168, -1.7900]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 0.4921,  0.4395, -1.2656],
#>         [-0.0320,  1.3168, -1.7900],
#>         [ 0.4921,  0.4395, -1.2656],
#>         [-0.0320,  1.3168, -1.7900],
#>         [ 0.4921,  0.4395, -1.2656],
#>         [-0.0320,  1.3168, -1.7900]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 0.4921,  0.4395, -1.2656,  0.4921,  0.4395, -1.2656,  0.4921,  0.4395,
#>          -1.2656],
#>         [-0.0320,  1.3168, -1.7900, -0.0320,  1.3168, -1.7900, -0.0320,  1.3168,
#>          -1.7900]])
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

### With decimal incrememets

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
#> tensor([[ 1.0778,  2.6806,  0.0049,  1.2087],
#>         [ 0.5792, -0.8799,  0.1769,  0.5263],
#>         [ 2.1545, -1.4511, -1.3506, -0.2843]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 1.0778,  2.6806,  0.0049,  1.2087],
#>         [ 2.1545, -1.4511, -1.3506, -0.2843]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[ 1.0778,  0.0049],
#>         [ 0.5792,  0.1769],
#>         [ 2.1545, -1.3506]])
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
