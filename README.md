
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rTorch

The goal of `rTorch` is providing an R wrapper to
\[PyTorch\](<https://pytorch.org/>. We have borrowed ideas and code used
in R [tensorflow](https://github.com/rstudio/tensorflow) to implement
`rTorch`.

Besides the module `torch`, which provides `PyTorch` methods, classes
and functions, the package also provides `numpy` as a method called
`np`, and `torchvision`, as well. The dollar sign `$` after the module
will provide you access to those objects.

## Installation

`rTorch` is available in GitHub only at this moment.

Install `rTorch` with:

`devtools::install_github("f0nzie/rTorch")`

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

**Note.** `matplotlib` and `pandas` are not really necessary, but I was
asked if `matplotlib` or `pandas` would in PyTorch, that I decided to
put them for testing and experimentation. They both work.

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
#> tensor([[2.6033e-12, 4.0058e-11, 2.6257e-06],
#>         [2.5193e-09, 2.5930e-09, 8.2990e-33],
#>         [1.3563e-19, 1.3563e-19, 1.3563e-19]])
dt
#> tensor([[ 1.9352e+26, 1.4043e-309, 2.4891e-296],
#>         [4.8524e-273,  2.3396e+50, 2.4509e-296],
#>         [1.4754e-296,  6.5853e+64, 1.4043e-309]], dtype=torch.float64)
Bt
#> tensor([[ True,  True,  True, False,  True],
#>         [ True, False, False,  True,  True],
#>         [ True, False,  True,  True, False],
#>         [False,  True, False, False, False],
#>         [False, False, False, False,  True]], dtype=torch.bool)
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
#> tensor([[[ 1.0440e-38,  4.5869e-41],
#>          [-1.6367e-18,  3.0694e-41],
#>          [-1.6367e-18,  3.0694e-41]],
#> 
#>         [[ 1.8233e-38,  4.5869e-41],
#>          [ 5.0447e-44,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00]],
#> 
#>         [[ 1.4013e-45,  1.4013e-45],
#>          [ 1.4013e-45,  1.4013e-45],
#>          [ 1.4013e-45,  1.4013e-45]],
#> 
#>         [[ 1.4013e-45,  1.4013e-45],
#>          [ 1.4013e-45,  1.4013e-45],
#>          [ 1.4013e-45,  1.4013e-45]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(1.0440e-38)
```

``` r
bt
#> tensor([[ 32, 116, 111],
#>         [ 32, 115, 101],
#>         [108, 101,  99]], dtype=torch.uint8)
# [torch.ByteTensor of size 3x3]
```

``` r
ft
#> tensor([[2.6033e-12, 4.0058e-11, 2.6257e-06],
#>         [2.5193e-09, 2.5930e-09, 8.2990e-33],
#>         [1.3563e-19, 1.3563e-19, 1.3563e-19]])
# [torch.FloatTensor of size 3x3]
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
# add a scalar to a tensor
# 3x5 matrix uniformly distributed between 0 and 1
mat0 <- torch$FloatTensor(3L, 5L)$uniform_(0L, 1L)
mat0 + 0.1
#> tensor([[0.3678, 0.4165, 1.0787, 0.4303, 0.7644],
#>         [0.5709, 0.2592, 0.8960, 0.5213, 0.4316],
#>         [0.5281, 0.7360, 0.9107, 0.1881, 0.1969]])
```

> The expression `tensor.index(m)` is equivalent to `tensor[m]`.

``` r
# add an element of tensor to a tensor
# fill a 3x5 matrix with 0.1
mat1 <- torch$FloatTensor(3L, 5L)$uniform_(0.1, 0.1)
# a vector with all ones
mat2 <- torch$FloatTensor(5L)$uniform_(1, 1)
mat1[1, 1] + mat2
#> tensor([1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
```

``` r
# add two tensors
mat1 + mat0
#> tensor([[0.3678, 0.4165, 1.0787, 0.4303, 0.7644],
#>         [0.5709, 0.2592, 0.8960, 0.5213, 0.4316],
#>         [0.5281, 0.7360, 0.9107, 0.1881, 0.1969]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[0.5827, 1.3528, 0.3879, 1.0177],
#>         [0.8920, 0.4784, 0.7958, 1.0434],
#>         [1.5189, 0.5560, 0.2093, 1.1601],
#>         [1.2766, 1.1187, 0.7862, 1.1416],
#>         [1.0569, 1.5518, 0.9173, 1.2216]])
print(x + y)
#> tensor([[0.5827, 1.3528, 0.3879, 1.0177],
#>         [0.8920, 0.4784, 0.7958, 1.0434],
#>         [1.5189, 0.5560, 0.2093, 1.1601],
#>         [1.2766, 1.1187, 0.7862, 1.1416],
#>         [1.0569, 1.5518, 0.9173, 1.2216]])
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
#>           [,1]      [,2]       [,3]      [,4]       [,5]
#> [1,] 0.1439174 0.4468810 0.62225590 0.1503744 0.60251373
#> [2,] 0.3713924 0.3805039 0.37274455 0.3855800 0.07768096
#> [3,] 0.9684601 0.5616011 0.09965187 0.6452330 0.35620753
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
#> [1,] 0.1965942 0.1965942 0.1965942 0.1965942 0.1965942 0.1965942 0.1965942
#> [2,] 0.1587902 0.1587902 0.1587902 0.1587902 0.1587902 0.1587902 0.1587902
#> [3,] 0.2631154 0.2631154 0.2631154 0.2631154 0.2631154 0.2631154 0.2631154
#>           [,8]      [,9]     [,10]
#> [1,] 0.1965942 0.1965942 0.1965942
#> [2,] 0.1587902 0.1587902 0.1587902
#> [3,] 0.2631154 0.2631154 0.2631154
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
#> tensor([0.6866, 0.6993, 0.4760, 0.1691, 0.8748])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.2453, 0.7374, 0.0284, 0.8671, 0.0182], dtype=torch.float64)
```

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.2452, 0.7373, 0.0284, 0.8672, 0.0182], dtype=torch.float16)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[-1.4278e-12,  3.0694e-41,  0.0000e+00,  0.0000e+00,  1.4013e-45,
#>           1.4013e-45,  1.4013e-45],
#>         [ 1.4013e-45,  1.4013e-45,  1.4013e-45,  1.4013e-45,  1.4013e-45,
#>          -1.4278e-12,  3.0694e-41],
#>         [-1.4278e-12,  3.0694e-41, -1.4278e-12,  3.0694e-41,  1.4013e-45,
#>           0.0000e+00,  1.4013e-45],
#>         [ 1.4013e-45,  0.0000e+00,  1.4013e-45,  1.4013e-45,  1.4013e-45,
#>           1.4013e-45,  1.4013e-45],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -7.0809e+18,
#>           4.5867e-41,  1.4013e-45]])
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
#> tensor([[ 0.7143, -0.6344,  1.6618,  1.0122,  0.7928,  1.5777,  0.0562],
#>         [-0.5508, -0.0142,  0.1620, -0.1602, -0.6033,  0.1212, -2.8524],
#>         [ 0.9932,  1.2865, -0.3777, -0.9227, -1.1836,  0.1720, -0.5065],
#>         [-0.0029, -0.0295,  0.1739, -0.9623, -0.9255,  1.4831, -1.1404],
#>         [ 0.9300, -0.2455, -0.7558, -1.0228,  1.1187, -0.8432,  0.4857]])
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
#> tensor([[0.4048, 0.2752, 0.8718, 0.8433, 0.5080],
#>         [0.0794, 0.2943, 0.2788, 0.7415, 0.1813],
#>         [0.8469, 0.2618, 0.4419, 0.1241, 0.7221]])
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
#> tensor([  0.,  23.,  86., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[4., 4.],
#>         [3., 9.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([1.4085])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([0.4595])
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
#>           [,1]      [,2]      [,3]      [,4]
#> [1,] 0.7926896 0.3663375 0.4548309 0.3566701
#> [2,] 0.2202458 0.7451641 0.1623685 0.4005885
#> [3,] 0.3303923 0.4294647 0.9481562 0.4425483
#> [4,] 0.2784317 0.8438489 0.1080590 0.7240599
#> [5,] 0.3879938 0.0954060 0.7586034 0.1896458
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
#> tensor([-0.5623, -0.2793, -1.3963,  0.8381, -0.1313,  1.0144])
print(z)
#> tensor([[-0.5623, -0.2793],
#>         [-1.3963,  0.8381],
#>         [-0.1313,  1.0144]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[ 0.1506, -0.5918,  0.6749],
#>         [-0.0084,  0.0291,  1.5142]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 0.1506, -0.5918,  0.6749],
#>         [-0.0084,  0.0291,  1.5142],
#>         [ 0.1506, -0.5918,  0.6749],
#>         [-0.0084,  0.0291,  1.5142],
#>         [ 0.1506, -0.5918,  0.6749],
#>         [-0.0084,  0.0291,  1.5142]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 0.1506, -0.5918,  0.6749,  0.1506, -0.5918,  0.6749,  0.1506, -0.5918,
#>           0.6749],
#>         [-0.0084,  0.0291,  1.5142, -0.0084,  0.0291,  1.5142, -0.0084,  0.0291,
#>           1.5142]])
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
#> tensor([[ 1.3260, -1.2029, -1.4712,  0.3896],
#>         [ 1.2002, -1.6276,  0.1444, -0.4902],
#>         [-0.3415,  0.0786, -0.2730,  0.3593]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 1.3260, -1.2029, -1.4712,  0.3896],
#>         [-0.3415,  0.0786, -0.2730,  0.3593]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[ 1.3260, -1.4712],
#>         [ 1.2002,  0.1444],
#>         [-0.3415, -0.2730]])
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

## Logical operations

``` r
m0 = torch$zeros(3L, 5L)
m1 = torch$ones(3L, 5L)
m2 = torch$eye(3L, 5L)

print(m1 == m0)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```

``` r
print(m1 != m1)
#> tensor([[0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0],
#>         [0, 0, 0, 0, 0]], dtype=torch.uint8)
```

``` r
print(m2 == m2)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```

``` r
# AND
m1 & m1
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```

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

``` r
all_boolean <- function(x) {
  # convert tensor of 1s and 0s to a unique boolean
  as.logical(torch$all(x)$numpy())
}

# tensor is less than
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

# is C < A
all_boolean(torch$lt(C, A))
#> [1] TRUE
all_boolean(C < A)
#> [1] TRUE
# is A < C
all_boolean(A < C)
#> [1] FALSE
```

``` r
# tensor is greater than
A <- torch$ones(60000L, 1L, 28L, 28L)
D <- A * 2.0
all_boolean(torch$gt(D, A))
#> [1] TRUE
all_boolean(torch$gt(A, D))
#> [1] FALSE
```

``` r
# tensor is less than or equal
A1 <- torch$ones(60000L, 1L, 28L, 28L)
all_boolean(torch$le(A1, A1))
#> [1] TRUE
all_boolean(A1 <= A1)
#> [1] TRUE

# tensor is greater than or equal
A0 <- torch$zeros(60000L, 1L, 28L, 28L)
all_boolean(torch$ge(A0, A0))
#> [1] TRUE
all_boolean(A0 >= A0)
#> [1] TRUE

all_boolean(A1 >= A0)
#> [1] TRUE
all_boolean(A1 <= A0)
#> [1] FALSE
```

### Logical NOT

``` r
lo <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
lo
#> tensor([True, True, True, True], dtype=torch.bool)

# logical NOT
!lo
#> tensor([False, False, False, False], dtype=torch.bool)
```

``` r
d <- torch$eye(5L)
d
#> tensor([[1., 0., 0., 0., 0.],
#>         [0., 1., 0., 0., 0.],
#>         [0., 0., 1., 0., 0.],
#>         [0., 0., 0., 1., 0.],
#>         [0., 0., 0., 0., 1.]])

# logical NOT
!d
#> tensor([[False,  True,  True,  True,  True],
#>         [ True, False,  True,  True,  True],
#>         [ True,  True, False,  True,  True],
#>         [ True,  True,  True, False,  True],
#>         [ True,  True,  True,  True, False]], dtype=torch.bool)
```
