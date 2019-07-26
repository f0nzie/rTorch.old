
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
#> tensor([[2.0061e-28, 4.5692e-41, 2.0061e-28],
#>         [4.5692e-41, 8.0254e+17, 1.3556e-19],
#>         [1.3563e-19, 1.3563e-19, 1.2456e-11]])
dt
#> tensor([[1.4659e-296,  4.9116e-90, 1.4043e-309],
#>         [2.4748e-296, 4.8524e-273,  5.9377e-66],
#>         [2.4509e-296, 1.4683e-296,  1.6713e-51]], dtype=torch.float64)
Bt
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [ True, False, False, False,  True]], dtype=torch.bool)
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
#> tensor([[[ 2.0061e-28,  4.5692e-41],
#>          [-1.8283e-31,  3.0858e-41],
#>          [-1.8283e-31,  3.0858e-41]],
#> 
#>         [[ 3.3317e-28,  4.5692e-41],
#>          [ 5.0447e-44,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00]],
#> 
#>         [[ 0.0000e+00,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00]],
#> 
#>         [[ 0.0000e+00,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00],
#>          [ 0.0000e+00,  0.0000e+00]]])
```

``` r
# get first element in a tensor
ft3d[1, 1, 1]
#> tensor(2.0061e-28)
```

``` r
bt
#> tensor([[ 32,  32,  32],
#>         [ 32, 116, 101],
#>         [110, 115, 111]], dtype=torch.uint8)
# [torch.ByteTensor of size 3x3]
```

``` r
ft
#> tensor([[2.0061e-28, 4.5692e-41, 2.0061e-28],
#>         [4.5692e-41, 8.0254e+17, 1.3556e-19],
#>         [1.3563e-19, 1.3563e-19, 1.2456e-11]])
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
#> tensor([[1.0839, 0.9064, 0.6813, 0.9646, 0.2046],
#>         [0.2874, 0.4976, 0.7172, 0.9893, 0.8474],
#>         [1.0875, 0.8817, 0.2295, 0.4622, 0.5312]])
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
#> tensor([[1.0839, 0.9064, 0.6813, 0.9646, 0.2046],
#>         [0.2874, 0.4976, 0.7172, 0.9893, 0.8474],
#>         [1.0875, 0.8817, 0.2295, 0.4622, 0.5312]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.0804, 1.7201, 1.0710, 1.8228],
#>         [1.1242, 0.7680, 1.1740, 1.0665],
#>         [0.4203, 0.9011, 1.2631, 1.5187],
#>         [1.0484, 1.1879, 0.6711, 0.8500],
#>         [0.9293, 0.4384, 0.1748, 0.9582]])
print(x + y)
#> tensor([[1.0804, 1.7201, 1.0710, 1.8228],
#>         [1.1242, 0.7680, 1.1740, 1.0665],
#>         [0.4203, 0.9011, 1.2631, 1.5187],
#>         [1.0484, 1.1879, 0.6711, 0.8500],
#>         [0.9293, 0.4384, 0.1748, 0.9582]])
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
#>           [,1]      [,2]      [,3]       [,4]      [,5]
#> [1,] 0.8145557 0.6733549 0.3844327 0.98476023 0.1041791
#> [2,] 0.7513170 0.4895471 0.9277387 0.99433254 0.3592503
#> [3,] 0.1332810 0.6318775 0.2263621 0.03703317 0.5701774
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
#> [1,] 0.2961283 0.2961283 0.2961283 0.2961283 0.2961283 0.2961283 0.2961283
#> [2,] 0.3522186 0.3522186 0.3522186 0.3522186 0.3522186 0.3522186 0.3522186
#> [3,] 0.1598731 0.1598731 0.1598731 0.1598731 0.1598731 0.1598731 0.1598731
#>           [,8]      [,9]     [,10]
#> [1,] 0.2961283 0.2961283 0.2961283
#> [2,] 0.3522186 0.3522186 0.3522186
#> [3,] 0.1598731 0.1598731 0.1598731
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
#> tensor([0.3140, 0.2500, 0.5210, 0.2053, 0.0533])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.0404, 0.1890, 0.4240, 0.3835, 0.8244], dtype=torch.float64)
```

``` r
# convert tensor to float 16-bits
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.0404, 0.1890, 0.4241, 0.3835, 0.8242], dtype=torch.float16)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 0.0000e+00,  0.0000e+00, -9.8711e-28,  3.0858e-41, -9.8711e-28,
#>           3.0858e-41,  0.0000e+00],
#>         [ 0.0000e+00, -1.4987e-27,  3.0858e-41, -1.4987e-27,  3.0858e-41,
#>          -1.4987e-27,  3.0858e-41],
#>         [ 0.0000e+00,  0.0000e+00,  1.8217e-44,  1.9618e-44,  2.1019e-44,
#>           2.2421e-44,  2.3822e-44],
#>         [ 2.5223e-44,  2.6625e-44,  2.8026e-44, -1.4987e-27,  3.0858e-41,
#>          -1.4987e-27,  3.0858e-41],
#>         [-1.4987e-27,  3.0858e-41,  1.4013e-45,  0.0000e+00,  4.0638e-44,
#>           4.2039e-44,  4.4842e-44]])
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
#> tensor([[-0.0152, -0.9210, -0.7314, -0.2066, -0.9004, -1.3987, -1.1698],
#>         [ 2.5679, -0.1953, -2.0245,  0.1443,  0.9890, -0.9604, -1.1180],
#>         [ 0.4337, -0.0487, -0.0389,  1.2430, -0.5579,  0.0711, -0.2099],
#>         [-0.4682,  0.7232, -1.2404,  1.5204, -1.8637, -1.2061, -0.7214],
#>         [-1.9573, -1.1071,  0.8893, -0.7099,  1.8351, -0.6846, -0.8137]])
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
#> tensor([[0.3171, 0.5590, 0.6175, 0.7671, 0.0630],
#>         [0.4905, 0.2510, 0.4126, 0.2579, 0.4875],
#>         [0.9394, 0.1107, 0.2851, 0.9385, 0.4021]])
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
#> tensor([  0.,  13.,  84., 100.])
```

``` r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[ 1.,  4.],
#>         [ 6., 10.]])
```

### Exponential distribution

``` r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([2.3248])
```

### Weibull distribution

``` r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([0.8735])
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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 0.8538347 0.87358449 0.20427671 0.2890593
#> [2,] 0.3969110 0.50510862 0.13423308 0.1276941
#> [3,] 0.7868708 0.75445191 0.07171152 0.9229554
#> [4,] 0.5113785 0.02812665 0.66926989 0.7256969
#> [5,] 0.8930258 0.45739453 0.32583652 0.6564938
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
#> tensor([ 0.9355,  0.1048, -0.0538, -1.0046,  0.5553,  1.2893])
print(z)
#> tensor([[ 0.9355,  0.1048],
#>         [-0.0538, -1.0046],
#>         [ 0.5553,  1.2893]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[2.8453, 1.2085, 0.8231],
#>         [0.5264, 1.9540, 1.4110]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[2.8453, 1.2085, 0.8231],
#>         [0.5264, 1.9540, 1.4110],
#>         [2.8453, 1.2085, 0.8231],
#>         [0.5264, 1.9540, 1.4110],
#>         [2.8453, 1.2085, 0.8231],
#>         [0.5264, 1.9540, 1.4110]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[2.8453, 1.2085, 0.8231, 2.8453, 1.2085, 0.8231, 2.8453, 1.2085, 0.8231],
#>         [0.5264, 1.9540, 1.4110, 0.5264, 1.9540, 1.4110, 0.5264, 1.9540, 1.4110]])
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
#> tensor([[-0.5561,  0.0926,  0.5210,  0.1659],
#>         [-1.6556,  0.3982, -0.1057, -0.8660],
#>         [-0.8821,  0.4584,  0.9556,  0.2085]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[-0.5561,  0.0926,  0.5210,  0.1659],
#>         [-0.8821,  0.4584,  0.9556,  0.2085]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[-0.5561,  0.5210],
#>         [-1.6556, -0.1057],
#>         [-0.8821,  0.9556]])
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
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
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
# all_boolean <- function(x) {
#   # convert tensor of 1s and 0s to a unique boolean
#   as.logical(torch$all(x)$numpy())
# }

# tensor is less than
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

# is C < A
all(torch$lt(C, A))
#> [1] TRUE
all(C < A)
#> [1] TRUE
# is A < C
all(A < C)
#> [1] FALSE
```

``` r
# tensor is greater than
A <- torch$ones(60000L, 1L, 28L, 28L)
D <- A * 2.0
all(torch$gt(D, A))
#> [1] TRUE
all(torch$gt(A, D))
#> [1] FALSE
```

``` r
# tensor is less than or equal
A1 <- torch$ones(60000L, 1L, 28L, 28L)
all(torch$le(A1, A1))
#> [1] TRUE
all(A1 <= A1)
#> [1] TRUE

# tensor is greater than or equal
A0 <- torch$zeros(60000L, 1L, 28L, 28L)
all(torch$ge(A0, A0))
#> [1] TRUE
all(A0 >= A0)
#> [1] TRUE

all(A1 >= A0)
#> [1] TRUE
all(A1 <= A0)
#> [1] FALSE
```

### Logical NOT

``` r
all_true <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
all_true
#> tensor([True, True, True, True], dtype=torch.bool)

# logical NOT
not_all_true <- !all_true
not_all_true
#> tensor([False, False, False, False], dtype=torch.bool)
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
