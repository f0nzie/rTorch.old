
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rTorch

The goal of rTorch is providing an R wrapper to PyTorch. We have
borrowed ideas and code used in R `tensorflow` to implement `rTorch`.

Besides the module `torch`, which provides methods, classes and
function, we are also providing numpy as `np` and `torchvision` as well.
The dollar sign after the module will provide access to their objects.

## Installation

`rTorch` is available in GitHub only at this moment.

Install rTorch with: `devtools::install_github("f0nzie/rTorch")`

Before start running the `rTorch`, install a Python Anaconda first.
Then:

1.  Create a conda environment with `conda create -n myenv python=3.7`

2.  Activate the new environment with `conda activate myenv`

3.  Install PyTorch packages with:  
    `conda install python=3.6.6 pytorch-cpu torchvision-cpu matplotlib
    pandas -c pytorch`

Now, you can load `rTorch`.

The automatic installation, like in Tensorflow, may be available later.

## Matrices and Linear Algebra

There are four major type of Tensors in PyTorch

``` r
library(rTorch)

bt <- torch$ByteTensor(3L, 3L)
ft <- torch$FloatTensor(3L, 3L)
dt <- torch$DoubleTensor(3L, 3L)
lt <- torch$LongTensor(3L, 3L)
Bt <- torch$BoolTensor(5L, 5L)

ft
#> tensor([[1.2120e+25, 1.3556e-19, 1.8567e-01],
#>         [1.9492e-19, 7.5553e+28, 5.2839e-11],
#>         [1.7589e+22, 2.5038e-12, 1.1362e+30]])
dt
#> tensor([[6.9038e-310, 4.6731e-310, 4.6731e-310],
#>         [4.6731e-310, 2.5517e+151, 3.7027e-297],
#>         [3.1315e-294,  5.6295e+14, 5.1881e-313]], dtype=torch.float64)
Bt
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```

A 3D tensor:

``` r
ft3d <- torch$FloatTensor(4L, 3L, 2L)
ft3d
#> tensor([[[5.3673e+30, 4.5590e-41],
#>          [2.9671e-22, 3.0859e-41],
#>          [3.0914e-22, 3.0859e-41]],
#> 
#>         [[8.7292e+30, 4.5590e-41],
#>          [5.0447e-44, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]]])
```

``` r
# get first element in the tensor
ft3d[1, 1, 1]
#> tensor(5.3673e+30)
```

``` r
bt
#> tensor([[160, 124, 135],
#>         [114,  22, 127],
#>         [  0,   0, 160]], dtype=torch.uint8)
# [torch.ByteTensor of size 3x3]
```

``` r
ft
#> tensor([[1.2120e+25, 1.3556e-19, 1.8567e-01],
#>         [1.9492e-19, 7.5553e+28, 5.2839e-11],
#>         [1.7589e+22, 2.5038e-12, 1.1362e+30]])
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
#> tensor([[0.4277, 0.2211, 0.3817, 0.4889, 0.3291],
#>         [0.4027, 0.1602, 0.6137, 0.9191, 0.4289],
#>         [0.8782, 0.7294, 0.7372, 0.5336, 0.0336]])
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[0.5277, 0.3211, 0.4817, 0.5889, 0.4291],
#>         [0.5027, 0.2602, 0.7137, 1.0191, 0.5289],
#>         [0.9782, 0.8294, 0.8372, 0.6336, 0.1336]])
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
#> tensor([[0.5277, 0.3211, 0.4817, 0.5889, 0.4291],
#>         [0.5027, 0.2602, 0.7137, 1.0191, 0.5289],
#>         [0.9782, 0.8294, 0.8372, 0.6336, 0.1336]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.0354, 0.5581, 1.7790, 0.9795],
#>         [1.0373, 1.0907, 1.3264, 0.7812],
#>         [1.0063, 0.6625, 0.2921, 0.6206],
#>         [0.7858, 0.3652, 1.0501, 1.1220],
#>         [0.5572, 1.8915, 1.6525, 1.0929]])
print(x + y)
#> tensor([[1.0354, 0.5581, 1.7790, 0.9795],
#>         [1.0373, 1.0907, 1.3264, 0.7812],
#>         [1.0063, 0.6625, 0.2921, 0.6206],
#>         [0.7858, 0.3652, 1.0501, 1.1220],
#>         [0.5572, 1.8915, 1.6525, 1.0929]])
```

## NumPy and PyTorch

`numpy` has been made available as a module in R. We can call functions
from `numpy` refrerring to it as `np$_a_function`.

``` r

# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.6745745 0.3451708 0.7902848 0.7492494 0.8262522
#> [2,] 0.3799743 0.1277246 0.7412834 0.4173623 0.2784184
#> [3,] 0.6971719 0.4268406 0.8009821 0.8948375 0.9444909
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
#> [1,] 0.3385532 0.3385532 0.3385532 0.3385532 0.3385532 0.3385532 0.3385532
#> [2,] 0.1944763 0.1944763 0.1944763 0.1944763 0.1944763 0.1944763 0.1944763
#> [3,] 0.3764323 0.3764323 0.3764323 0.3764323 0.3764323 0.3764323 0.3764323
#>           [,8]      [,9]     [,10]
#> [1,] 0.3385532 0.3385532 0.3385532
#> [2,] 0.1944763 0.1944763 0.1944763
#> [3,] 0.3764323 0.3764323 0.3764323
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
#> tensor([0.5083, 0.8065, 0.6555, 0.7190, 0.3952])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.2256, 0.5068, 0.2667, 0.0901, 0.5687], dtype=torch.float64)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[1.0503e-19, 3.0859e-41, 8.7292e+30, 4.5590e-41, 7.1466e-44, 0.0000e+00,
#>          0.0000e+00],
#>         [0.0000e+00, 1.5414e-44, 3.7835e-44, 1.4013e-45, 2.8026e-44, 4.2039e-45,
#>          4.6243e-44],
#>         [5.4651e-44, 5.6052e-45, 3.0829e-44, 7.0065e-45, 5.6052e-45, 2.8026e-44,
#>          1.1210e-44],
#>         [1.2331e-43, 9.8091e-45, 4.2039e-44, 3.6434e-44, 1.2612e-44, 2.8026e-44,
#>          4.2039e-45],
#>         [4.6243e-44, 5.4651e-44, 1.4013e-44, 3.0829e-44, 1.5414e-44, 1.0510e-43,
#>          8.2677e-44]])
```

Initialize a tensor randomized with a normal distribution with mean=0,
var=1:

``` r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[ 0.3665, -0.7369,  0.7525,  0.0826,  0.4183, -0.5149,  0.0156],
#>         [ 1.3433,  0.3450,  0.0565, -0.6167, -1.4804, -0.4524, -1.7806],
#>         [-0.8994, -1.9612,  0.9508, -0.4930,  0.3138,  0.4446, -0.1251],
#>         [-0.3616, -0.6571, -2.8227,  0.7759,  0.5943,  0.0195,  2.0955],
#>         [-1.4006, -0.5770,  0.7562,  1.7071,  1.0787, -1.5930,  1.1139]])
print(a$size())
#> torch.Size([5, 7])
```

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

``` r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>           [,1]      [,2]       [,3]      [,4]
#> [1,] 0.3195799 0.3900589 0.21219825 0.4646301
#> [2,] 0.7049406 0.4321013 0.04386786 0.2194034
#> [3,] 0.7734179 0.2338938 0.88183137 0.3054772
#> [4,] 0.6134444 0.1613261 0.95275420 0.4343568
#> [5,] 0.2641842 0.8155563 0.82914579 0.7011797
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
#> tensor([-1.8049,  2.3365, -0.2426, -2.2770,  1.6500,  0.6187])
print(z)
#> tensor([[-1.8049,  2.3365],
#>         [-0.2426, -2.2770],
#>         [ 1.6500,  0.6187]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[-0.7319, -0.1195,  0.8836],
#>         [-0.1058,  1.4148, -0.8342]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[-0.7319, -0.1195,  0.8836],
#>         [-0.1058,  1.4148, -0.8342],
#>         [-0.7319, -0.1195,  0.8836],
#>         [-0.1058,  1.4148, -0.8342],
#>         [-0.7319, -0.1195,  0.8836],
#>         [-0.1058,  1.4148, -0.8342]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[-0.7319, -0.1195,  0.8836, -0.7319, -0.1195,  0.8836, -0.7319, -0.1195,
#>           0.8836],
#>         [-0.1058,  1.4148, -0.8342, -0.1058,  1.4148, -0.8342, -0.1058,  1.4148,
#>          -0.8342]])
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
#> tensor([[ 0.1070,  0.6690,  0.2102, -1.0013],
#>         [-1.3740, -0.0609, -0.7966, -0.4999],
#>         [ 0.5656, -2.7573,  0.4045, -1.0029]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 0.1070,  0.6690,  0.2102, -1.0013],
#>         [ 0.5656, -2.7573,  0.4045, -1.0029]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[ 0.1070,  0.2102],
#>         [-1.3740, -0.7966],
#>         [ 0.5656,  0.4045]])
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
