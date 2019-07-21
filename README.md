
<!-- README.md is generated from README.Rmd. Please edit that file -->

# rTorch

The goal of rTorch is to …

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
#> tensor([[4.6241e+30, 1.0552e+24, 6.5826e+19],
#>         [1.7565e+25, 1.2120e+25, 6.4600e+19],
#>         [1.8728e+31, 1.9005e-19, 1.1432e+27]])
dt
#> tensor([[6.9207e-310, 6.9207e-310, 4.6396e-310],
#>         [4.6396e-310,  1.6496e+82, 4.0378e+202],
#>         [8.0334e-304, 1.9947e+159, 6.0133e-154]], dtype=torch.float64)
Bt
#> tensor([[ True, False, False, False, False],
#>         [False, False, False,  True,  True],
#>         [ True,  True,  True,  True, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```

A 3D tensor:

``` r
ft3d <- torch$FloatTensor(4L, 3L, 2L)
ft3d
#> tensor([[[1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45]],
#> 
#>         [[1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45]],
#> 
#>         [[1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45]],
#> 
#>         [[1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45],
#>          [1.4013e-45, 1.4013e-45]]])
```

``` r
# get first element in the tensor
ft3d[1, 1, 1]
#> tensor(1.4013e-45)
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
#> tensor([[4.6241e+30, 1.0552e+24, 6.5826e+19],
#>         [1.7565e+25, 1.2120e+25, 6.4600e+19],
#>         [1.8728e+31, 1.9005e-19, 1.1432e+27]])
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
#> tensor([[0.8426, 0.5986, 0.7825, 0.5386, 0.0319],
#>         [0.0267, 0.4988, 0.4100, 0.7180, 0.2770],
#>         [0.5074, 0.8404, 0.3781, 0.9276, 0.5806]])
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[0.9426, 0.6986, 0.8825, 0.6386, 0.1319],
#>         [0.1267, 0.5988, 0.5100, 0.8180, 0.3770],
#>         [0.6074, 0.9404, 0.4781, 1.0276, 0.6806]])
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
#> tensor([[0.9426, 0.6986, 0.8825, 0.6386, 0.1319],
#>         [0.1267, 0.5988, 0.5100, 0.8180, 0.3770],
#>         [0.6074, 0.9404, 0.4781, 1.0276, 0.6806]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.1921, 0.0409, 1.6039, 0.7562],
#>         [0.2946, 1.0969, 0.5855, 0.2313],
#>         [0.2570, 0.5303, 0.9826, 0.9218],
#>         [1.3402, 0.5994, 0.8006, 1.6379],
#>         [1.0287, 0.8580, 1.7787, 1.1186]])
print(x + y)
#> tensor([[1.1921, 0.0409, 1.6039, 0.7562],
#>         [0.2946, 1.0969, 0.5855, 0.2313],
#>         [0.2570, 0.5303, 0.9826, 0.9218],
#>         [1.3402, 0.5994, 0.8006, 1.6379],
#>         [1.0287, 0.8580, 1.7787, 1.1186]])
```

## NumPy and PyTorch

`numpy` has been made available as a module in R. We can call functions
from `numpy` refrerring to it as `np$_a_function`.

``` r

# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>            [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.25069225 0.2648084 0.6890624 0.4255420 0.9372934
#> [2,] 0.49393233 0.6593353 0.9640101 0.5831821 0.9119953
#> [3,] 0.01142465 0.2075041 0.8905141 0.9953501 0.8692165
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
#> [1,] 0.2567398 0.2567398 0.2567398 0.2567398 0.2567398 0.2567398 0.2567398
#> [2,] 0.3612455 0.3612455 0.3612455 0.3612455 0.3612455 0.3612455 0.3612455
#> [3,] 0.2974009 0.2974009 0.2974009 0.2974009 0.2974009 0.2974009 0.2974009
#>           [,8]      [,9]     [,10]
#> [1,] 0.2567398 0.2567398 0.2567398
#> [2,] 0.3612455 0.3612455 0.3612455
#> [3,] 0.2974009 0.2974009 0.2974009
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
#> tensor([0.5197, 0.1911, 0.0670, 0.8066, 0.8497])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.7381, 0.9813, 0.0647, 0.0632, 0.8395], dtype=torch.float64)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 0.0000e+00,  0.0000e+00,  2.8026e-45,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  1.5414e-44,         nan, -5.9346e-32,  3.0638e-41,
#>           1.4013e-45,  0.0000e+00],
#>         [-9.8435e-31,  3.0638e-41,  0.0000e+00,  0.0000e+00,  2.8026e-45,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00]])
```

Initialize a tensor randomized with a normal distribution with mean=0,
var=1:

``` r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[ 0.6402,  0.0923,  0.8651, -1.5228,  1.0050, -0.6329, -0.8516],
#>         [ 0.2033, -1.2978, -0.1110, -1.8614, -2.1776,  0.4630,  0.3526],
#>         [-1.2910, -2.9543, -0.3408, -1.3283,  0.1056,  1.5224,  0.6714],
#>         [ 1.6542,  0.1252,  0.9072, -0.3551, -1.4455,  0.8159, -0.7064],
#>         [ 1.0737,  0.1001, -1.9219, -0.2293,  1.5733,  0.2581,  0.2830]])
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
#> [1,] 0.5348180 0.7128507 0.57026316 0.4276033
#> [2,] 0.7034598 0.6940123 0.20835836 0.9909361
#> [3,] 0.5951309 0.6611798 0.99138779 0.9055337
#> [4,] 0.6788318 0.5276599 0.03416215 0.4388200
#> [5,] 0.4267013 0.2611167 0.09646742 0.6675985
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
#> tensor([ 0.0563,  0.1190,  0.0966, -0.6201, -0.7275,  0.3240])
print(z)
#> tensor([[ 0.0563,  0.1190],
#>         [ 0.0966, -0.6201],
#>         [-0.7275,  0.3240]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[ 0.2066,  0.8162, -0.7448],
#>         [-1.0551, -2.2986,  0.9924]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 0.2066,  0.8162, -0.7448],
#>         [-1.0551, -2.2986,  0.9924],
#>         [ 0.2066,  0.8162, -0.7448],
#>         [-1.0551, -2.2986,  0.9924],
#>         [ 0.2066,  0.8162, -0.7448],
#>         [-1.0551, -2.2986,  0.9924]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 0.2066,  0.8162, -0.7448,  0.2066,  0.8162, -0.7448,  0.2066,  0.8162,
#>          -0.7448],
#>         [-1.0551, -2.2986,  0.9924, -1.0551, -2.2986,  0.9924, -1.0551, -2.2986,
#>           0.9924]])
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
#> tensor([[-1.1306,  1.2503, -1.3821, -0.0909],
#>         [-0.9653,  0.1755,  0.1561, -0.0658],
#>         [ 0.2166,  1.1579, -1.3451, -0.5843]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[-1.1306,  1.2503, -1.3821, -0.0909],
#>         [ 0.2166,  1.1579, -1.3451, -0.5843]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[-1.1306, -1.3821],
#>         [-0.9653,  0.1561],
#>         [ 0.2166, -1.3451]])
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
