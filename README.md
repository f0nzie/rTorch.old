
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
#> tensor([[6.9285e-310, 6.9285e-310, 4.6365e-310],
#>         [4.6365e-310,  1.6496e+82, 4.0378e+202],
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
# get first element in the tensor
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
#> tensor([[0.4824, 0.8834, 0.0670, 0.6708, 0.4605],
#>         [0.2080, 0.2750, 0.3855, 0.6211, 0.2739],
#>         [0.6159, 0.5168, 0.8544, 0.7852, 0.7750]])
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]])
```

``` r
# add a scalar to a tensor
mat0 + 0.1
#> tensor([[0.5824, 0.9834, 0.1670, 0.7708, 0.5605],
#>         [0.3080, 0.3750, 0.4855, 0.7211, 0.3739],
#>         [0.7159, 0.6168, 0.9544, 0.8852, 0.8750]])
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
#> tensor([[0.5824, 0.9834, 0.1670, 0.7708, 0.5605],
#>         [0.3080, 0.3750, 0.4855, 0.7211, 0.3739],
#>         [0.7159, 0.6168, 0.9544, 0.8852, 0.8750]])
```

``` r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[1.3619, 1.2379, 1.5480, 0.6477],
#>         [1.3740, 1.2082, 0.7725, 0.9368],
#>         [0.5728, 0.6864, 0.8105, 0.7118],
#>         [0.9297, 0.8585, 1.2465, 0.8521],
#>         [1.4437, 1.0823, 1.2867, 0.7754]])
print(x + y)
#> tensor([[1.3619, 1.2379, 1.5480, 0.6477],
#>         [1.3740, 1.2082, 0.7725, 0.9368],
#>         [0.5728, 0.6864, 0.8105, 0.7118],
#>         [0.9297, 0.8585, 1.2465, 0.8521],
#>         [1.4437, 1.0823, 1.2867, 0.7754]])
```

## NumPy and PyTorch

`numpy` has been made available as a module in R. We can call functions
from `numpy` refrerring to it as `np$_a_function`.

``` r

# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
syn0
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.3372018 0.4780166 0.0507536 0.2212633 0.2668096
#> [2,] 0.7794283 0.4945991 0.5793476 0.1200084 0.5611083
#> [3,] 0.8985207 0.3932445 0.7112183 0.6469191 0.5202781
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
#> [1,] 0.1354045 0.1354045 0.1354045 0.1354045 0.1354045 0.1354045 0.1354045
#> [2,] 0.2534492 0.2534492 0.2534492 0.2534492 0.2534492 0.2534492 0.2534492
#> [3,] 0.3170181 0.3170181 0.3170181 0.3170181 0.3170181 0.3170181 0.3170181
#>           [,8]      [,9]     [,10]
#> [1,] 0.1354045 0.1354045 0.1354045
#> [2,] 0.2534492 0.2534492 0.2534492
#> [3,] 0.3170181 0.3170181 0.3170181
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
#> tensor([0.0607, 0.0354, 0.4365, 0.1912, 0.2755])
```

``` r
# tensor as a float of 64-bits
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
ft2
#> tensor([0.3604, 0.5951, 0.7564, 0.4025, 0.9547], dtype=torch.float64)
```

Create a tensor of size (5 x 7) with uninitialized memory:

``` r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00, -6.6126e+06],
#>         [ 3.0617e-41,  0.0000e+00,  0.0000e+00, -3.1916e+07,  3.0617e-41,
#>           0.0000e+00,  0.0000e+00],
#>         [ 1.4013e-45,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00, -6.6126e+06,  3.0617e-41,  0.0000e+00,  0.0000e+00,
#>          -3.8243e+07,  3.0617e-41],
#>         [ 0.0000e+00,  0.0000e+00,  1.4013e-45,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00]])
```

Initialize a tensor randomized with a normal distribution with mean=0,
var=1:

``` r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[-0.6857,  1.8998,  1.2109,  0.8839, -0.3856, -0.0515, -0.8038],
#>         [ 0.5220,  0.6832,  0.2354,  0.6999, -1.3065, -0.4856, -1.3040],
#>         [-0.5699, -0.6044, -0.1927, -0.5193, -0.5097,  1.5697, -0.9093],
#>         [-0.1169,  2.1797,  0.0131,  0.4660,  1.5612,  1.5282,  0.6207],
#>         [-1.1846,  0.7288,  1.2124,  0.1413,  0.9937,  0.5898, -0.9183]])
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
#>           [,1]       [,2]       [,3]      [,4]
#> [1,] 0.1798993 0.32108961 0.41064204 0.6102033
#> [2,] 0.5648966 0.19569490 0.25037534 0.4306042
#> [3,] 0.3134462 0.03026954 0.02128372 0.7419126
#> [4,] 0.2437470 0.40620183 0.21900641 0.3241785
#> [5,] 0.1259210 0.03505706 0.01479204 0.7328812
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
#> tensor([ 1.7728,  0.1919, -0.7039, -1.6760,  0.5264, -1.9424])
print(z)
#> tensor([[ 1.7728,  0.1919],
#>         [-0.7039, -1.6760],
#>         [ 0.5264, -1.9424]])
```

### concatenate tensors

``` r
# concatenate tensors
x = torch$randn(2L, 3L)
print(x)
#> tensor([[ 0.8237, -0.7180,  2.7661],
#>         [-0.3161, -2.8480,  1.2003]])

# concatenate tensors by dim=0"
torch$cat(list(x, x, x), 0L)
#> tensor([[ 0.8237, -0.7180,  2.7661],
#>         [-0.3161, -2.8480,  1.2003],
#>         [ 0.8237, -0.7180,  2.7661],
#>         [-0.3161, -2.8480,  1.2003],
#>         [ 0.8237, -0.7180,  2.7661],
#>         [-0.3161, -2.8480,  1.2003]])

# concatenate tensors by dim=1
torch$cat(list(x, x, x), 1L)
#> tensor([[ 0.8237, -0.7180,  2.7661,  0.8237, -0.7180,  2.7661,  0.8237, -0.7180,
#>           2.7661],
#>         [-0.3161, -2.8480,  1.2003, -0.3161, -2.8480,  1.2003, -0.3161, -2.8480,
#>           1.2003]])
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
#> tensor([[-0.7649, -0.7077,  0.7372, -0.7294],
#>         [-0.3859, -0.3046,  0.6638,  0.6001],
#>         [-0.4010,  0.6676,  0.1977, -0.3313]])

# Select indices, dim=0
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[-0.7649, -0.7077,  0.7372, -0.7294],
#>         [-0.4010,  0.6676,  0.1977, -0.3313]])

# "Select indices, dim=1
torch$index_select(x, 1L, indices)
#> tensor([[-0.7649,  0.7372],
#>         [-0.3859,  0.6638],
#>         [-0.4010,  0.1977]])
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
