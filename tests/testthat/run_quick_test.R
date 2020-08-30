
source("tests/testthat/tensor_functions.R")

m1 = torch$ones(3L, 5L)
m1$shape

tensor_ndim(m1)
# [1] 2
tensor_dim(m1)
# [1] 3 5

# library(magrittr)
# m1 %>%
#   as.array()

all_boolean <- function(x) {
  # convert tensor of 1s and 0s to a unique boolean
  as.logical(torch$all(x)$numpy())
}


# testing that new generic all, any for tensors work
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5
all(torch$lt(C, A))
all(C < A)
all(A < C)


all(torch$tensor(list(1, 1, 1)))

all(torch$tensor(list(1, 1, 0)))

any(torch$tensor(list(1, 1, 0)))

any(torch$tensor(list(0, 0, 0)))

all(torch$eye(3L))
any(torch$eye(3L))
