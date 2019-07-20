
source("tests/testthat/tensor_functions.R")

m1 = torch$ones(3L, 5L)
m1$shape

tensor_dim_(m1)
# [1] 2
tensor_dim(m1)
# [1] 3 5

library(magrittr)
m1 %>%
  as.array()
