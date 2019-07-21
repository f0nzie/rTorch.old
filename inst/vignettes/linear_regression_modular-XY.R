## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
library(rTorch)

torch      <- import("torch")
Variable   <- import("torch.autograd")$Variable
np         <- import("numpy")
optim      <- import("torch.optim") 
py         <- import_builtins()

