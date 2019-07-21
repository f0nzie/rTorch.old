## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
library(rTorch)

Variable   <- import("torch.autograd")$Variable
np         <- import("numpy")
optim      <- import("torch.optim") 
py         <- import_builtins()

## ------------------------------------------------------------------------
# load or download MNIST dataset
mnist <- dataset_mnist(onehot = FALSE)
trX   <- mnist[[1]]; teX = mnist[[2]]; trY = mnist[[3]]; teY = mnist[[4]]

trX <- torch$from_numpy(trX)$float()      # FloatTensor
teX <- torch$from_numpy(teX)$float()      # FloatTensor
trY <- torch$from_numpy(trY)$long()       # LongTensor
teY <- torch$from_numpy(teY)$long()       # LongTensor

## ------------------------------------------------------------------------
# make it reproducible
torch$manual_seed(42L)

# in Python was: n_examples, n_features = trX.size()
# using new R function torch_size()
n_examples    <- torch_size(trX$size())[1]
n_features    <- torch_size(trX$size())[2]

learning_rate <- 0.01
momentum      <- 0.9
n_classes     <- 10L
batch_size    <- 100L
epochs        <- 2        # original value for epochs = 100
neurons       <- 512L

## ------------------------------------------------------------------------
build_model <- function(input_dim, output_dim) {
    model <- torch$nn$Sequential()
    model$add_module("linear_1", torch$nn$Linear(input_dim, neurons, bias = FALSE))
    model$add_module("sigmoid_1", torch$nn$Sigmoid())
    model$add_module("linear_2", torch$nn$Linear(neurons, output_dim, bias = FALSE))
    return(model)
}

train <- function(model, loss, optimizer, x, y) {
    x = Variable(x, requires_grad = FALSE)
    y = Variable(y, requires_grad = FALSE)
    
    # reset gradient
    optimizer$zero_grad()
    
    # forward
    fx     <- model$forward(x)
    output <- loss$forward(fx, y)
    
    # backward
    output$backward()
    
    # update parameters
    optimizer$step()
    
    return(output$data$index(0L))
}

predict <- function(model, x) {
    xvar <-  Variable(x, requires_grad = FALSE)
    output = model$forward(xvar)
    return(np$argmax(output$data, axis = 1L))
}


batching <- function(k) {
    k <- k - 1                             # index in Python start at [0]
    start <- as.integer(k * batch_size)
    end   <- as.integer((k + 1) * batch_size)
    
    cost  <- train(model, loss, optimizer,
                       trX$narrow(0L, start, end-start),
                       trY$narrow(0L, start, end-start))
    
    # allow ccost to accumulate. beware of the <<-
    ccost <<- ccost + cost$numpy()   # because we don't have yet `+` func
    return(list(model = model, cost = ccost))
}


model     <- build_model(n_features, n_classes)
loss      <- torch$nn$CrossEntropyLoss(size_average = TRUE)
optimizer <- optim$SGD(model$parameters(), lr = learning_rate, momentum = momentum)

