library(testthat)

source("tensor_functions.R")



# folders where te images are located
train_data_path = '~/mnist_png_full/training/'
test_data_path  = '~/mnist_png_full/testing/'

# read the datasets without normalization or corrections; as-is
train_dataset <<-  torchvision$datasets$ImageFolder(
    root = train_data_path,
    transform = torchvision$transforms$ToTensor()
)

test_dataset = torchvision$datasets$ImageFolder(
    root = test_data_path,
    transform = torchvision$transforms$ToTensor()
)


context("methods of train_dataset")

test_that("names of methods", {
    expect_equal(names(train_dataset),
    c("class_to_idx", "classes", "extensions", "extra_repr",
      "imgs", "loader", "root", "samples", "target_transform",
      "targets", "transform", "transforms"))
})


test_that("attributes of train_dataset", {

    expect_true(all(py_list_attributes(train_dataset) %in% c(
    "class_to_idx", "classes", "extensions", "extra_repr",
    "imgs", "loader", "root", "samples", "target_transform",
    "targets", "transform", "transforms",

    "__add__", "__class__", "__delattr__",
    "__dict__","__dir__", "__doc__",
    "__eq__","__format__", "__ge__",
    "__getattribute__", "__getitem__", "__gt__",
    "__hash__", "__init__", "__init_subclass__",
    "__le__", "__len__", "__lt__",
    "__module__", "__ne__", "__new__",
    "__reduce__", "__reduce_ex__", "__repr__",
    "__setattr__", "__sizeof__", "__str__",
    "__subclasshook__", "__weakref__", "_find_classes",
    "_format_transform_repr", "_repr_indent"))
    )
})

test_that("train_dataset has __getitem()__ method", {
    expect_error(train_dataset[0])
    expect_error(train_dataset[[0]])
    result <- train_dataset$`__getitem__`(0L)
    expect_equal(length(result), 2)
})

test_that("train_dataset returns a list of 2 elements with py_get_item()", {
    result <- py_get_item(train_dataset, 0L)
    expect_equal(class(result), "list")
    expect_equal(length(result), 2)
})

test_that("train_dataset is a tuple in Python", {
    expect_equal(as.character(py_eval("type(r.train_dataset[0])")),
                 "<class 'tuple'>")
})

test_that("1st member of train_dataset list is a tensor: image", {
    result <- py_get_item(train_dataset, 0L)[[1]]
    expect_true(is_tensor(result))
})

test_that("2nd member of train_dataset list is an integer: label", {
    result <- py_get_item(train_dataset, 0L)[[2]]
    expect_true(is.integer(result))
})

test_that("train_dataset is a Python tuple object but an R list", {
    result <- train_dataset$`__getitem__`(0L)
    expect_equal(class(result), "list")
    expect_equal(as.character(py_eval("type(r.train_dataset[0])")),
                 "<class 'tuple'>")
})

test_that("dimension of the tensor is 3D", {
    result <- py_get_item(train_dataset, 0L)[[1]]
    expect_equal(tensor_dim_(result), 3)
})

test_that("dimensions of the tensor is 3x28x28", {
    result <- py_get_item(train_dataset, 0L)[[1]]
    expect_equal(tensor_dim(result), c(3, 28, 28))
})

test_that("length of the train_dataset is 60000", {
    result <- py_len(train_dataset)
    expect_equal(result, 60000)
})

test_that("length of the test_dataset is 10000", {
  result <- py_len(test_dataset)
  expect_equal(result, 10000)
})

test_that("1st label of the train_dataset is 0", {
  result <- py_get_item(train_dataset, 0L)[[2]]
  expect_equal(result, 0)
  result <- py_get_item(train_dataset, 59999L)[[2]]
})

test_that("59999th label of the train_dataset is 9", {
  result <- py_get_item(train_dataset, 59999L)[[2]]
  expect_equal(result, 9)
})

test_that("last label of the train_dataset is 9", {
  result <- py_get_item(train_dataset, py_len(train_dataset)-1L)[[2]]
  expect_equal(result, 9)
})

test_that("last label of the train_dataset using py_object_last()", {
  result <- py_get_item(train_dataset, py_object_last(train_dataset))[[2]]
  expect_equal(result, 9)
})

test_that("train and test datasets have __len__ method in Python", {
  expect_true(py_has_length(train_dataset))
  expect_true(py_has_length(test_dataset))
})

test_that("these objects do not have Python __len__", {
  expect_true(py_has_length(py_get_item(train_dataset, 0L)[[1]]))
  expect_false(py_has_length(py_get_item(train_dataset, 0L)[[2]]))
  expect_false(py_has_length(py_get_item(train_dataset, 0L)))
  result <-  r_to_py(py_get_item(train_dataset, 0L)[[2]])
  expect_equal(as.character(result$`__class__`), "<class 'int'>")
})

test_that("dataset label is of type integer", {
  expect_true(is.integer(py_get_item(train_dataset, 0L)[[2]]))
  result <-  r_to_py(py_get_item(train_dataset, 0L)[[2]])
  expect_equal(as.character(result$`__class__`), "<class 'int'>")
})
