library(testthat)

skip_on_cran()

context("parse_torch_version()")
test_that("default, cpu, stable", {
    version <- "default"
    ver <- parse_torch_version(version)
    expect_equal(ver$version, "1.4")
    expect_equal(ver$package, "pytorch==1.4")
    expect_equal(ver$cpu_gpu_packages, "cpuonly")
    expect_equal(ver$channel, "pytorch")
})

test_that("1.6, cpu, nightly", {
    version <- "1.6"
    ver <- parse_torch_version(version)
    expect_equal(ver$version, "1.6")
    expect_equal(ver$package, "pytorch==1.6")
    expect_equal(ver$cpu_gpu_packages, "cpuonly")
    expect_equal(ver$channel, "pytorch")
})


test_that("default, gpu, stable", {
    version <- "default"
    ver <- parse_torch_version(version, cuda_version = "9.2")
    expect_equal(ver$version, "1.4")
    expect_equal(ver$package, "pytorch==1.4")
    expect_equal(ver$cpu_gpu_packages, "cudatoolkit==9.2")
    expect_equal(ver$channel, "pytorch")
})


test_that("1.5, gpu, stable", {
    version <- "1.5"
    ver <- parse_torch_version(version, cuda_version = "9.2")
    expect_equal(ver$version, "1.5")
    expect_equal(ver$package, "pytorch==1.5")
    expect_equal(ver$cpu_gpu_packages, "cudatoolkit==9.2")
    expect_equal(ver$channel, "pytorch")
})


test_that("default, gpu=9.2, nightly", {
    version <- "default"
    ver <- parse_torch_version(version, cuda_version = "9.2", channel = "nightly")
    expect_equal(ver$version, "1.4")
    expect_equal(ver$package, "pytorch==1.4")
    expect_equal(ver$cpu_gpu_packages, "cudatoolkit==9.2")
    expect_equal(ver$channel, "pytorch-nightly")
})


test_that("1.2, gpu=10.1, stable", {
    ver <- parse_torch_version(version = "1.2", cuda_version = "10.1",
                               channel = "stable")
    expect_equal(ver$version, "1.2")
    expect_equal(ver$package, "pytorch==1.2")
    expect_equal(ver$cpu_gpu_packages, "cudatoolkit==10.1")
    expect_equal(ver$channel, "pytorch")
})

test_that("1.3, gpu=10.2, nightly", {
    ver <- parse_torch_version(version = "1.3", cuda_version = "10.2",
                               channel = "nightly")
    expect_equal(ver$version, "1.3")
    expect_equal(ver$package, "pytorch==1.3")
    expect_equal(ver$cpu_gpu_packages, "cudatoolkit==10.2")
    expect_equal(ver$channel, "pytorch-nightly")
})



context("install_pytorch, dry-run")

test_that("default", {
    res <- install_pytorch(dry_run = TRUE)
    expect_equal(res$package, "pytorch==1.4")
    expect_equal(res$extra_packages, c("cpuonly", "torchvision"))
    expect_equal(res$envname, "r-torch")
    expect_equal(res$conda, "auto")
    expect_equal(res$conda_python_version, "3.6")
    expect_equal(res$channel, "pytorch")
    expect_equal(res$pip, FALSE)
})

test_that("1.2, Python 3.7", {
    res <- install_pytorch(version = "1.2", conda_python_version = "3.7",
                           dry_run = TRUE)
    expect_equal(res$package, "pytorch==1.2")
    expect_equal(res$extra_packages, c("cpuonly", "torchvision"))
    expect_equal(res$envname, "r-torch")
    expect_equal(res$conda, "auto")
    expect_equal(res$conda_python_version, "3.7")
    expect_equal(res$channel, "pytorch")
    expect_equal(res$pip, FALSE)
})

test_that("1.2, Python 3.7, Nightly", {
    res <- install_pytorch(version = "1.2", conda_python_version = "3.7",
                           channel = "nightly",
                           dry_run = TRUE)

    expect_equal(res$package, "pytorch==1.2")
    expect_equal(res$extra_packages, c("cpuonly", "torchvision"))
    expect_equal(res$envname, "r-torch")
    expect_equal(res$conda, "auto")
    expect_equal(res$conda_python_version, "3.7")
    expect_equal(res$channel, "pytorch-nightly")
    expect_equal(res$pip, FALSE)
})

test_that("1.6, Python 3.6, pandas", {
    res <- install_pytorch(version = "1.6", conda_python_version = "3.6",
                           extra_packages = "pandas",
                           dry_run = TRUE)

    expect_equal(res$package, "pytorch==1.6")
    expect_equal(res$extra_packages, c("cpuonly", "torchvision", "pandas"))
    expect_equal(res$envname, "r-torch")
    expect_equal(res$conda, "auto")
    expect_equal(res$conda_python_version, "3.6")
    expect_equal(res$channel, "pytorch")
    expect_equal(res$pip, FALSE)
})


test_that("1.3, Python 3.6, pandas+matplotlib, gpu=9.2", {
    res <- install_pytorch(version = "1.3", conda_python_version = "3.6",
                           extra_packages = c("pandas", "matplotlib"),
                           cuda_version = "9.2",
                           dry_run = TRUE)

    expect_equal(res$package, "pytorch==1.3")
    expect_equal(res$extra_packages, c("cudatoolkit==9.2", "torchvision", "pandas", "matplotlib"))
    expect_equal(res$envname, "r-torch")
    expect_equal(res$conda, "auto")
    expect_equal(res$conda_python_version, "3.6")
    expect_equal(res$channel, "pytorch")
    expect_equal(res$pip, FALSE)
})




context("install_pytorch, live, no dry-run")

test_that("PyTorch 1.6, Python 3.7, pandas", {
    res <- install_pytorch(version = "1.6", conda_python_version = "3.7",
                           extra_packages = "pandas",
                           dry_run = FALSE)

    # use torch_config for live test
    res <- torch_config()
    expect_equal(res$available, TRUE)
    expect_equal(res$version_str, "1.6.0")
    expect_equal(res$python_version, "3.7")
    expect_equal(res$numpy_version, "1.19.1")
    expect_equal(res$env_name, "r-torch")
})
