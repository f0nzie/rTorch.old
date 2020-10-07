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
