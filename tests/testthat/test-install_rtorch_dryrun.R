library(testthat)


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


