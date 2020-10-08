# The purpose of these test was initially see if the installation performed as
# initially planned. It works.
# But we cannot enable this test for CRAN because it will take some time and
# may not work due to the PyTorch installation process.
# The major problem I found with these tests is that the `torch_config`
# objects do not update after issuing a new `install_pytorch`.
#
skip("do nothing")
# skip_if_no_torch()

context("install_pytorch, live, no dry-run")
# devtools::reload(pkg = ".", quiet = FALSE)
# unloadNamespace("rTorch")
# library(rTorch)

test_that("PyTorch 1.6, Python 3.7, pandas", {
    res <- install_pytorch(version = "1.6", conda_python_version = "3.7",
                           extra_packages = "pandas",
                           dry_run = FALSE)
    #detach("package:rTorch", unload=TRUE)
    #require(rTorch)
    # devtools::reload(pkg = ".", quiet = FALSE)
    # use torch_config for live test
    # unloadNamespace("rTorch")
    # detach("package:rTorch", unload=TRUE)
    # library(rTorch)
    res <- torch_config()
    expect_equal(res$available, TRUE)
    expect_equal(res$version_str, "1.6.0")
    expect_equal(res$python_version, "3.7")
    expect_equal(res$numpy_version, "1.19.1")
    expect_equal(res$env_name, "r-torch")
})


test_that("PyTorch 1.4, Python 3.6, pandas, matplotlib install from the console", {
    res <- install_pytorch(version = "1.4", conda_python_version = "3.6",
                           extra_packages = c("pandas", "matplotlib"),
                           dry_run = FALSE)

    # detach("package:rTorch", unload=TRUE)
    # require(rTorch)
    # devtools::reload(pkg = ".", quiet = FALSE)
    # unloadNamespace("rTorch")
    # detach("package:rTorch", unload=TRUE)
    # library(rTorch)
    res <- torch_config()
    expect_equal(res$available, TRUE)
    expect_equal(res$version_str, "1.4.0")
    expect_equal(res$python_version, "3.6")
    expect_equal(res$numpy_version, "1.19.1")
    expect_equal(res$env_name, "r-torch")
})


# library(rTorch)
# pkg <- "package:rTorch"
# detach(pkg, character.only = TRUE)
# sessionInfo()
# library(rTorch)
# .rs.restartR()
# library(rTorch)
# library(testthat)

res <- install_pytorch(version = "1.3", conda_python_version = "3.6",
                       extra_packages = c("pandas", "matplotlib"),
                       dry_run = FALSE)

test_that("PyTorch 1.3, Python 3.6, pandas, matplotlib install from the console", {
    res <- torch_config()
    expect_equal(res$available, TRUE)
    expect_equal(res$version_str, "1.3.0")
    expect_equal(res$python_version, "3.6")
    expect_equal(res$numpy_version, "1.19.1")
    expect_equal(res$env_name, "r-torch")
    sessionInfo()
})
