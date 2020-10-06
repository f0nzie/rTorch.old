

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
