#' Install PyTorch and its dependencies
#'
#' @inheritParams reticulate::conda_list
#'
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on _Windows_ (as this isn't supported by _PyTorch_). Note also
#'   that since this command runs without privillege the "system" method is
#'   available only on _Windows_.
#'
#' @param version PyTorch version to install. The "default" version is __1.4__.
#'   You can specify a specific __PyTorch__ version with `version="1.2"`,
#'   or `version="1.6"`.
#'
#' @param envname Name of Python or conda environment to install within.
#'   The default environment name is `r-torch`.
#'
#' @param extra_packages Additional Python packages to install along with
#'   PyTorch. If more than one package use a character vector:
#'   `c("pandas", "matplotlib")`.
#'
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#'
#' @param conda_python_version the _Python_ version installed in the created _conda_
#'   environment. Python __3.4__ is installed by default. But you could specify for instance:
#'   `conda_python_version="3.7"`.
#'
#' @param pip logical
#'
#' @param channel conda channel. The default channel is `stable`.
#'   The alternative channel is `nightly`.
#'
#' @param cuda_version string for the cuda toolkit version to install. For example,
#'   to install a specific CUDA version use `cuda_version="10.2"`.
#'
#' @param dry_run logical, set to TRUE for unit tests, otherwise will execute
#'   the command.
#'
#' @param ... other arguments passed to [reticulate::conda_install()] or
#'   [reticulate::virtualenv_install()].
#'
#' @importFrom jsonlite fromJSON
#'
#' @export
install_pytorch <- function(method = c("conda", "virtualenv", "auto"),
                               conda = "auto",
                               version = "default",
                               envname = "r-torch",
                               extra_packages = NULL,
                               restart_session = TRUE,
                               conda_python_version = "3.6",
                               pip = FALSE,
                               channel = "stable",
                               cuda_version = NULL,
                               dry_run = FALSE,
                               ...) {

  # verify 64-bit
  if (.Machine$sizeof.pointer != 8) {
    stop("Unable to install PyTorch on this platform.",
         "Binary installation is only available for 64-bit platforms.")
  }

  method <- match.arg(method)

  # unroll version
  ver <- parse_torch_version(version, cuda_version, channel)

  version <- ver$version
  gpu <- ver$gpu
  package <- ver$package
  cpu_gpu_packages <- ver$cpu_gpu_packages
  channel <- ver$channel

  # Packages in this list should always be installed.

  default_packages <- c("torchvision")

  # # Resolve torch probability version.
  # if (!is.na(version) && substr(version, 1, 4) %in% c("1.1.0", "1.1", "1.1.0")) {
  #   default_packages <- c(default_packages, "pandas")
  #   # install pytorch-nightly
  # } else if (is.na(version) ||(substr(version, 1, 4) %in% c("2.0.") || version == "nightly")) {
  #   default_packages <- c(default_packages, "numpy")
  # }

  extra_packages <- unique(c(cpu_gpu_packages, default_packages, extra_packages))

  if (dry_run) {
      os <- ifelse(is_osx(), "osx",
                   ifelse(is_linux(), "linux",
                          ifelse(is_windows(), "windows", "None")))
      out <- list(package = package, extra_packages = extra_packages,
                  envname = envname, conda = conda,
                  conda_python_version = conda_python_version,
                  channel = channel, pip = pip, os = os)
      return(out)
  }

  # Main OS verification.
  if (is_osx() || is_linux()) {

    if (method == "conda") {
      install_conda(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        conda = conda,
        conda_python_version = conda_python_version,
        channel = channel,
        pip = pip,
        ...
      )
    } else if (method == "virtualenv" || method == "auto") {
      install_virtualenv(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        ...
      )
    }

  } else if (is_windows()) {

    if (method == "virtualenv") {
      stop("Installing PyTorch into a virtualenv is not supported on Windows",
           call. = FALSE)
    } else if (method == "conda" || method == "auto") {

      install_conda(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        conda = conda,
        conda_python_version = conda_python_version,
        channel = channel,
        pip = pip,
        ...
      )

    }

  } else {
    stop("Unable to install PyTorch on this platform. ",
         "Binary installation is available for Windows, OS X, and Linux")
  }

  message("\nInstallation complete.\n\n")

  if (restart_session && rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()

  invisible(NULL)
}



install_conda <- function(package, extra_packages, envname, conda,
                          conda_python_version, channel, pip, ...) {

  # Example:
  # rTorch:::install_conda(package="pytorch=1.4",
  # extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas")
  # envname="r-torch", conda="auto", conda_python_version = "3.6",
  # channel="pytorch", pip=FALSE
  # )

  # find if environment exists
  envname_exists <- envname %in% reticulate::conda_list(conda = conda)$name

  # remove environment
  if (envname_exists) {
    message("Removing ", envname, " conda environment... \n")
    reticulate::conda_remove(envname = envname, conda = conda)
  }


  message("Creating ", envname, " conda environment... \n")
  reticulate::conda_create(
    envname = envname, conda = conda,
    packages = paste0("python=", conda_python_version)
  )

  message("Installing python modules...\n")
  # rTorch::conda_install(envname="r-torch-37", packages="pytorch-cpu",
  #         channel = "pytorch", conda="auto", python_version = "3.7")
  conda_install(
    envname = envname,
    packages = c(package, extra_packages),
    conda = conda,
    pip = pip,       # always use pip since it's the recommend way.
    channel = channel,
    ...
  )

}

install_virtualenv <- function(package, extra_packages, envname, ...) {

  # find if environment exists
  envname_exists <- envname %in% reticulate::virtualenv_list()

  # remove environment
  if (envname_exists) {
    message("Removing ", envname, " virtualenv environment... \n")
    reticulate::virtualenv_remove(envname = envname, confirm = FALSE)
  }

  message("Creating ", envname, " virtualenv environment... \n")
  reticulate::virtualenv_create(envname = envname)

  message("Installing python modules...\n")
  reticulate::virtualenv_install(
    envname = envname,
    packages = c(package, extra_packages),
    ...
  )

}


parse_torch_version <- function(version, cuda_version = NULL, channel = "stable") {
  default_version <- "1.4"
  # channel <- "pytorch"    # this is the channel

  ver <- list(
    version = default_version,
    gpu = FALSE,
    package = NULL,
    cuda_version = cuda_version,
    cpu_gpu_packages = NULL,
    channel = channel
  )

  if (version == "default") {
    ver$package <- paste0("pytorch==", ver$version)
  } else {
    ver$version <- version
    ver$package <- paste0("pytorch==", ver$version)
  }


  if (is.null(ver$cuda_version)) {
    ver$cpu_gpu_packages <- "cpuonly"
  } else {
    ver$cuda_version <- cuda_version
    ver$cpu_gpu_packages <- paste0("cudatoolkit==", ver$cuda_version)
  }

  if (channel == "stable") {
    ver$channel <- "pytorch"
  } else if (channel == "nightly") {
    ver$channel <- "pytorch-nightly"
  } else {
    stop("not a valid channel")
  }

  ver
}



#' Install additional Python packages alongside PyTorch
#'
#' This function is deprecated. Use the `extra_packages` argument in function
#' `install_pytorch()` to install additional packages.
#'
#' @param packages Python packages to install
#' @param conda Path to conda executable (or "auto" to find conda using the PATH
#'   and other conventional install locations). Only used when PyTorch is
#'   installed within a conda environment.
#'
#' @keywords internal
#'
install_torch_extras <- function(packages, conda = "auto") {
  message("Extra packages not installed (this function is deprecated). \n",
          "Use the extra_packages argument to install_pytorch() to ",
          "install additional packages.")
}
