#' Conda install function (modified from reticulate). Necessary to add `channel`
#' to list of parameters
#'
#' R functions for managing Python \href{https://conda.io/docs/user-guide/tasks/manage-environments.html}{conda environments}.
#'
#' @param envname Name of conda environment
#'
#' @param conda Path to conda executable (or "auto" to find conda using the
#'   PATH and other conventional install locations).
#'
#' @param packages Character vector with package names to install or remove.
#'
#' @param pip `TRUE` to use pip (defaults to `FALSE`)
#'
#' @param forge Include the  \href{https://conda-forge.org/}{Conda Forge} repository.
#'
#' @param pip_ignore_installed Ignore installed versions when using pip. This is `TRUE` by default
#'   so that specific package versions can be installed even if they are downgrades. The `FALSE`
#'   option is useful for situations where you don't want a pip install to attempt an overwrite
#'   of a conda binary package (e.g. _SciPy_ on _Windows_ which is very difficult to install via
#'   `pip` due to compilation requirements).
#'
#' @param python_version is the _Python_ version to install. Python __3.4__ is
#'   installed by default. But you could specify for instance:
#'   `python_version="3.7"`.
#'
#' @param channel conda channel for pytorch
#'
#' @param ... Optional arguments, reserved for future expansion.
#'
#' @return `conda_list()` returns a data frame with the names and paths to the
#'   respective python binaries of available environments. `conda_create()`
#'   returns the Path to the python binary of the created environment.
#'   `conda_binary()` returns the location of the main conda binary or `NULL`
#'   if none can be found.
#'
#' @rdname conda-tools
#'
#' @export
conda_install <- function(envname = NULL,
                          packages,
                          forge = FALSE,
                          pip = FALSE,
                          pip_ignore_installed = TRUE,
                          conda = "auto",
                          python_version = NULL,
                          channel = NULL,
                          ...)
{
  # rTorch::conda_install(envname="r-torch-37", packages="pytorch-cpu",
  #         channel = "pytorch", conda="auto", python_version = "3.7")
  #
  #
  # resolve conda binary
  conda <- conda_binary(conda)

  # resolve environment name
  envname <- condaenv_resolve(envname)

  # honor request for specific Python
  python_package <- NULL
  if (!is.null(python_version))
    python_package <- paste("python", python_version, sep = "=")

  # check if the environment exists, and create it on demand if needed.
  # if the environment does already exist, but a version of Python was
  # requested, attempt to install that in the existing environment
  # (effectively re-creating it if the Python version differs)
  # python object carries the path to the executable
  python <- tryCatch(conda_python(envname = envname, conda = conda), error = identity)

  if (inherits(python, "error") || !file.exists(python)) {
    warning("conda_error_create")
    conda_create(envname, packages = python_package, conda = conda)
  } else if (!is.null(python_package)) {
    warning("python_package not null")
    args_ <- conda_args("install", envname, python_package)
    status <- system2(conda, shQuote(args_))
    if (status != 0L) {
      fmt <- "installation of '%s' into environment '%s' failed [error code %i]"
      msg <- sprintf(fmt, python_package, envname, status)
      stop(msg, call. = FALSE)
    }
  }

  if (pip) {
    # use pip package manager
    condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
    cmd <- sprintf("%s%s %s && pip install --upgrade %s %s%s",
                   ifelse(is_windows(), "", ifelse(is_osx(), "source ", "/bin/bash -c \"source ")),
                   shQuote(path.expand(condaenv_bin("activate"))),
                   envname,
                   ifelse(pip_ignore_installed, "--ignore-installed", ""),
                   paste(shQuote(packages), collapse = " "),
                   ifelse(is_windows(), "", ifelse(is_osx(), "", "\"")))
    result <- system(cmd)

  } else {
    # use conda
    args_ <- conda_args("install", envname)
    if (forge)
      args_ <- c(args_, "-c", "conda-forge")
    if (!is.null(channel))
      args_ <- c(args_, "-c", channel)

    args_ <- c(args_, python_package, packages)
    result <- system2(conda, shQuote(args_))
  }

  # check for errors
  if (result != 0L) {
    stop("Error ", result, " occurred installing packages into conda environment ",
         envname, call. = FALSE)
  }

  invisible(NULL)
}



conda_args <- function(action, envname = NULL, ...) {

  envname <- condaenv_resolve(envname)

  # use '--prefix' as opposed to '--name' if envname looks like a path
  args <- c(action, "--yes")
  if (grepl("[/\\]", envname))
    args <- c(args, "--prefix", envname, ...)
  else
    args <- c(args, "--name", envname, ...)

  args

}


condaenv_resolve <- function(envname = NULL) {

  python_environment_resolve(
    envname = envname,
    resolve = identity
  )

}


python_environment_resolve <- function(envname = NULL, resolve = identity) {

  # use RETICULATE_PYTHON_ENV as default
  envname <- envname %||% Sys.getenv("RETICULATE_PYTHON_ENV", unset = "r-torch")

  # treat environment 'names' containing slashes as full paths
  if (grepl("[/\\]", envname)) {
    envname <- normalizePath(envname, winslash = "/", mustWork = FALSE)
    return(envname)
  }

  # otherwise, resolve the environment name as necessary
  resolve(envname)

}


#' @rdname conda-tools
#' @export
conda_python <- function(envname = NULL, conda = "auto") {
  # resolve envname
  envname <- condaenv_resolve(envname)

  # for fully-qualified paths, construct path explicitly
  if (grepl("[/\\\\]", envname)) {
    suffix <- if (is_windows()) "python.exe" else "bin/python"
    path <- file.path(envname, suffix)
    if (file.exists(path))c

    fmt <- "no conda environment exists at path '%s'"
    stop(sprintf(fmt, envname))
  }

  # otherwise, list conda environments and try to find it
  conda_envs <- conda_list(conda = conda)
  env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(env) > 0)
    path.expand(env$python[[1]])
  else
    stop("conda environment ", envname, " not found")
}

