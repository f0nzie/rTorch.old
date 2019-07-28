#' Conda install
#' @param forge Include the [Conda Forge](https://conda-forge.org/) repository.
#' @param pip_ignore_installed Ignore installed versions when using pip. This is `TRUE` by default
#'   so that specific package versions can be installed even if they are downgrades. The `FALSE`
#'   option is useful for situations where you don't want a pip install to attempt an overwrite
#'   of a conda binary package (e.g. SciPy on Windows which is very difficult to install via
#'   pip due to compilation requirements).
#'
#'
#' @keywords internal
#'
#' @export
conda_install <- function(envname = NULL,
                          packages,
                          forge = TRUE,
                          pip = FALSE,
                          pip_ignore_installed = TRUE,
                          conda = "auto",
                          python_version = NULL,
                          channel = NULL,
                          ...)
{
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
  python <- tryCatch(conda_python(envname = envname, conda = conda), error = identity)
  if (inherits(python, "error") || !file.exists(python)) {
    conda_create(envname, packages = python_package, conda = conda)
  } else if (!is.null(python_package)) {
    args <- conda_args("install", envname, python_package)
    status <- system2(conda, shQuote(args))
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
    args <- conda_args("install", envname)
    if (forge && is.null(conda))
      args <- c(args, "-c", "conda-forge")
    else
      args <- c(args, "-c", channel)
    args <- c(args, python_package, packages)
    print(shQuote(args))
    result <- system2(conda, shQuote(args))
  }

  # check for errors
  if (result != 0L) {
    stop("Error ", result, " occurred installing packages into conda environment ",
         envname, call. = FALSE)
  }

  invisible(NULL)
}
