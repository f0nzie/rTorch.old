# set a Linux environment mixing Travis, R, Python Anaconda
FROM travisci/ci-opal:packer-1558622096-f909ac5
LABEL maintainer="Alfonso R. Reyes <alfonso.reyes@oilgainsanalytics.com>"

# Locale configuration --------------------------------------------------------#
RUN localedef -i en_US -f UTF-8 en_US.UTF-8
ENV LANG en_US.UTF-8

USER travis
WORKDIR /home/travis

ENV R_VERSION="4.0.0"
ENV PYTHON_VERSION="3.6"
ENV PYTORCH_VERSION="1.4"
ENV TRAVIS_HOME /home/travis
ENV R_BUILD_LIBS=/home/travis/R/Library
ENV R_LIBS_USER=/home/travis/R/Library
ENV CONDA_BIN=miniconda/bin
ENV R_BIN=R-bin/bin
ENV TEX_BIN=texlive/bin/x86_64-linux
ENV PYTORCH_CPP_LIBS=miniconda/envs/r-torch/lib
ENV R_HOME=${TRAVIS_HOME}/R-bin/lib/R
ENV PATH=${TRAVIS_HOME}/${R_BIN}:${TRAVIS_HOME}/${CONDA_BIN}:${TRAVIS_HOME}/${TEX_BIN}:$PATH
ENV LD_LIBRARY_PATH=${TRAVIS_HOME}/${PYTORCH_CPP_LIBS}


# repositories
RUN sudo add-apt-repository -y "ppa:marutter/rrutter4.0" && \
    sudo add-apt-repository -y "ppa:c2d4u.team/c2d4u4.0+" && \
    sudo add-apt-repository -y "ppa:ubuntugis/ppa" && \
    sudo add-apt-repository -y "ppa:cran/travis"
    

# install Linux dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    libblas-dev liblapack-dev \
    libncurses5-dev libreadline-dev libjpeg-dev libpcre3-dev \
    libpng-dev zlib1g-dev libbz2-dev liblzma-dev libicu-dev \
    cdbs qpdf texinfo libssh2-1-dev devscripts gfortran

## install R dependencies
## download R. lsb_release -cs will yield xenial"
RUN curl -fLo /tmp/R-${R_VERSION}-$(lsb_release -cs).xz https://travis-ci.rstudio.org/R-${R_VERSION}-$(lsb_release -cs).xz && \
    tar xJf /tmp/R-${R_VERSION}-$(lsb_release -cs).xz -C ~ && \
    rm /tmp/R-${R_VERSION}-$(lsb_release -cs).xz && \
    sudo mkdir -p /usr/local/lib/R/site-library $R_LIBS_USER && \
    sudo chmod 2777 /usr/local/lib/R /usr/local/lib/R/site-library $R_LIBS_USER && \
    echo 'options(repos = c(CRAN = "http://cloud.r-project.org"))' > ~/.Rprofile.site


## texlive & pandoc
RUN curl -fLo /tmp/texlive.tar.gz https://github.com/jimhester/ubuntu-bin/releases/download/latest/texlive.tar.gz && \
    tar xzf /tmp/texlive.tar.gz -C ~  && \
    tlmgr update --self && \
    ## pandoc
    curl -fLo /tmp/pandoc-2.2-1-amd64.deb https://github.com/jgm/pandoc/releases/download/2.2/pandoc-2.2-1-amd64.deb && \
    sudo dpkg -i /tmp/pandoc-2.2-1-amd64.deb && \
    sudo apt-get install -f && \
    rm /tmp/pandoc-2.2-1-amd64.deb


# change .Rprofile.site to .Rprofile otherwise ignored by R with "trying to use CRAN without setting a mirror"
RUN echo 'options(repos = c(CRAN = "http://cloud.r-project.org"))' > ~/.Rprofile && \
    Rscript -e 'sessionInfo()' && \
    # install essential packages. will exit with error 1 if any was not installed    
    Rscript -e 'install.packages(c("logging", "reticulate", "jsonlite", "R6", "rstudioapi", "data.table"));\
    if (!all(c("logging", "reticulate", "jsonlite", "R6", "rstudioapi", "data.table") \
     %in% installed.packages())) { q(status = 1, save = "no")}' && \
    # packages for testing the package
    Rscript -e 'install.packages(c("testthat", "rmarkdown", "knitr", "devtools"))'


# install Python Miniconda latest 
# TODO: consider installing a fixed version of Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $HOME/miniconda && \
    hash -r && \
    rm miniconda.sh && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda && \
    conda init bash && \
    conda info -a

# copy all the rTorch project to install, build and test in the Travis machine
# it will automatically detect if files were modified
COPY . /home/travis/f0nzie/rTorch/

# copy .git folder files. will allow to checkout branches in the container
COPY .git/ /home/travis/f0nzie/rTorch/.git/

RUN ls /home/travis/f0nzie/rTorch/.git && \
    cd f0nzie/rTorch && \
    git status

RUN sudo chown -R travis:travis /home/travis/f0nzie/ && \
    cd f0nzie/rTorch && \
    R CMD INSTALL .

RUN R -e 'rTorch:::install_conda(\
    package=paste0("pytorch=", Sys.getenv("PYTORCH_VERSION")), \
    envname="r-torch", conda="auto", \
    conda_python_version=Sys.getenv("PYTHON_VERSION"), \
    pip=FALSE, channel="pytorch", \
    extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas") \
    )'

# remove link to libstdc++.so.6 in the Linux installation forcing to use
# the c++ libs in $HOME/miniconda/lib
RUN sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    # check as CRAN
    # cd f0nzie && \
    # R CMD build rTorch && \
    cd f0nzie/rTorch && \
    export PKGNAME=$(sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION) && \
    export PKGVERS=$(sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION) && \
    cd .. && \
    echo $PKGNAME_$PKGVERS.tar.gz && \
    R CMD build $PKGNAME && \
    R CMD check ${PKGNAME}_${PKGVERS}.tar.gz --as-cran && \
    # test
    cd rTorch && \
    Rscript -e 'devtools::check()' && \
    Rscript -e 'devtools::test(cran=TRUE, run_dont_test=TRUE)'


WORKDIR /home/travis/f0nzie/rTorch