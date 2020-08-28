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
ENV PYTORCH_VERSION="1.1"
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



# #! branch: develop
# # clone repo rTorch; master branch modified "master-r-3.6"
# # TODO: change repo to desired branch below
# RUN git clone --depth=50 --branch=develop https://github.com/f0nzie/rTorch.git f0nzie/rTorch && \
#     # git clone --depth=50 --branch=develop-fix-cpu-suffix https://github.com/f0nzie/rTorch.git f0nzie/rTorch && \
#     # git clone --depth=50 --branch=fix-suffix-cpu-in-install_pytorch-travis https://github.com/f0nzie/rTorch.git f0nzie/rTorch && \
#     cd f0nzie/rTorch && \
#     git checkout -qf 3e0551c441dbf0e2f6ab1c55f94c936bfb3e7a31  && \
#     # git checkout -qf 7a97572ed22abbe1b095b8a30dc83376c717b2bf  && \
#     # head of "develop"
#     # git checkout -qf 32616d0bf202318d8ec4a50641e947c3ad60dc43  && \
#     # head of "branch fix-suffix-cpu-in-install_pytorch-travis"
#     # git checkout -qf 7eab93c11e5a0147983a18e571c19e9e75851fb9  
#     # git checkout -qf 211787d7a9ed2cb08053402afbd291bc5b97da54
#     # install the core package
#     R CMD INSTALL .


# RUN R -e 'rTorch:::install_conda(\
#     package=paste0("pytorch=", Sys.getenv("PYTORCH_VERSION")), \
#     envname="r-torch", conda="auto", \
#     conda_python_version=Sys.getenv("PYTHON_VERSION"), \
#     pip=FALSE, channel="pytorch", \
#     extra_packages=c("torchvision", "cpuonly", "matplotlib", "pandas") \
#     )'

# # remove link to libstdc++.so.6 in ther Linux installation
# RUN sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
#     # check as CRAN
#     cd f0nzie && \
#     R CMD build rTorch && \
#     R CMD check rTorch_0.0.3.9003.tar.gz --as-cran && \
#     # test
#     cd rTorch && \
#     Rscript -e 'devtools::check()' && \
#     Rscript -e 'devtools::test(cran=TRUE, run_dont_test=TRUE)'


WORKDIR /home/travis/f0nzie/rTorch