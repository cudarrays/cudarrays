#!/bin/bash
# Script called by Travis to install the build environment for CUDArrays. This script must be called with sudo.

set -e

# Install CUDA
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
CUDA_FILE=/tmp/cuda_install.run
curl $CUDA_URL -o $CUDA_FILE
chmod a+x $CUDA_FILE
$CUDA_FILE --toolkit --toolkitpath=$HOME/cuda -silent
rm -f $CUDA_FILE

# Install CMake
CMAKE_URL=http://www.cmake.org/files/v3.1/cmake-3.1.3-Linux-x86_64.tar.gz
CMAKE_FILE=/tmp/cmake.tar.gz
curl $CMAKE_URL -o $CMAKE_FILE
tar -xvf $CMAKE_FILE -C $HOME
rm -f $CMAKE_FILE
