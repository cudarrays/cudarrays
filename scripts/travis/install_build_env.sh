#!/bin/bash
# Script called by Travis to install the build environment for CUDArrays. This script must be called with sudo.

set -e

# Install CUDA
echo "Installing CUDA"
CUDA_URL=https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
CUDA_FILE=/tmp/cuda_install.run
wget $CUDA_URL -O $CUDA_FILE
chmod a+x $CUDA_FILE
$CUDA_FILE --toolkit --toolkitpath=$HOME/cuda -silent
rm -f $CUDA_FILE

# Install CMake
echo "Installing CMake"
CMAKE_URL=https://cmake.org/files/v3.4/cmake-3.4.3-Linux-x86_64.tar.gz
CMAKE_FILE=/tmp/cmake.tar.gz
wget --no-check-certificate $CMAKE_URL -O $CMAKE_FILE
tar -xvf $CMAKE_FILE -C $HOME
rm -f $CMAKE_FILE
