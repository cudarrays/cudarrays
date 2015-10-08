#!/bin/bash
# Script called by Travis to install the build environment for CUDArrays. This script must be called with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"

# Install apt packages where the Ubuntu 12.04 default and ppa works for Caffe

# This ppa is for gcc-4.9
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get -y update
apt-get install \
    gcc-4.9 g++-4.9

# Install CUDA
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
curl $CUDA_URL -o $CUDA_FILE
dpkg -i $CUDA_FILE
rm -f $CUDA_FILE
apt-get -y update
# Install the minimal CUDA subpackages required to test Caffe build.
# For a full CUDA installation, add 'cuda' to the list of packages.
apt-get -y install cuda-core-7-5 cuda-cudart-7-5 cuda-cudart-dev-7-5
# Create CUDA symlink at /usr/local/cuda
# (This would normally be created by the CUDA installer, but we create it
# manually since we did a partial installation.)
ln -s /usr/local/cuda-7.5 /usr/local/cuda
