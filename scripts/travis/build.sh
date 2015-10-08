#!/bin/bash
# Script called by Travis to build CUDArrays

export PATH=/usr/local/cmake/bin:$PATH

set -e
set -x
MAKE="make --jobs=$NUM_THREADS --keep-going"

CONFIGURE="../configure --with-gcc=/usr/bin/g++-4.9 --with-cuda=$CUDA_HOME"

if $BUILD_DEBUG
then
     CONFIGURE="$CONFIGURE --enable-debug"
fi

mkdir build
cd build

$CONFIGURE
$MAKE

tests/unit/UnitTests

$MAKE clean

cd -
