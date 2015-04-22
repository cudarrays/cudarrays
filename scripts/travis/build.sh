#!/bin/bash
# Script called by Travis to build CUDArrays

export PATH=/usr/local/cmake/bin:$PATH

set -e
set -x
MAKE="make --jobs=$NUM_THREADS --keep-going"

CONFIGURE="../configure --with-gcc=`which g++-4.9`"

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
