#!/bin/bash
# Script called by Travis to build CUDArrays

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

CONFIGURE="../configure --with-gcc=g++-4.9"

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
