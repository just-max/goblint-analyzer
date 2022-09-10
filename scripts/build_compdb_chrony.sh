#!/usr/bin/env bash

export LC_ALL=C.UTF-8
./configure
make clean
compiledb -- make -j 1
