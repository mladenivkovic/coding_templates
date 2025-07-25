#!/bin/bash

autoreconf --install # calls autoconf, automake, and related commands in the right order.
./configure
make

# Basic idea is to run
# 1)    aclocal (see Auto-generating aclocal.m4: Invoking aclocal),
# 2)    autoconf (see The Autoconf Manual),
# 3)    (if needed) autoheader (part of the Autoconf distribution), and
# 4)    automake (see Creating a Makefile.in: Invoking automake).
