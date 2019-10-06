#!/bin/bash
# main.sh
#
# Run everything
# Miguel Hernández Cabronero <miguel.hernandez@uab.cat>

make clean
make

cd bin
./benchmark
cd ..

cp bin/benchmark_results.csv analysis/
cd analysis
./compress_fapec.py
./plot_results.py

