#!/bin/bash
# main.sh
#
# Run all benchmarks
#
# Miguel Hernández Cabronero <miguel.hernandez@uab.cat>

echo "Running all benchmarks..."
echo "(see README.md in case any library is not found)"
echo

cd ht_benchmark
echo "Building HT benchmark..."
date
make clean
make
cd "... built!"
    cd bin
    echo "Running HT benchmark..."
    date
    ./benchmark
    echo "... done!"

    cp benchmark_results.csv ../ht_analysis/

    cd ../ht_analysis/
    ./clean.sh
    echo "Adding FAPEC results to HT benchnmark..."
    date
    ./compress_fapec.py
    echo "... added!"
    echo "Plotting HT benchmark results"
    ./plot_results.py
    echo "... plotted!"
    echo "Finished HT benchmark"
cd ../../

cd python_benchmark
./clean.sh
echo "Running Python benchmark..."
date
./compare_marlin_yamamoto.py
echo "... done!"
cd ..

echo "All benchmarks run"

