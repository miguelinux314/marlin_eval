#!/usr/bin/env bash
set -ex

cat /proc/cpuinfo
cat /etc/issue

echo "Running all benchmarks..."
echo "(see README.md in case any library is not found)"
echo

cd python_benchmark
./clean.sh
echo "Running Python benchmark..."
date
./compare_marlin_yamamoto.py
echo "... done!"
cp -r plots_all ../../results/
cd ..

cd ht_benchmark
echo "Building HT benchmark..."
date
make clean
echo "\tBuilding external codec libraries"
date
mkdir -p bin
make
echo "... built HT benchamrk!"
    
    cd bin
    echo "Running HT benchmark..."
    date
    ./benchmark
    echo "... done!"

    mv -v benchmark_results.csv ../ht_analysis/

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
    cp *results.csv allcodecs*pdf ../../../results/
cd ../../

echo "All benchmarks run. See ../results"

