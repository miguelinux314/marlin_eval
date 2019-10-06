#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compress data with fapec and produce a dataframe compatible with
the one produced with Marlin's benchmark
"""
__author__ = "Miguel Hernández Cabronero <miguel.hernandez@uab.cat>"
__date__ = "10/09/2019"

import sys
import os
import glob
import random
import filecmp
import pandas as pd
import subprocess
import shutil
import time
import itertools
import tempfile

import plot_results

# ---- Begin configuration
compression_count = 5
decompression_count = 5
original_csv_path = "./benchmark_results.csv"
preexisting_fapec_csv_path = "./precomputed_fapec_results.csv"
with_fapec_csv_path = os.path.join(os.path.dirname(original_csv_path), f"withfapec_{os.path.basename(original_csv_path)}")
# ---- End configuration

class FapecCodec:
    tmp_dir = tempfile.mkdtemp(prefix="fapec", dir="/dev/shm")

    def compress(self, path):
        """
        :return: compression_time_s, decompression_time_s, compression_bytes, sample_count
        """
        actual_path = str(path)

        rn = random.randint(0, 10000)
        input_path = os.path.join(self.tmp_dir, f"{os.path.basename(actual_path)}_{rn}")
        output_path = os.path.join(self.tmp_dir, f"compressed_{rn}.fapec")
        reconstructed_path = os.path.join(self.tmp_dir, f"reconstructed_{rn}.fapec")
        shutil.copy(actual_path, input_path)
        try:
            min_test_time_s = 0.5
            min_test_count = 5

            compression_time_s = 0
            compression_count = 0
            while compression_time_s < min_test_time_s or compression_count < min_test_count:
                compression_count += 1
                if os.path.exists(output_path):
                    os.remove(output_path)
                time_before = time.process_time()
                assert os.path.exists("./fapec/fapec"), "A FAPEC codec binary must be provided by the user"
                status, output = subprocess.getstatusoutput(
                    f"./fapec/fapec -o {output_path} -mt 1 -dtype 8 -np us {input_path}")
                compression_time_s += time.process_time() - time_before

                assert status == 0, output
            compression_time_s /= compression_count

            decompression_time_s = 0
            decompression_count = 0
            while decompression_time_s < min_test_time_s:
                decompression_count += 1
                if os.path.exists(reconstructed_path):
                    os.remove(reconstructed_path)
                time_before = time.process_time()
                assert os.path.exists("./fapec/unfapec"), "A FAPEC decoder binary must be provided by the user"
                status, output = subprocess.getstatusoutput(
                    f"./fapec/unfapec -o {reconstructed_path} {output_path}")
                decompression_time_s += time.process_time() - time_before
                assert status == 0, output
            decompression_time_s /= decompression_count

            assert filecmp.cmp(input_path, reconstructed_path)
            assert filecmp.cmp(actual_path, reconstructed_path)

            compression_bytes = os.path.getsize(output_path)
            sample_count = os.path.getsize(input_path)
            return compression_time_s, decompression_time_s, compression_bytes, sample_count
        finally:
            for p in [input_path, output_path, reconstructed_path]:
                if os.path.exists(p):
                    os.remove(p)


if __name__ == '__main__':
    if not os.path.exists("./fapec/fapec") or not os.path.exists("./fapec/fapec"):
        print(f"Codec binaries not found (these must be obtained by the user)\n"
              f"See https://www.dapcom.es/fapec/")
        print(
            f"Appending preexisting {preexisting_fapec_csv_path} to {original_csv_path} into {with_fapec_csv_path}...")
        with open(with_fapec_csv_path, "w") as output_file:
            output_file.write(open(original_csv_path).read())
            output_file.write("\n")
            output_file.write(open(preexisting_fapec_csv_path, "r").read())
        sys.exit(0)

    print("Compressing with FAPEC...")
    fapec_codec = FapecCodec()

    df = pd.read_csv(original_csv_path)

    all_images = list(itertools.chain(*(glob.glob(f"../test_datasets/{d}/*.pgm") for d in plot_results.included_dirs)))
    for i, input_image in enumerate(all_images):

        print(f"Compressing image {i + 1}: {input_image}")
        compression_time_s, decompression_time_s, compression_bytes, sample_count = fapec_codec.compress(input_image)

        df = df.append(pd.Series({
            "codec_name": "fapec",
            "directory": os.path.dirname(input_image),
            "file": input_image,
            "pixel_min": 0,
            "pixel_max": 255,
            "pixel_count": sample_count,
            "compression_bytes": compression_bytes,
            "compression_count": compression_count,
            "decompression_count": decompression_count,
            "compression_avg_time_s": compression_time_s,
            "decompression_avg_time_s": decompression_time_s,
            "lossless": 1}), ignore_index=True)
    df.to_csv(with_fapec_csv_path)