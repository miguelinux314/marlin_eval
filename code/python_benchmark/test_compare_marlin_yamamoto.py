#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for compare_marlin_yamamoto.py
"""
__author__ = "Miguel Hernández Cabronero <miguel.hernandez@uab.cat>"
__date__ = "03/08/2019"

import tempfile
import unittest
from compare_marlin_yamamoto import *


class TestSource(unittest.TestCase):
    def test_laplacian_generation(self):
        for symbol_count in [8, 16, 32, 64, 128, 256, 512, 1024]:
            max_entropy = math.log2(symbol_count)
            for entropy_fraction in np.linspace(0.1, 1, 20):
                s = Source.get_laplacian(
                    symbol_count=symbol_count, max_entropy_fraction=entropy_fraction)
                assert abs(s.entropy - (max_entropy * entropy_fraction)) < 1e-9, \
                    (s.entropy, (max_entropy * entropy_fraction), s.entropy - (max_entropy * entropy_fraction))

    def test_laplacian_minimum_entropy(self):
        for symbol_count in [8, 16, 32, 64, 128, 256, 512, 1024]:
            max_entropy = math.log2(symbol_count)
            entropy_fraction = 1e-6
            s = Source.get_laplacian(symbol_count=symbol_count, max_entropy_fraction=entropy_fraction)
            assert abs(s.entropy - (max_entropy * entropy_fraction)) < 1e-9, \
                (s.entropy, (max_entropy * entropy_fraction), s.entropy - (max_entropy * entropy_fraction))
            print(f"sc = {symbol_count} ok ")


class TestPlotting(unittest.TestCase):
    def test_linedata(self):
        tmp_file = tempfile.mkstemp()
        plt.figure()
        data = LineData(x_values=list(range(10)), y_values=[x ** 2 for x in range(10)],
                        x_label="x", y_label="y",
                        label=r"Test data $\hat{f}(x) = \left(\sqrt{x^2}\right)^2$")
        data.render()
        plt.savefig(tmp_file[1], format="pdf")
        plt.close()


if __name__ == '__main__':
    unittest.main()
