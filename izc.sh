#!/bin/bash

convert /tmp/in.bmp -colorspace RGB -format ppm /tmp/in.ppm
ext/imagezero/build/iz c /tmp/in.ppm /tmp/out.file
