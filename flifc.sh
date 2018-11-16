#!/bin/bash

convert /tmp/in.bmp -colorspace RGB -format ppm /tmp/in.ppm
./ext/FLIF/src/flif --overwrite -e /tmp/in.ppm /tmp/out.file 2> /dev/null
