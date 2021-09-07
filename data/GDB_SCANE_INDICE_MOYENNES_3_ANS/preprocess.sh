#!/bin/bash

python3 without_IDC.py
gdal_rasterize -burn 1 -tr 2.5 2.5 -a_nodata 0 -ot Byte not_idc_2016.gpkg not_idc_2016.tif