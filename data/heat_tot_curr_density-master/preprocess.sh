#!/bin/bash
# Bounds to idc data
gdalwarp -t_srs EPSG:2056 -te 2486205.3516 1110439.1572 2512815.8984 1134224.7039 data/heat_tot_curr_density.tif data/heat_tot_curr_density_2056.tif
gdal_translate data/heat_tot_curr_density_2056.tif data/heat_tot_curr_density_2056.xyz
