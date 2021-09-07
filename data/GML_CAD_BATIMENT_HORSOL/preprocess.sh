#!/bin/bash
gdal_rasterize -a HAUTEUR -tr 2.5 2.5 -a_nodata 0 -ot Byte CAD_BATIMENT_HORSOL.gml CAD_BATIMENT_HORSOL.tif