import geopandas as gpd

year = 2016

idc = gpd.read_file("SCANE_INDICE_MOYENNES_3_ANS.gdb")
idc = idc[idc.ANNEE == year]
bldgs = gpd.read_file("../GML_CAD_BATIMENT_HORSOL/CAD_BATIMENT_HORSOL.gml")

not_idc = bldgs.loc[~bldgs.EGID.isin(idc.EGID)] #without EGID

not_idc = not_idc.set_index("EGID")
not_idc = not_idc.loc[:,["geometry"]]

not_idc.to_file("not_idc_{}.gpkg".format(year),driver="GPKG")
