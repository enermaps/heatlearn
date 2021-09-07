# HeatLearn

**A CNN model for prediction of aggregated building energy demand from land use maps**

This repository contains the models presented in a paper entitled *A machine-learning model for the prediction of aggregated building energy demand from pan-European open dataset* presented at the CISBAT 2021 Conference.

# Data

## Source data

- European Settlement Map (ESM) 2012 - release 2017 [`200km_2p5m_N26E38`](https://land.copernicus.eu/pan-european/GHSL/european-settlement-map/esm-2012-release-2017-urban-green) for Geneva Area

- Geneva IDC ("*indice de dépense de chaleur*", building heating demand indicator) [`GDB_SCANE_INDICE_MOYENNES_3_ANS`](https://ge.ch/sitg/fiche/2177 )

- Geneva cadastral footprints [`GML_CAD_BATIMENT_HORSOL`](https://ge.ch/sitg/fiche/9810 )

- [Federal Statistics of Buildings and Dwellings (StatBL), 2019](https://www.bfs.admin.ch/bfs/fr/home/services/geostat/geodonnees-statistique-federale/batiments-logements-menages-personnes/batiments-logements-des-2010.assetdetail.14716197.html) - this is used for creating a fishnet grid according to Swiss official statistical data.


## Reference data

Additional data for model benchmarking are obtained from the Hotmaps project:

- Calculated heated gross floor area of residential and non-residential buildings on a hectare level 100x100m for EU28, Norway, Iceland and Switzerland in 2012 [`heat_tot_curr_density`](https://gitlab.com/hotmaps/heat/heat_tot_curr_density)

- Calculated Heat density (final energy demand for space heating and DHW) on a hectare level 100x100m for EU28, Norway, Iceland and Switzerland in 2015 [`gfa_tot_curr_density`](https://gitlab.com/hotmaps/gfa_tot_curr_density)


## Preprocessed data

This is produced using `prepareData.py`:

- `data/{grid_type}_{size}_{tile_type}.csv`: aggegated statistics per tile from RegBL. The last  columns contain the aggregated statistics:

    - `nIDC`: number of buildings with IDC value

    - `nBLDG`: total number of buildings

    - `footprint_area`: footprint area from vector shapes

    - `floor_area`: heated floor area ("*surface de référence énergétique*")

    - `idc_norm`: mean normalized IDC (MWh/m2 floor area). The average has been weighted according the building footprint area.

    - `idc_norm_foot`: mean normalized IDC (MWh/m2 footprint area). This is to compare with HotMaps results.

    - `idc_abs`: cumulative absolute IDC (MWh)

- `/data/200km_2p5m_N26E38/tiles_{grid_type}_{tile_type}/tiles_{size}`: tiles of ESM

- `/data/GML_CAD_BATIMENT_HORSOL/tiles_{grid_type}_{tile_type}/tiles_{size}`: tiles of the rasterized building footprint dataset (burnt with the height value)

- `/data/GDB_SCANE_INDICE_MOYENNES_3_ANS/tiles_{grid_type}_{tile_type}/tiles_{size}`: tiles representing buildings that do not have IDC data.

- `/data/{grid_type}_{size}.csv`: Grid from a given `grid_type` dataset and tile `{size}`. Note that tiles with sizes different than 100 might be overlapped, as the center is the same as the original 100x100 m grid.

- `/data/hotmaps_heat_density.csv`: statistics for the HotMaps dataset used as reference engineering model.


### Main parameters

The main parameters are:
- `{size}` is the size of the tile in meters
- `{grid_type}` is the grid source:
    - hotmaps, grid extracted from Hotmaps project
    - statbl, grid extracted from Swiss statistics data
    - grid50, a custom-made grid covering Geneva canton spaced at 50 m
- `{tile_type}`, defining the dispositions of the tiles
    - contiguous
    - overlapped


## Results

The results folder contains the outputs of the model as computed by the `cnn_regression.py`file.

A file with the parameters used to run the model named `parameters.json` is contained in each directory.

# Usage

## Obtaining and preprocessing the source data 

Instructions to download and preprocess the source data are provided in the `README.md` file in each `data` subdirectory.

## Preparing data input

The script `prepareData.py` is  used to prepare the data for the model (notably for tiling and calculating summary statistics).
The following parameters combinations for `size`, `grid_type`, `tile_type` and `tile_type` are to be computed:

| grid_type | tile_size | tile_type  |
|-----------|-----------|------------|
| hotmaps   | 100       | contiguous |
| grid50    | 300       | overlapped |
| grid50    | 100       | contiguous |
| grid50    | 500       | overlapped |
| grid50    | 100       | overlapped |
| statbl    | 300       | contiguous |

## Running the model

The `cnn_regression.py`is used to train the model.

A file `parameters.json` containing the model parameters should be placed in the same directory. The `parameters_template.json` can be renamed and used for this purpose.

## Data analysis

The notebook `analysis.ipynb`provides the code for recreating the plots and data of the paper.

# Requirements

The main scripts are written in Python3. These were tested with a Conda installation of Python 3.7.9.

A conda environment `heatlearn.yml` with the required packages is provided.

Some preprocessing scripts require GDAL (v.3.1.4) and Bash.


# Contact

Feel free to contact the authors at giuseppe[dot]peronato[at]alumni[dot]epfl[dot]ch
The preprocessed data can be also obtained on demand via a file-transfer system.