# Investigating the limiting aircraft design-dependent and environmental factors of persistent contrail formation

_Authors_:

- Liam Megill (1, 2), https://orcid.org/0000-0002-4199-6962   
- Volker Grewe (1, 2), https://orcid.org/0000-0002-8012-6783

_Affiliation (1)_: Deutsches Zentrum für Luft- und Raumfahrt (DLR), Institut für Physik der Atmosphäre, Oberpfaffenhofen, Germany

_Affiliation (2)_: Delft University of Technology (TU Delft), Faculty of Aerospace Engineering, Section Aircraft Noise and Climate Effects (ANCE), Delft, The Netherlands

_Corresponding author_: Liam Megill, liam.megill@dlr.de

_doi_: https://doi.org/10.5194/egusphere-2024-3398

---

## General introduction

This dataset contains all data and code developed during research towards the linked article. It contains all elements required to reproduce the linked figures and analysis. The research was carried out by Liam Megill during his PhD project at the DLR.

---

## Description of the data

The analysis was conducted on the supercomputer Levante of the German Climate Computing Center (DKRZ) using ERA5 data. The ERA5 data is stored on Levante in GRIB format, but is otherwise also available from ECMWF at https://cds.climate.copernicus.eu/. The DEPA 2050 air traffic scenario is available from Zenodo at https://doi.org/10.5281/zenodo.11442323. 

The aircraft naming differs to that in the linked paper. For reference, AC0 = CON-LG (last generation conventional); AC1 = CON-NG (next generation conventional); AC3 = HYB-80 (80% hybrid electric); AC4 = H2C-04 (hydrogen combustion); AC7 = WET-50 (Water Enhanced Turbofan with 50% EIH2O reduction); and AC8 = WET-75 (Water Enhanced Turbofan with 75% EIH2O reduction).

---

## Data and code organisation and naming

To view and use the code and data, unzip `dataset.zip` into the main directory with this readme file. The data is then provided in `data.zip`, which should be unzipped to a folder `data` within the main directory. The input data to each notebook is explained in more detail in the notebook. Plots can be saved by following the instructions in the notebooks. 

The structure of the full project directory is the following:


    ├── LICENSES
    │   ├── CC-BY-4.0.txt  <- CC-BY 4.0 license for data files       
    │   └── Apache-License-v2.0.txt  <- Apache License v2.0 for scripts      
    │
    ├── README.md 
    │
    ├── data (unzipped `data.zip`)
    │   ├── external       <- Data from third party sources.
    │   │   ├── (DEPA 2050)  
    │   │   │   └── (FP_con_2050.txt)  <- can be downloaded from https://doi.org/10.5281/zenodo.11442323
    │   │   └── (ERA5-IAGOS_QM-CDF_Hofer2024.csv)  <- can be requested from sina.hofer@dlr.de
    │   │
    │   ├── processed 
    │   │   ├── limfac
    │   │   │   ├── AC* (0, 1, 3, 4, 7, 8)
    │   │   │   │   └── data per AC can be requested from the corresponding author
    │   │   │   ├── areas_grib.pickle
    │   │   │   ├── limfac_allAC_rmS_ERA5_GRIB_allcorr_v3.nc
    │   │   │   ├── neighbours_grib.pickle
    │   │   │   ├── nonborder_limfac_allAC_rmS_ERA5_GRIB_allcorr.nc
    │   │   │   ├── perimeters_grib.pickle
    │   │   │   ├── vert_limfac_allAC_rmS_ERA5_GRIB_allcorr_v3.nc
    │   │   │   └── vertical_neighbors_grib.pickle
    │   │   │
    │   │   └── ppcf
    │   │       ├── ppcfhist_M_2010s_ERA5_GRIB_v2.nc
    │   │       └── ppcfhist_S_2010s_ERA5_GRIB_v2.nc
    │   │
    │   ├── raw  <- suggested location for ERA5 data if not using DKRZ Levante
    │   │
    │   └── aircraft_specs_v2.nc
    │
    ├── notebooks                              <- Jupyter notebooks.
    |   ├── figs                               <- folder for output figures
    │   ├── 02-lm-create_aircraft_specs.ipynb  <- creates aircraft design specifications
    │   ├── 11-lm-supporting_graphs.ipynb      <- creates supporting graphs and data
    │   ├── 15-lm-Gmax_grib.ipynb              <- performs Gmax calculations using ERA5 data
    │   ├── 17-lm-analyse_Gmax.ipynb           <- analyses ppcf vs G analysis
    │   ├── 40-lm-random_limfac.ipynb          <- performs horizontal limiting factors (borders) calculations using ERA5 data
    │   ├── 41-lm-analyse_limfac.ipynb         <- analyses limiting factors results
    │   ├── 43-lm-random_vertical_limfac.ipynb <- performs vertical limiting factors (borders) calculations using ERA5 data
    │   ├── 50-lm-num_limfac_limiting.ipynb    <- performs limiting factors (non-border) calculations using ERA5 data
    │   ├── 53-lm-concat_results.ipynb         <- combines all individual datasets into single ones
    │   └── helper.py                          <- helper functions for all notebooks
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment


---

## Python requirements

The analyses were performed with python 3.11. A suitable python installation can be created automatically using conda and `requirements.txt`. When using conda, create the environment with:

```
conda create --name <name_environment> python==3.11
conda activate <name_environment>
pip install -r requirements.txt
```

---

## License

All data files are licensed under a CC-BY 4.0 (see `LICENSE/CC-BY-4.0.txt` file). All Juypter notebooks and Python scripts are licensed under an Apache License v2.0 (see `LICENSE/Apache-License-v2.0.txt` file).

## Copyright

Copyright  © 2024 Liam Megill

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

