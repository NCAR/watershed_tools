### watershed_tools ###

A collection of python scripts to discretize a watershed shapefile into sub-areas to account for geospatial attributes.

#### Synopsis ####

The experimental application goal was to provide an effective discretization of watersheds based on 3 primary watershed attributes influencing hydrologic runoff variability:
* Elevation;
* Vegetation; and,
* Solar radiation exposure.

In addition, the application targeted use in SUMMA modeling efforts, in which watersheds are viewed as 'grouped response units' (GRUs) with the sub-watersheds termed 'hydrologic response units' (HRUs) -- a naming convention used throughout the code.  <br>

Rather than use data-driven clustering approaches for arbitrary attributes to derive HRUs, known controlling factors are applied in a binary fashion to create 8 potential discretization levels (all permutations of the 3 factors) for the GRUs.<br>

Small HRUs can be eliminated (based on area fraction or area thresholds).<br>

The overarching objective is to support the implementation of watershed models that represent spatial and process heterogeneity with a computationally frugal approach -- i.e., 1-8 HRU elements per watershed in this case -- although this limit is not prescribed and the code is extensible to allow for the introduction of other factors.   

#### Getting Started ####

##### Input Data Requirements #####

These scripts require underlying geospatial data in order to run. For example, full domain basin shapefiles, dem, land and soil classification. Data requirements
and the preprocessing for this data is found in the 'data prep' directory. Check that you can access the needed data before proceeding.

#### Installation Instructions ####

Following the instructions below will help ensure that you have the correct python packages
needed to run the scripts and Jupyter Notebooks. It is recommended to create a virtual environment.

>  GDAL is a required installation for watershed_tools. Assuming a local installation of GDAL exists, check the installation and version number by running `gdalinfo --version` from the terminal or command prompt.

`cd /path/to/watershed_tools` <br>
`virtualenv watershed_tools_venv`<br>
`source /path/to/watershed_tools_venv/bin/activate`<br>
`pip install -r requirements.txt`<br>
`pip install gdal=='version_number'`

#### Running Jupyter Notebook ####

From the command prompt, activate your virtual environment than open a Jupyter Notebook as below.

`source watershed_tools_venv/bin/activate`<br>
`jupyter notebook`<br>

#### Code organization ####
The code is organized into subdirectories as follows:

 * data_prep/
 * discretize/
 * analysis/
 * functions/
 * docs/
 * test_cases/

#### Typical workflow ####
The recommended workflow for appying this code is the following:<br>
 1. Create a new test case, using the existing example and control file as a template.
 1. In data_prep/, first run ...




#### Contacts ####
Andy Wood, andywood@ucar.edu
Hongli Liu, hongli.liu@usask.edu

Authors:  2020-2021 Andy Wood and Hongli Liu designed and wrote the codes. <br>
          HL did initial code prototyping for the specified design and AW upgraded codes through revision, debugging, streamlining, testing and documentation. <br>


#### References and Acknowledgements ####
The development of this code base and the associated experimental project (led by A. Wood) was funded by the US Bureau of Reclamation under Cooperative Agreement #R16AC00039.<br>

The initial application of the code is described in:<br>

    Liu, H, AW Wood, D Broman, G Brown, and J Lanini, 2021.  Impact of SUMMA hydrologic model discretization on the representation of snowmelt and runoff variability.  J. Hydromet. (in prep, target submission Nov 2021)

We thank Genevieve Brown of the University of Waterloo for providing the code implementation of the radiation algorithm that was included in the radiation preparation step.  <br>

We also thank Wouter Knoben for providing a MERIT-based DEM file and Naoki Mizukami for providing the landcover dataset file and soiltype files that were used in the initial experiment & development. <br>
