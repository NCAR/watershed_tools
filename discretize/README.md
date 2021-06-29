### Watershed Tools ###

#### Synopsis ####
A collection of python scripts to discretize a watershed shapefile into sub-areas to account for geospatial attributes.<br>
Authors:  2020-2021 Andy Wood and Hongli Liu designed and wrote the codes. <br>
          HL did initial code prototyping for the specified design and AW upgraded codes through revission, debugging, streamlining, testing and documentation. <br>

The experimental application goal was to provide an effective discretization of watersheds based on 3 primary watershed attributes influencing hydrologic runoff variability:  elevation, vegetation, and solar radiation exposure. <br>
In addition, the application targeted use in SUMMA modeling efforts, in which watersheds are viewed as 'grouped response units' (GRUs) with the sub-watersheds termed 'hydrologic response units' (HRUs) -- a naming convention used throughout the code.  <br>
Rather than use data-driven clustering approaches for arbitrary attributes to derive HRUs, known controlling factors are applied in a binary fashion to create 8 potential discretization levels (all permutations of the 3 factors) for the GRUs.<br>
Small HRUs can be eliminated (based on area fraction or area thresholds).<br>
The overarching objective is to support the implementation of watershed models that represent spatial and process heterogeneity with a computationally frugal approach -- i.e., 1-8 HRU elements per watershed in this case -- although this limit is not prescribed and the code is extensible to allow for the introduction of other factors.   

#### Code organization ####
The code is organized into subdirectories as follows:<br>
 * data_prep/
 * discretize/
 * analysis/
 * functions/
 * docs/
 * test_cases/

#### Code workflow ####
The recommended workflow for appying this code is the following:<br>
 1. In data_prep/, first run ... 





#### Contacts ####
Andy Wood, andywood@ucar.edu
Hongli Liu, hongli.liu@usask.edu




#### Acknowledgements ####
The development of this code base and the associated experimental project (led by A. Wood) was funded by the US Bureau of Reclamation under Cooperative Agreement #___.<br>
The initial application of the code is described in [REF].<br>
We thank Genevieve __ for providing the code implementation of the radiation algorithm that was included in the radiation preparation step.  <br>
We also thank Wouter Knoben for providing a MERIT-based DEM file and Naoki Mizukami for providing the landcover dataset file and soiltype files that were used in the initial experiment & development. <br>










