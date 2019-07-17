# Homophily as a process generating social networks

## Authors

 * Szymon Talaga, <stalaga@protonmail.com>
   * The Robert Zajonc Institute for Social Studies, University of Warsaw
 * Andrzej Nowak
   * The Robert Zajonc Institute for Social Studies, University of Warsaw

# Introduction

This is a code repository for the paper "Homophily as a process generating social networks: insights from Social Distance Attachment model".
It provides all information, code and data necessary to replicate all the simulations and analyses presented in the paper.
This document contains the overall instruction as well as description of the content of the repository.
Details regarding particular stages are documented within source files as comments.

# Requirements

The project uses both Python and R langages.
Code is multiplatform and should run on any standard operating system (i.e. Linux/Unix, MacOS, Windows).

## Python dependencies

 - Python3.6+
 - Other dependencies are specified in the `requirements.txt` file.

   To install them run `pip install -r requirements.txt`.

 - It is a good practice to install python dependencies in an isolated virtual environment.

## R dependencies

 - R version 3.6.1 (2019-07-05) or newer.
   In all probability earlier version will suffice too, but this is the version that the code was tested against.
 - Packages:
   - here
   - Cairo
   - knitr
   - rmarkdown
   - magrittr
   - boot
   - tidyverse
   - feather
   - emmeans
   - broom
   - ggsignif
   - ggpubr
   - gridExtra

# Content of the repository

## Scripts

  - `1_simulate_sda.py`
    - This is the first script to be run. It runs the simulations for SDA model with the same setup as used in the paper.
      Details, including control parameters are documented in the source file.
  - `2_simulate_sdc.py`
    - This is the second script to be run. It runs the simulations for SDC model with the same setup as used in the paper.
      Details, including the control parameters are documented in the source file.
  - `3_appendices_A_D.py`
    - The third script to be run. It generates network visualizations from the appendices A and D. Details are provided in the source file.
  - `4_appendix_C.py`
    - The last script to be run. It computes the table from the appendix C. Details are provided in the source file.

## Python modules

  - `_.py`
    - General module with routines for running simulations.
  - `da.py`
    - Module with utilities for data analysis.
  - `sdnet`
    - Module with implementation of SDA and SDC models and related utilities. Detais are documented in the source files.
  - `tailest`
    - Module implementing methods from Voitalov et al. (2018). Details are documented in the source files.

## Directories

  - `raw-data`
    - Here data as produced in simulations is saved. The directory is automatically created.
  - `data`
    - Here final data prepared for analyses is stored: `sda-data.feather` (SDA model data) `sda-data-cm.feather` (SDC model data).
      It is created automatically during simulations.
  - `figures`
    - It is created by `3_appendices_A_D.py`.
  - `R-sda-paper`
    - This directory stores R code for data analysis and visualization.

## Format of the results

Simulation results are saved as data frames in [feather format](https://blog.rstudio.com/2016/03/29/feather/)

## CoMSES repository

Feather files with original simulation results are available in CoMSES version of the repository.
They are stored in standard `results` subdirectory.
They have to moved to `code/data` directory for R scripts to work without any changes.

# Computation time

Simulations are comptationally expensive and may take several hours even when run on multiple cores in parallel.
