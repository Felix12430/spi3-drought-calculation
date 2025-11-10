# SPI-3 Drought Calculation
## Overview
Google Earth Engine JavaScript code for calculating 3-month Standardized Precipitation Index (SPI-3) from CHIRPS daily rainfall data.
## Features
- Computes 3-month rolling precipitation sums
- Calculates SPI-3 using mean and standard deviation
- Generates balanced training points for Random Forest classification
- Exports results as CSV and GeoTIFF
## Usage
1. Load into Google Earth Engine
2. Define your study area geometry
3. Run the script
4. Export training points and SPI-3 rasters
## Data Sources
- CHIRPS Daily Precipitation (UCSB-CHG/CHIRPS/DAILY)
- Study area: Marsabit County, Kenya (2005-2024)

