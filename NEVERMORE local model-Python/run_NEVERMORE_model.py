# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:51:23 2030

@author: ivaram
"""

import pysd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import warnings
import time

warnings.filterwarnings('ignore')

start_code = time.time()

#%% 

#Cargar modelo

model = pysd.load('Modelo_NEVERMORE.py')


####------ Selected variables for plotting the results -------####

#Filtramos para simular solo con las variables necesarias
output_vars = [
    "EXPECTED LOSSES BY FLOODS[TEN]",
    "EXPECTED LOSSES BY FLOODS[FIFTY]",
    "EXPECTED LOSSES BY FLOODS[HUNDRED]",
    "SENSITIVITY FOR POPULATION",
    "ADAPTATION FOR POPULATION",
    "RISK INDEX POPULATION",
    "CROPS ADAPTATION",
    "CROPS SENSITIVITY",
    "RISK INDEX CROPS",
    "AGRICULTURE",
    "STORAGE WATER",
    "GROUNDWATER RESOURCES",
    "WATER DEMAND",
    "WATER SUPPLY",
    "WATER SECURITY",
    "WATER DEMAND FROM AGRICULTURE",
    "WATER DEMAND FROM INDUSTRY",
    "REQUEST FROM GROUNDWATER EXTRACTION",
    "FOREST",
    "WETLANDS",
    "IRRIGATED CROP AREA",
    "ATTRACTIVENESS FOR TOURISM",
    "TOTAL INTERNATIONAL VISITORS",
    "TOTAL NATIONAL VISITORS",
    "INFRASTRUCTURE CAPACITY",
    "RISK INDEX TOURISM",
    "TOURISM INCOME",
    "WATER DEMAND FROM URBAN",
    "ENERGY CONSUMPTION",
    "ENERGY CONSUMPTION SHARE[COAL]",
    "ENERGY CONSUMPTION SHARE[GAS]",
    "ENERGY CONSUMPTION SHARE[OIL]",
    "ENERGY CONSUMPTION SHARE[RENEWABLES]",
    "TOTAL CARBON EMISSIONS FROM ENERGY",
    "TOTAL PRODUCTION",
    "TOTAL PRODUCTION CONSIDERING H2",
    "IMPORT",
    "IMPORT WITH H2",
    "EXPORT",
    "EXPORT WITH H2",
    "SOLAR PRODUCTION",
    "WIND PRODUCTION",
    "NON RENEWABLE PRODUCTION",
    "ROOFTOP PRODUCTION",
    "BIOMASS PRODUCTION",
    "HYDRO PRODUCTION",
    "H2 PRODUCTION",
    "TOTAL EMISSIONS FROM TRANSPORT",
    "VEHICLES[TRUCK,PETROL]",
    "VEHICLES[TRUCK,DIESEL]",
    "VEHICLES[TRUCK,ELECTRICITY]",
    "VEHICLES[VAN,PETROL]",
    "VEHICLES[VAN,DIESEL]",
    "VEHICLES[VAN,ELECTRICITY]",
    "VEHICLES[BUS,PETROL]",
    "VEHICLES[BUS,DIESEL]",
    "VEHICLES[BUS,ELECTRICITY]",
    "VEHICLES[CAR,PETROL]",
    "VEHICLES[CAR,DIESEL]",
    "VEHICLES[CAR,ELECTRICITY]",
    "VEHICLES[MOTORCYCLE,PETROL]",
    "VEHICLES[MOTORCYCLE,DIESEL]",
    "VEHICLES[MOTORCYCLE,ELECTRICITY]",
    "VEHICLES[TRACTOR,PETROL]",
    "VEHICLES[TRACTOR,DIESEL]",
    "VEHICLES[TRACTOR,ELECTRICITY]",
    "WATER USED FOR SNOW PRODUCTION",
    "ENERGY USED FOR SNOW PRODUCTION",
    "ARTIFICIAL SNOW PRODUCTION YEARLY",
    "PROTECTED FOREST LAND",
    "PROTECTED WETLANDS",
    "BIOMASS STOCK",
    "TOTAL CARBON STOCK",
    "SOLAR LAND",
    "WATER REQUIREMENTS FROM H2 PRODUCTION",
    "ENERGY REQUESTED FROM H2 PRODUCTION"
]

####------ Model execution -------####

#Use of "timestamps" to define the simulation period
stocks = model.run(return_columns=output_vars, return_timestamps=np.arange(2018, 2061))

#Filter to store the results after 2025 deleting the historical part
stocks_filtrado = stocks[~((stocks.index >= 2018) & (stocks.index < 2025))]

#%% 

####------ Save results -------####

#Guardar resultados
stocks.to_csv('model_results.csv')
stocks.to_excel('model_results.xlsx')

end_code = time.time()
print('T(s):')
print(end_code-start_code)