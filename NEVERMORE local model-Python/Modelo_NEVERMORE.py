"""
@author: Fundación CARTIF

'NEVERMORE_local_model'
"""

from pathlib import Path
import numpy as np
import xarray as xr

from pysd.py_backend.functions import integer, if_then_else, sum, zidz, modulo
from pysd.py_backend.statefuls import (
    SampleIfTrue,
    Delay,
    Initial,
    Integ,
    DelayFixed,
    Smooth,
)
from pysd.py_backend.external import ExtConstant, ExtData
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


_subscript_dict = {
    "FLOOD DEPTH": [
        "ONE METER",
        "TWO METERS",
        "THREE METERS",
        "FOUR METERS",
        "HIGHER THAN FOUR METERS",
    ],
    "FLOODED LAND": ["URBAN LAND", "OTHER LAND", "RAIL RAILWAYS"],
    "RETURN PERIOD": ["TEN", "FIFTY", "HUNDRED"],
    "VEHICLE AGE": [
        "LESS THAN 4 YEARS",
        "FROM 5 TO 9 YEARS",
        "FROM 10 TO 14 YEARS",
        "FROM 15 TO 19 YEARS",
        "MORE THAN 20 YEARS",
    ],
    "TYPE OF VEHICLE": ["TRUCK", "VAN", "BUS", "CAR", "MOTORCYCLE", "TRACTOR"],
    "TYPE OF FUEL": ["PETROL", "DIESEL", "ELECTRICITY"],
    "SEASONS": ["SPRING", "SUMMER", "AUTUMN", "WINTER"],
    "MONTHS": [
        "OCTOBER",
        "NOVEMBER",
        "DECEMBER",
        "JANUARY",
        "FEBRUARY",
        "MARCH",
        "APRIL",
        "MAY",
        "JUNE",
        "JULY",
        "AUGUST",
        "SEPTEMBER",
    ],
    "TYPE OF ENERGY": ["COAL", "GAS", "OIL", "RENEWABLES"],
    "TYPE OF RENEWABLES": ["SOLAR", "WIND", "ROOFTOP", "BIOMASS", "HYDRO"],
    "CROPS I": [
        "CEREALS",
        "VEGETABLES",
        "FRUITS AND NUTS",
        "OILSEED CROPS",
        "ROOTS AND TUBERS",
        "BEVERAGE AND SPICE CROPS",
        "LEGUMINOUS CROPS",
        "SUGAR CROPS",
        "OTHER CROPS",
    ],
}

component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 2018,
    "final_time": lambda: 2060,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Year",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Year",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="URBAN EXPANSION FROM WETLANDS",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "expansion_ratio": 2,
        "conversion_speed_wetlands_to_urban": 2,
        "urban": 2,
        "protected_wetlands": 1,
        "wetlands": 1,
    },
)
def urban_expansion_from_wetlands():
    return if_then_else(
        ((expansion_ratio() * conversion_speed_wetlands_to_urban()) / 100)
        * urban()
        * 0.1
        > 0,
        lambda: float(
            np.minimum(
                ((expansion_ratio() * conversion_speed_wetlands_to_urban()) / 100)
                * urban()
                * 0.1,
                (wetlands() - protected_wetlands()) / 1,
            )
        ),
        lambda: 0,
    )


@component.add(
    name="WETLANDS DESERTION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"wetlands": 3, "agricultural_pressure": 2, "protected_wetlands": 1},
)
def wetlands_desertion():
    return if_then_else(
        wetlands() * agricultural_pressure() > 0,
        lambda: float(
            np.minimum(
                wetlands() * agricultural_pressure(),
                (wetlands() - protected_wetlands()) / 1,
            )
        ),
        lambda: 0,
    )


@component.add(
    name="PROTECTED WETLANDS NOT CORRECTED",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_protected_wetlands_not_corrected": 1},
    other_deps={
        "_integ_protected_wetlands_not_corrected": {
            "initial": {"initial_protected_wetlands": 1},
            "step": {
                "switch_protected_wetlands": 3,
                "increase_objective_for_protected_wetlands": 1,
                "initial_year_for_protected_wetlands": 2,
                "time": 2,
                "final_year_for_protected_wetlands": 2,
            },
        }
    },
)
def protected_wetlands_not_corrected():
    """
    IF THEN ELSE(SWITCH PROTECTED WETLANDS=0,0,IF THEN ELSE(SWITCH PROTECTED WETLANDS =1:AND:(Time<INITIAL YEAR FOR PROTECTED WETLANDS),0, IF THEN ELSE(SWITCH PROTECTED WETLANDS=1:AND:(Time>FINAL YEAR FOR PROTECTED WETLANDS-1),0, MIN((INCREASE OBJECTIVE FOR PROTECTED WETLANDS/(FINAL YEAR FOR PROTECTED WETLANDS-INITIAL YEAR FOR PROTECTED WETLANDS )),DELAY AVAILABLE LAND))))
    """
    return _integ_protected_wetlands_not_corrected()


_integ_protected_wetlands_not_corrected = Integ(
    lambda: if_then_else(
        switch_protected_wetlands() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_protected_wetlands() == 1,
                time() < initial_year_for_protected_wetlands(),
            ),
            lambda: 0,
            lambda: if_then_else(
                np.logical_and(
                    switch_protected_wetlands() == 1,
                    time() > final_year_for_protected_wetlands() - 1,
                ),
                lambda: 0,
                lambda: increase_objective_for_protected_wetlands()
                / (
                    final_year_for_protected_wetlands()
                    - initial_year_for_protected_wetlands()
                ),
            ),
        ),
    ),
    lambda: initial_protected_wetlands(),
    "_integ_protected_wetlands_not_corrected",
)


@component.add(
    name="PROTECTED FOREST LAND",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"protected_forest_land_not_corrected": 2, "forest": 2},
)
def protected_forest_land():
    return if_then_else(
        protected_forest_land_not_corrected() >= forest(),
        lambda: forest(),
        lambda: protected_forest_land_not_corrected(),
    )


@component.add(
    name="AVAILABLE FOREST LAND",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"forest": 1, "protected_forest_land": 1},
)
def available_forest_land():
    return forest() - protected_forest_land()


@component.add(
    name="AVAILABLE LAND",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"wetlands": 1, "protected_wetlands": 1},
)
def available_land():
    return wetlands() - protected_wetlands()


@component.add(
    name="PROTECTED FOREST LAND NOT CORRECTED",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_protected_forest_land_not_corrected": 1},
    other_deps={
        "_integ_protected_forest_land_not_corrected": {
            "initial": {"initial_forest_protected_land": 1},
            "step": {
                "switch_protected_forest_land": 3,
                "increase_objective_for_protected_forest_land": 1,
                "initial_year_for_protected_forest_land": 2,
                "time": 2,
                "final_year_for_protected_forest_land": 2,
            },
        }
    },
)
def protected_forest_land_not_corrected():
    return _integ_protected_forest_land_not_corrected()


_integ_protected_forest_land_not_corrected = Integ(
    lambda: if_then_else(
        switch_protected_forest_land() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_protected_forest_land() == 1,
                time() < initial_year_for_protected_forest_land(),
            ),
            lambda: 0,
            lambda: if_then_else(
                np.logical_and(
                    switch_protected_forest_land() == 1,
                    time() > final_year_for_protected_forest_land() - 1,
                ),
                lambda: 0,
                lambda: increase_objective_for_protected_forest_land()
                / (
                    final_year_for_protected_forest_land()
                    - initial_year_for_protected_forest_land()
                ),
            ),
        ),
    ),
    lambda: initial_forest_protected_land(),
    "_integ_protected_forest_land_not_corrected",
)


@component.add(
    name="PROTECTED WETLANDS",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"protected_wetlands_not_corrected": 2, "wetlands": 2},
)
def protected_wetlands():
    return if_then_else(
        protected_wetlands_not_corrected() >= wetlands(),
        lambda: wetlands(),
        lambda: protected_wetlands_not_corrected(),
    )


@component.add(
    name="ARTIFICIAL SNOW PRODUCTION MONTHLY",
    units="m3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_effects_of_artificial_snow_production_in_tourism": 1,
        "initial_year_sustainable_snowmaking_and_conservation_practices": 3,
        "time": 3,
        "final_year_sustainable_snowmaking_and_conservation_practices": 2,
        "ski_slope_area": 5,
        "volume_snow_accumulation": 4,
        "sustainable_snowmaking_and_conservation_practices_objective": 1,
        "switch_sustainable_snowmaking_and_conservation_practices": 2,
        "snow_volume_for_skiing": 4,
    },
)
def artificial_snow_production_monthly():
    """
    IF THEN ELSE(SWITCH EFFECTS OF ARTIFICIAL SNOW PRODUCTION IN TOURISM=0,0,IF THEN ELSE(MONTHS=MAY:OR:MONTHS=JUNE:OR:MONTHS =JULY:OR:MONTHS=AUGUST:OR:MONTHS=SEPTEMBER:OR:MONTHS=OCTOBER,0,IF THEN ELSE ((SNOW VOLUME FOR SKIING-VOLUME SNOW ACCUMULATION[MONTHS])<0,SKI SLOPE AREA*0.2,IF THEN ELSE(SWITCH SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES =0,(SNOW VOLUME FOR SKIING -VOLUME SNOW ACCUMULATION[MONTHS])+SKI SLOPE AREA*0.2,IF THEN ELSE(SWITCH SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES=1:AND:Time<=INITIAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES :OR:Time>=FINAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES ,SKI SLOPE AREA*0.2+(SNOW VOLUME FOR SKIING-VOLUME SNOW ACCUMULATION[MONTHS]),MAX(SKI SLOPE AREA*0.2+(SNOW VOLUME FOR SKIING-VOLUME SNOW ACCUMULATION[MONTHS])*SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES OBJECTIVE*((Time-INITIAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES)/(FINAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES -INITIAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES))),SKI SLOPE AREA*0.2)))))
    """
    return if_then_else(
        switch_effects_of_artificial_snow_production_in_tourism() == 0,
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 8,
                np.logical_or(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 9,
                    np.logical_or(
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        )
                        == 10,
                        np.logical_or(
                            xr.DataArray(
                                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            )
                            == 11,
                            np.logical_or(
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 12,
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 1,
                            ),
                        ),
                    ),
                ),
            ),
            lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
            lambda: if_then_else(
                snow_volume_for_skiing() - volume_snow_accumulation() < 0,
                lambda: xr.DataArray(
                    ski_slope_area() * 0.2,
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                ),
                lambda: if_then_else(
                    switch_sustainable_snowmaking_and_conservation_practices() == 0,
                    lambda: (snow_volume_for_skiing() - volume_snow_accumulation())
                    + ski_slope_area() * 0.2,
                    lambda: if_then_else(
                        np.logical_and(
                            switch_sustainable_snowmaking_and_conservation_practices()
                            == 1,
                            np.logical_or(
                                time()
                                <= initial_year_sustainable_snowmaking_and_conservation_practices(),
                                time()
                                >= final_year_sustainable_snowmaking_and_conservation_practices(),
                            ),
                        ),
                        lambda: ski_slope_area() * 0.2
                        + (snow_volume_for_skiing() - volume_snow_accumulation()),
                        lambda: np.maximum(
                            ski_slope_area() * 0.2
                            + (snow_volume_for_skiing() - volume_snow_accumulation())
                            * (
                                1
                                - sustainable_snowmaking_and_conservation_practices_objective()
                            )
                            * (
                                (
                                    time()
                                    - initial_year_sustainable_snowmaking_and_conservation_practices()
                                )
                                / (
                                    final_year_sustainable_snowmaking_and_conservation_practices()
                                    - initial_year_sustainable_snowmaking_and_conservation_practices()
                                )
                            ),
                            ski_slope_area() * 0.2,
                        ),
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="DAMAGE BY DEPTH IN ROADS RAILWAYS",
    units="km2",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"roads_railways_damage_at_depth": 1, "flood_area_roads_railways": 1},
)
def damage_by_depth_in_roads_railways():
    return (
        roads_railways_damage_at_depth()
        * flood_area_roads_railways().transpose("FLOOD DEPTH", "RETURN PERIOD")
    ).transpose("RETURN PERIOD", "FLOOD DEPTH")


@component.add(
    name="DAMAGE BY DEPTH IN URBAN",
    units="km2",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_policy_flood_defenses_for_urban_areas": 2,
        "urban_damage_at_depth": 2,
        "flood_area_urban": 3,
        "urban_damage_at_depth_considering_flood_defenses": 1,
    },
)
def damage_by_depth_in_urban():
    return if_then_else(
        switch_policy_flood_defenses_for_urban_areas() == 0,
        lambda: urban_damage_at_depth()
        * flood_area_urban().transpose("FLOOD DEPTH", "RETURN PERIOD"),
        lambda: if_then_else(
            switch_policy_flood_defenses_for_urban_areas() == 1,
            lambda: urban_damage_at_depth_considering_flood_defenses()
            * flood_area_urban().transpose("FLOOD DEPTH", "RETURN PERIOD"),
            lambda: urban_damage_at_depth()
            * flood_area_urban().transpose("FLOOD DEPTH", "RETURN PERIOD"),
        ),
    ).transpose("RETURN PERIOD", "FLOOD DEPTH")


@component.add(
    name="DAMAGE BY DEPTH IN OTHER",
    units="km2",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"other_damage_at_depth": 1, "flood_area_other": 1},
)
def damage_by_depth_in_other():
    return (
        other_damage_at_depth()
        * flood_area_other().transpose("FLOOD DEPTH", "RETURN PERIOD")
    ).transpose("RETURN PERIOD", "FLOOD DEPTH")


@component.add(
    name="NEW INFRASTRUCTURE",
    units="Infrastructure",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_investment_in_touristic_infrastructures": 2,
        "time": 2,
        "initial_year_investment_in_touristic_infrastructures": 2,
        "infrastructure_capacity": 1,
        "final_year_investment_in_touristic_infrastructures": 2,
        "investment_in_touristic_infrastructure_objectives": 1,
    },
)
def new_infrastructure():
    """
    IF THEN ELSE(SWITCH INVESTMENT IN INFRASTRUCTURE=0,0,IF THEN ELSE(SWITCH INVESTMENT IN INFRASTRUCTURE=1:AND:Time<INITIAL YEAR INVESTMENT IN INFRASTRUCTURE :OR:Time>FINAL YEAR INVESTMENT IN INFRASTRUCTURE,0,(INVESTMENT IN INFRASTRUCTURE OBJECTIVE/(FINAL YEAR INVESTMENT IN INFRASTRUCTURE -INITIAL YEAR INVESTMENT IN INFRASTRUCTURE))*(TOTAL VISITORS-INFRASTRUCTURE CAPACITY)))
    """
    return if_then_else(
        switch_investment_in_touristic_infrastructures() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_investment_in_touristic_infrastructures() == 1,
                np.logical_or(
                    time() < initial_year_investment_in_touristic_infrastructures(),
                    time() > final_year_investment_in_touristic_infrastructures(),
                ),
            ),
            lambda: 0,
            lambda: (
                investment_in_touristic_infrastructure_objectives()
                / (
                    final_year_investment_in_touristic_infrastructures()
                    - initial_year_investment_in_touristic_infrastructures()
                )
            )
            * infrastructure_capacity(),
        ),
    )


@component.add(
    name="FINAL YEAR INVESTMENT IN TOURISTIC INFRASTRUCTURES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_investment_in_touristic_infrastructures"
    },
)
def final_year_investment_in_touristic_infrastructures():
    return _ext_constant_final_year_investment_in_touristic_infrastructures()


_ext_constant_final_year_investment_in_touristic_infrastructures = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "FINAL_YEAR_INVESTMENT_IN_INFRASTRUCTURE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_investment_in_touristic_infrastructures",
)


@component.add(
    name="INITIAL YEAR INVESTMENT IN TOURISTIC INFRASTRUCTURES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_investment_in_touristic_infrastructures"
    },
)
def initial_year_investment_in_touristic_infrastructures():
    return _ext_constant_initial_year_investment_in_touristic_infrastructures()


_ext_constant_initial_year_investment_in_touristic_infrastructures = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "INITIAL_YEAR_INVESTMENT_IN_INFRASTRUCTURE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_investment_in_touristic_infrastructures",
)


@component.add(
    name="SWITCH INVESTMENT IN TOURISTIC INFRASTRUCTURES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def switch_investment_in_touristic_infrastructures():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Tourism system', 'SWITCH_INVESTMENT_IN_INFRASTRUCTURE*')
    """
    return 1


@component.add(
    name="MEAN HISTORICAL TCI",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="SampleIfTrue",
    depends_on={"_sampleiftrue_mean_historical_tci": 1},
    other_deps={
        "_sampleiftrue_mean_historical_tci": {
            "initial": {"mean_tci": 1},
            "step": {"time": 1, "mean_tci": 1},
        }
    },
)
def mean_historical_tci():
    return _sampleiftrue_mean_historical_tci()


_sampleiftrue_mean_historical_tci = SampleIfTrue(
    lambda: time() == 2018,
    lambda: mean_tci(),
    lambda: mean_tci(),
    "_sampleiftrue_mean_historical_tci",
)


@component.add(
    name='"POPULATION <5 NORMALIZED"',
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_of_population_5": 5},
)
def population_5_normalized():
    return if_then_else(
        ratio_of_population_5() <= 0.04,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(
                ratio_of_population_5() <= 0.06, ratio_of_population_5() > 0.04
            ),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(
                    ratio_of_population_5() <= 0.08, ratio_of_population_5() > 0.06
                ),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="MORTALITY RATE NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"mortality_rate": 5},
)
def mortality_rate_normalized():
    return if_then_else(
        mortality_rate() <= 0.008,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(mortality_rate() <= 0.01, mortality_rate() > 0.008),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(mortality_rate() <= 0.012, mortality_rate() > 0.01),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="POPULATION BASELINE SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "population_65_normalized": 1,
        "population_5_normalized": 1,
        "sensitivity_from_density": 1,
        "mortality_rate_normalized": 1,
        "birth_rate_normalized": 1,
        "green_urban_area_normalized": 1,
        "unemployment_normalized": 1,
        "energy_poverty_normalized": 1,
        "health_services_normalized": 1,
    },
)
def population_baseline_sensitivity():
    """
    He quitado la media por ahora
    """
    return (
        population_65_normalized()
        + population_5_normalized()
        + sensitivity_from_density()
        + mortality_rate_normalized()
        + birth_rate_normalized()
        + green_urban_area_normalized()
        + unemployment_normalized()
        + energy_poverty_normalized()
        + health_services_normalized()
    ) / 9


@component.add(
    name="HEALTH SERVICES NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"health_services": 5},
)
def health_services_normalized():
    return if_then_else(
        health_services() >= 0.005,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(health_services() < 0.005, health_services() >= 0.003),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(health_services() < 0.003, health_services() >= 0.001),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name='"POPULATION >65 NORMALIZED"',
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_of_population_65": 5},
)
def population_65_normalized():
    return if_then_else(
        ratio_of_population_65() <= 0.15,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(
                ratio_of_population_65() <= 0.2, ratio_of_population_65() > 0.15
            ),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(
                    ratio_of_population_65() <= 0.25, ratio_of_population_65() > 0.2
                ),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="UNEMPLOYMENT NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"unemployment_rate": 5},
)
def unemployment_normalized():
    return if_then_else(
        unemployment_rate() <= 0.05,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(unemployment_rate() <= 0.1, unemployment_rate() > 0.05),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(unemployment_rate() <= 0.15, unemployment_rate() > 0.1),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="BIRTH RATE NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"birth_rate": 5},
)
def birth_rate_normalized():
    return if_then_else(
        birth_rate() >= 0.012,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(birth_rate() < 0.012, birth_rate() >= 0.01),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(birth_rate() < 0.01, birth_rate() >= 0.008),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="ENERGY POVERTY NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_energy_poverty": 5},
)
def energy_poverty_normalized():
    return if_then_else(
        ratio_energy_poverty() <= 0.05,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(
                ratio_energy_poverty() <= 0.1, ratio_energy_poverty() > 0.05
            ),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(
                    ratio_energy_poverty() <= 0.2, ratio_energy_poverty() > 0.1
                ),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="GREEN URBAN AREA NORMALIZED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_of_green_urban_area": 5},
)
def green_urban_area_normalized():
    return if_then_else(
        ratio_of_green_urban_area() >= 0.25,
        lambda: 0.25,
        lambda: if_then_else(
            np.logical_and(
                ratio_of_green_urban_area() < 0.25, ratio_of_green_urban_area() >= 0.15
            ),
            lambda: 0.5,
            lambda: if_then_else(
                np.logical_and(
                    ratio_of_green_urban_area() < 0.15,
                    ratio_of_green_urban_area() >= 0.05,
                ),
                lambda: 0.75,
                lambda: 1,
            ),
        ),
    )


@component.add(
    name="TEMPERATURE",
    units="ºc",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="Normal",
    depends_on={
        "select_temperature_scenario": 2,
        "mean_temperature_ssp245": 1,
        "mean_temperature_ssp585": 1,
    },
)
def temperature():
    return if_then_else(
        select_temperature_scenario() == 0,
        lambda: mean_temperature_ssp245(),
        lambda: if_then_else(
            select_temperature_scenario() == 1,
            lambda: mean_temperature_ssp585(),
            lambda: xr.DataArray(1, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        ),
    )


@component.add(
    name="RATIO WATER PER SNOW PRODUCTION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_technical_improvements_in_snow_production": 3,
        "initial_ratio_water_per_snow_production": 4,
        "final_year_technical_improvements_in_snow_production": 2,
        "time": 3,
        "water_use_decrease_in_snow_production_objective": 2,
        "initial_year_technical_improvements_in_snow_production": 3,
    },
)
def ratio_water_per_snow_production():
    return if_then_else(
        switch_technical_improvements_in_snow_production() == 0,
        lambda: initial_ratio_water_per_snow_production(),
        lambda: if_then_else(
            np.logical_and(
                switch_technical_improvements_in_snow_production() == 1,
                time() <= initial_year_technical_improvements_in_snow_production(),
            ),
            lambda: initial_ratio_water_per_snow_production(),
            lambda: if_then_else(
                np.logical_and(
                    switch_technical_improvements_in_snow_production() == 1,
                    time() >= final_year_technical_improvements_in_snow_production(),
                ),
                lambda: initial_ratio_water_per_snow_production()
                * (1 - water_use_decrease_in_snow_production_objective()),
                lambda: float(
                    np.maximum(
                        0,
                        initial_ratio_water_per_snow_production()
                        * (
                            1
                            - water_use_decrease_in_snow_production_objective()
                            * (
                                (
                                    time()
                                    - initial_year_technical_improvements_in_snow_production()
                                )
                                / (
                                    final_year_technical_improvements_in_snow_production()
                                    - initial_year_technical_improvements_in_snow_production()
                                )
                            )
                        ),
                    )
                ),
            ),
        ),
    )


@component.add(
    name="NATIONAL VISITORS WINTER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_national_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def national_visitors_winter():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_national_visitors().loc["WINTER"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_national_visitors().loc["WINTER"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_national_visitors().loc["WINTER"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_national_visitors().loc["WINTER"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="RATIO ENERGY PER SNOW PRODUCTION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_technical_improvements_in_snow_production": 3,
        "initial_ratio_energy_per_snow_production": 4,
        "energy_use_decrease_in_snow_production_objective": 2,
        "time": 3,
        "final_year_technical_improvements_in_snow_production": 2,
        "initial_year_technical_improvements_in_snow_production": 3,
    },
)
def ratio_energy_per_snow_production():
    return if_then_else(
        switch_technical_improvements_in_snow_production() == 0,
        lambda: initial_ratio_energy_per_snow_production(),
        lambda: if_then_else(
            np.logical_and(
                switch_technical_improvements_in_snow_production() == 1,
                time() <= initial_year_technical_improvements_in_snow_production(),
            ),
            lambda: initial_ratio_energy_per_snow_production(),
            lambda: if_then_else(
                np.logical_and(
                    switch_technical_improvements_in_snow_production() == 1,
                    time() >= final_year_technical_improvements_in_snow_production(),
                ),
                lambda: initial_ratio_energy_per_snow_production()
                * (1 - energy_use_decrease_in_snow_production_objective()),
                lambda: float(
                    np.maximum(
                        0,
                        initial_ratio_energy_per_snow_production()
                        * (
                            1
                            - energy_use_decrease_in_snow_production_objective()
                            * (
                                (
                                    time()
                                    - initial_year_technical_improvements_in_snow_production()
                                )
                                / (
                                    final_year_technical_improvements_in_snow_production()
                                    - initial_year_technical_improvements_in_snow_production()
                                )
                            )
                        ),
                    )
                ),
            ),
        ),
    )


@component.add(
    name="INITIAL NATIONAL VISITORS WINTER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_effects_of_artificial_snow_production_in_tourism": 1,
        "volume_snow_accumulation_yearly": 1,
        "snow_volume_for_skiing_yearly": 1,
        "time": 6,
        "final_year_promote_domestic_tourism": 4,
        "alfa_winter": 8,
        "mean_tci_winter": 8,
        "beta_winter": 8,
        "initial_year_promote_domestic_tourism": 6,
        "switch_promote_domestic_tourism": 6,
        "promote_domestic_tourism_objective": 4,
    },
)
def initial_national_visitors_winter():
    return if_then_else(
        np.logical_and(
            switch_effects_of_artificial_snow_production_in_tourism() == 0,
            snow_volume_for_skiing_yearly() > volume_snow_accumulation_yearly(),
        ),
        lambda: 0.8
        * if_then_else(
            switch_promote_domestic_tourism() == 0,
            lambda: alfa_winter() * mean_tci_winter() + beta_winter(),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_domestic_tourism() == 1,
                    time() < initial_year_promote_domestic_tourism(),
                ),
                lambda: alfa_winter() * mean_tci_winter() + beta_winter(),
                lambda: if_then_else(
                    np.logical_and(
                        switch_promote_domestic_tourism() == 1,
                        time() > final_year_promote_domestic_tourism(),
                    ),
                    lambda: (alfa_winter() * mean_tci_winter() + beta_winter())
                    * (1 + float(promote_domestic_tourism_objective().loc["WINTER"])),
                    lambda: (alfa_winter() * mean_tci_winter() + beta_winter())
                    * (
                        1
                        + float(promote_domestic_tourism_objective().loc["WINTER"])
                        * (
                            (time() - initial_year_promote_domestic_tourism())
                            / (
                                final_year_promote_domestic_tourism()
                                - initial_year_promote_domestic_tourism()
                            )
                        )
                    ),
                ),
            ),
        ),
        lambda: if_then_else(
            switch_promote_domestic_tourism() == 0,
            lambda: alfa_winter() * mean_tci_winter() + beta_winter(),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_domestic_tourism() == 1,
                    time() < initial_year_promote_domestic_tourism(),
                ),
                lambda: alfa_winter() * mean_tci_winter() + beta_winter(),
                lambda: if_then_else(
                    np.logical_and(
                        switch_promote_domestic_tourism() == 1,
                        time() > final_year_promote_domestic_tourism(),
                    ),
                    lambda: (alfa_winter() * mean_tci_winter() + beta_winter())
                    * (1 + float(promote_domestic_tourism_objective().loc["WINTER"])),
                    lambda: (alfa_winter() * mean_tci_winter() + beta_winter())
                    * (
                        1
                        + float(promote_domestic_tourism_objective().loc["WINTER"])
                        * (
                            (time() - initial_year_promote_domestic_tourism())
                            / (
                                final_year_promote_domestic_tourism()
                                - initial_year_promote_domestic_tourism()
                            )
                        )
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="WATER USE DECREASE IN SNOW PRODUCTION OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_water_use_decrease_in_snow_production_objective"
    },
)
def water_use_decrease_in_snow_production_objective():
    """
    e.g.decrease of 60%
    """
    return _ext_constant_water_use_decrease_in_snow_production_objective()


_ext_constant_water_use_decrease_in_snow_production_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "WATER_USE_DECREASE_IN_SNOW_PRODUCTION_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_water_use_decrease_in_snow_production_objective",
)


@component.add(
    name="WATER USED FOR SNOW PRODUCTION",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "artificial_snow_production_yearly": 1,
        "ratio_water_per_snow_production": 1,
    },
)
def water_used_for_snow_production():
    return (
        artificial_snow_production_yearly() * ratio_water_per_snow_production()
    ) / 1000000.0


@component.add(
    name="INITIAL YEAR TECHNICAL IMPROVEMENTS IN SNOW PRODUCTION",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_technical_improvements_in_snow_production"
    },
)
def initial_year_technical_improvements_in_snow_production():
    return _ext_constant_initial_year_technical_improvements_in_snow_production()


_ext_constant_initial_year_technical_improvements_in_snow_production = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_TECHNICAL_IMPROVEMENTS_IN_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_technical_improvements_in_snow_production",
)


@component.add(
    name="INITIAL RATIO WATER PER SNOW PRODUCTION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_ratio_water_per_snow_production"
    },
)
def initial_ratio_water_per_snow_production():
    """
    m3 water over m3 snow
    """
    return _ext_constant_initial_ratio_water_per_snow_production()


_ext_constant_initial_ratio_water_per_snow_production = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_RATIO_WATER_PER_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_water_per_snow_production",
)


@component.add(
    name="ENERGY USED FOR SNOW PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "artificial_snow_production_yearly": 1,
        "ratio_energy_per_snow_production": 1,
    },
)
def energy_used_for_snow_production():
    return artificial_snow_production_yearly() * ratio_energy_per_snow_production()


@component.add(
    name="INTERNATIONAL VISITORS WINTER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_international_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def international_visitors_winter():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_international_visitors().loc["WINTER"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_international_visitors().loc["WINTER"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_international_visitors().loc["WINTER"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_international_visitors().loc["WINTER"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY USE DECREASE IN SNOW PRODUCTION OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_energy_use_decrease_in_snow_production_objective"
    },
)
def energy_use_decrease_in_snow_production_objective():
    return _ext_constant_energy_use_decrease_in_snow_production_objective()


_ext_constant_energy_use_decrease_in_snow_production_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "ENERGY_USE_DECREASE_IN_SNOW_PRODUCTION_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_energy_use_decrease_in_snow_production_objective",
)


@component.add(
    name="FINAL YEAR TECHNICAL IMPROVEMENTS IN SNOW PRODUCTION",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_technical_improvements_in_snow_production"
    },
)
def final_year_technical_improvements_in_snow_production():
    return _ext_constant_final_year_technical_improvements_in_snow_production()


_ext_constant_final_year_technical_improvements_in_snow_production = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_TECHNICAL_IMPROVEMENTS_IN_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_final_year_technical_improvements_in_snow_production",
)


@component.add(
    name="SNOW VOLUME FOR SKIING YEARLY",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"snow_volume_for_skiing": 1},
)
def snow_volume_for_skiing_yearly():
    return snow_volume_for_skiing() * 5


@component.add(
    name="VOLUME SNOW ACCUMULATION YEARLY",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"volume_snow_accumulation": 1},
)
def volume_snow_accumulation_yearly():
    return sum(
        volume_snow_accumulation().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]
    )


@component.add(
    name="SWITCH EFFECTS OF ARTIFICIAL SNOW PRODUCTION IN TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_effects_of_artificial_snow_production_in_tourism"
    },
)
def switch_effects_of_artificial_snow_production_in_tourism():
    """
    0 = OFF 1 = ON
    """
    return _ext_constant_switch_effects_of_artificial_snow_production_in_tourism()


_ext_constant_switch_effects_of_artificial_snow_production_in_tourism = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_ARTIFICIAL_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_switch_effects_of_artificial_snow_production_in_tourism",
)


@component.add(
    name="INITIAL INTERNATIONAL VISITORS WINTER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_effects_of_artificial_snow_production_in_tourism": 1,
        "volume_snow_accumulation_yearly": 1,
        "snow_volume_for_skiing_yearly": 1,
        "mean_tci_winter": 2,
        "alfa_1_winter": 2,
        "beta_1_winter": 2,
    },
)
def initial_international_visitors_winter():
    return if_then_else(
        np.logical_and(
            switch_effects_of_artificial_snow_production_in_tourism() == 0,
            snow_volume_for_skiing_yearly() > volume_snow_accumulation_yearly(),
        ),
        lambda: 0.8 * alfa_1_winter() * mean_tci_winter() + beta_1_winter(),
        lambda: alfa_1_winter() * mean_tci_winter() + beta_1_winter(),
    )


@component.add(
    name="SWITCH TECHNICAL IMPROVEMENTS IN SNOW PRODUCTION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_technical_improvements_in_snow_production"
    },
)
def switch_technical_improvements_in_snow_production():
    """
    0 = OFF 1 = ON
    """
    return _ext_constant_switch_technical_improvements_in_snow_production()


_ext_constant_switch_technical_improvements_in_snow_production = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_TECHNICAL_IMPROVEMENTS_IN_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_switch_technical_improvements_in_snow_production",
)


@component.add(
    name="IRRIGATED CROP AREA",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"agriculture_no_solar": 1, "ratio_irrigated_crop_area": 1},
)
def irrigated_crop_area():
    return agriculture_no_solar() * ratio_irrigated_crop_area()


@component.add(
    name="RATIO RAINFED AGRICULTURE TO SOLAR",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_rainfed_agriculture_to_solar"},
)
def ratio_rainfed_agriculture_to_solar():
    return _ext_constant_ratio_rainfed_agriculture_to_solar()


_ext_constant_ratio_rainfed_agriculture_to_solar = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_RAINFED_AGRICULTURE_TO_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_ratio_rainfed_agriculture_to_solar",
)


@component.add(
    name="MAX SOLAR CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"max_potential_solar": 1, "rainfed_crop_area": 1, "other": 1},
)
def max_solar_capacity():
    return max_potential_solar() * (other() + rainfed_crop_area())


@component.add(
    name="AGRICULTURE NO SOLAR",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agriculture": 1,
        "ratio_rainfed_agriculture_to_solar": 1,
        "variation_of_solar_land": 1,
    },
)
def agriculture_no_solar():
    return (
        agriculture() + ratio_rainfed_agriculture_to_solar() * variation_of_solar_land()
    )


@component.add(
    name="RAINFED CROP AREA",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agriculture_no_solar": 1,
        "ratio_rainfed_crop_area": 1,
        "ratio_rainfed_agriculture_to_solar": 1,
        "variation_of_solar_land": 1,
    },
)
def rainfed_crop_area():
    return (
        agriculture_no_solar() * ratio_rainfed_crop_area()
        - ratio_rainfed_agriculture_to_solar() * variation_of_solar_land()
    )


@component.add(
    name="WATER",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_water": 1},
    other_deps={
        "_integ_water": {
            "initial": {"initial_water_area": 1},
            "step": {
                "agriculture_to_water": 1,
                "forest_to_water": 1,
                "urban_to_water": 1,
                "other_to_water": 1,
            },
        }
    },
)
def water():
    return _integ_water()


_integ_water = Integ(
    lambda: agriculture_to_water()
    + forest_to_water()
    + urban_to_water()
    + other_to_water(),
    lambda: initial_water_area(),
    "_integ_water",
)


@component.add(
    name="OTHER",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_other": 1},
    other_deps={
        "_integ_other": {
            "initial": {"initial_other_area": 1},
            "step": {
                "urban_expansion": 1,
                "other_to_water": 1,
                "variation_of_solar_land": 1,
                "ratio_other_to_solar": 1,
            },
        }
    },
)
def other():
    return _integ_other()


_integ_other = Integ(
    lambda: -urban_expansion()
    - other_to_water()
    - ratio_other_to_solar() * variation_of_solar_land(),
    lambda: initial_other_area(),
    "_integ_other",
)


@component.add(
    name="DELAY SOLAR LAND",
    units="km2",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_solar_land": 1},
    other_deps={
        "_delayfixed_delay_solar_land": {
            "initial": {"solar_land": 1},
            "step": {"solar_land": 1},
        }
    },
)
def delay_solar_land():
    return _delayfixed_delay_solar_land()


_delayfixed_delay_solar_land = DelayFixed(
    lambda: solar_land(),
    lambda: 1,
    lambda: solar_land(),
    time_step,
    "_delayfixed_delay_solar_land",
)


@component.add(
    name="AGRICULTURE",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_agriculture": 1},
    other_deps={
        "_integ_agriculture": {
            "initial": {"initial_agricultural_area": 1},
            "step": {
                "agricultural_expansion": 1,
                "wetlands_desertion": 1,
                "afforestation": 1,
                "agriculture_to_water": 1,
                "urbanization": 1,
                "ratio_rainfed_agriculture_to_solar": 1,
                "variation_of_solar_land": 1,
            },
        }
    },
)
def agriculture():
    return _integ_agriculture()


_integ_agriculture = Integ(
    lambda: agricultural_expansion()
    + wetlands_desertion()
    - afforestation()
    - agriculture_to_water()
    - urbanization()
    - ratio_rainfed_agriculture_to_solar() * variation_of_solar_land(),
    lambda: initial_agricultural_area(),
    "_integ_agriculture",
)


@component.add(
    name="OTHER TO WATER",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"select_location_water_storage": 1, "water_area": 1},
)
def other_to_water():
    return if_then_else(
        select_location_water_storage() == 1, lambda: water_area(), lambda: 0
    )


@component.add(
    name="VARIATION OF SOLAR LAND",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"solar_land": 1, "delay_solar_land": 1},
)
def variation_of_solar_land():
    return solar_land() - delay_solar_land()


@component.add(
    name="RATIO OTHER TO SOLAR",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_other_to_solar"},
)
def ratio_other_to_solar():
    return _ext_constant_ratio_other_to_solar()


_ext_constant_ratio_other_to_solar = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_OTHER_TO_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_ratio_other_to_solar",
)


@component.add(
    name="URBAN",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_urban": 1},
    other_deps={
        "_integ_urban": {
            "initial": {"initial_urban_area": 1, "initial_solar_land": 1},
            "step": {
                "urban_expansion": 1,
                "urban_expansion_from_wetlands": 1,
                "urbanization": 1,
                "urban_to_water": 1,
            },
        }
    },
)
def urban():
    return _integ_urban()


_integ_urban = Integ(
    lambda: urban_expansion()
    + urban_expansion_from_wetlands()
    + urbanization()
    - urban_to_water(),
    lambda: initial_urban_area() - initial_solar_land(),
    "_integ_urban",
)


@component.add(
    name="INITIAL SOLAR LAND",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Initial",
    depends_on={"_initial_initial_solar_land": 1},
    other_deps={
        "_initial_initial_solar_land": {"initial": {"solar_land": 1}, "step": {}}
    },
)
def initial_solar_land():
    return _initial_initial_solar_land()


_initial_initial_solar_land = Initial(
    lambda: solar_land(), "_initial_initial_solar_land"
)


@component.add(
    name="INITIAL BUILT UP URBAN",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Initial",
    depends_on={"_initial_initial_built_up_urban": 1},
    other_deps={
        "_initial_initial_built_up_urban": {
            "initial": {"built_up_urban": 1},
            "step": {},
        }
    },
)
def initial_built_up_urban():
    return _initial_initial_built_up_urban()


_initial_initial_built_up_urban = Initial(
    lambda: built_up_urban(), "_initial_initial_built_up_urban"
)


@component.add(
    name="INITIAL EXPOSED CROPS",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_exposed_crops"},
)
def initial_exposed_crops():
    return _ext_constant_initial_exposed_crops()


_ext_constant_initial_exposed_crops = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "INITIAL_EXPOSED_CROPS*",
    {},
    _root,
    {},
    "_ext_constant_initial_exposed_crops",
)


@component.add(
    name="INITIAL EXPOSED POPULATION",
    units="Inhabitants",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_exposed_population"},
)
def initial_exposed_population():
    return _ext_constant_initial_exposed_population()


_ext_constant_initial_exposed_population = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "INITIAL_EXPOSED_POPULATION*",
    {},
    _root,
    {},
    "_ext_constant_initial_exposed_population",
)


@component.add(
    name="MAX ROOFTOP CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "building_area": 1,
        "initial_built_up_urban": 1,
        "built_up_urban": 1,
        "max_potential_rooftop": 1,
    },
)
def max_rooftop_capacity():
    return (
        (building_area() / initial_built_up_urban())
        * built_up_urban()
        * max_potential_rooftop()
    )


@component.add(
    name="SOLAR LAND",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"solar_capacity": 1, "area_occupied_by_solar_per_mw": 1},
)
def solar_land():
    return solar_capacity() * area_occupied_by_solar_per_mw()


@component.add(
    name="EXPOSED CROPS",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_exposed_crops": 1},
    other_deps={
        "_integ_exposed_crops": {
            "initial": {"initial_exposed_crops": 1},
            "step": {"increase_exposed_crops": 1},
        }
    },
)
def exposed_crops():
    return _integ_exposed_crops()


_integ_exposed_crops = Integ(
    lambda: increase_exposed_crops(),
    lambda: initial_exposed_crops(),
    "_integ_exposed_crops",
)


@component.add(
    name="AREA OCCUPIED BY SOLAR PER MW",
    units="km2/MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_area_occupied_by_solar_per_mw"},
)
def area_occupied_by_solar_per_mw():
    return _ext_constant_area_occupied_by_solar_per_mw()


_ext_constant_area_occupied_by_solar_per_mw = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "AREA_OCCUPIED_BY_SOLAR_PER_MW*",
    {},
    _root,
    {},
    "_ext_constant_area_occupied_by_solar_per_mw",
)


@component.add(
    name="EXPOSED POPULATION",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_exposed_population": 1},
    other_deps={
        "_integ_exposed_population": {
            "initial": {"initial_exposed_population": 1},
            "step": {"increase_exposed_population": 1},
        }
    },
)
def exposed_population():
    return _integ_exposed_population()


_integ_exposed_population = Integ(
    lambda: increase_exposed_population(),
    lambda: initial_exposed_population(),
    "_integ_exposed_population",
)


@component.add(
    name="VEHICLES GROWTH",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_increase_in_electric_vehicles": 3,
        "time": 10,
        "alpha_vehicle": 3,
        "limit_1_vehicle_growth": 3,
        "beta_vehicle": 3,
        "gdp": 3,
        "limit_2_vehicle_growth": 3,
        "subsidies_electric_vehicles_objective": 1,
        "initial_year_increase_in_electric_vehicles": 3,
        "final_year_increase_in_electric_vehicles": 3,
    },
)
def vehicles_growth():
    """
    INTEGER(IF THEN ELSE(SWITCH SUBSIDIES ELECTRIC VEHICLES=0,(GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)),IF THEN ELSE(SWITCH SUBSIDIES ELECTRIC VEHICLES=1:AND:Time<INITIAL YEAR SUBSIDIES ELECTRIC VEHICLES:OR:Time>FINAL YEAR SUBSIDIES ELECTRIC VEHICLES,(GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)),IF THEN ELSE(SWITCH SUBSIDIES ELECTRIC VEHICLES=1:AND:(TYPE OF FUEL=ELECTRICITY):AND:Time>= INITIAL YEAR SUBSIDIES ELECTRIC VEHICLES:OR:Time<=FINAL YEAR SUBSIDIES ELECTRIC VEHICLES,((GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)))*(1+(SUBSIDIES ELECTRIC VEHICLES OBJECTIVE)/(FINAL YEAR SUBSIDIES ELECTRIC VEHICLES-INITIAL YEAR SUBSIDIES ELECTRIC VEHICLES)),IF THEN ELSE(TYPE OF FUEL=PETROL:OR:TYPE OF FUEL=DIESEL:AND:SWITCH SUBSIDIES ELECTRIC VEHICLES=1:AND:((GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)))<0, ((GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)))*(1+(SUBSIDIES ELECTRIC VEHICLES OBJECTIVE/2)/(FINAL YEAR SUBSIDIES ELECTRIC VEHICLES-INITIAL YEAR SUBSIDIES ELECTRIC VEHICLES)),((GDP*ALPHA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]*(1-LIMIT 1 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL]*(Time-2018))+(BETA VEHICLE[TYPE OF VEHICLE,TYPE OF FUEL]))*((LIMIT 2 VEHICLE GROWTH[TYPE OF VEHICLE,TYPE OF FUEL])^(Time-2018)))*(1-(SUBSIDIES ELECTRIC VEHICLES OBJECTIVE/2)/(FINAL YEAR SUBSIDIES ELECTRIC VEHICLES-INITIAL YEAR SUBSIDIES ELECTRIC VEHICLES)))))))
    """
    return integer(
        if_then_else(
            switch_increase_in_electric_vehicles() == 0,
            lambda: (
                gdp()
                * alpha_vehicle()
                * (1 - limit_1_vehicle_growth() * (time() - 2018))
                + beta_vehicle()
            )
            * limit_2_vehicle_growth() ** (time() - 2018),
            lambda: if_then_else(
                np.logical_and(
                    switch_increase_in_electric_vehicles() == 1,
                    np.logical_or(
                        time() < initial_year_increase_in_electric_vehicles(),
                        time() > final_year_increase_in_electric_vehicles(),
                    ),
                ),
                lambda: (
                    gdp()
                    * alpha_vehicle()
                    * (1 - limit_1_vehicle_growth() * (time() - 2018))
                    + beta_vehicle()
                )
                * limit_2_vehicle_growth() ** (time() - 2018),
                lambda: if_then_else(
                    np.logical_and(
                        switch_increase_in_electric_vehicles() == 1,
                        np.logical_or(
                            time() >= initial_year_increase_in_electric_vehicles(),
                            time() <= final_year_increase_in_electric_vehicles(),
                        ),
                    ),
                    lambda: (
                        (
                            gdp()
                            * alpha_vehicle()
                            * (1 - limit_1_vehicle_growth() * (time() - 2018))
                            + beta_vehicle()
                        )
                        * limit_2_vehicle_growth() ** (time() - 2018)
                    )
                    * (
                        1
                        + subsidies_electric_vehicles_objective()
                        / (
                            final_year_increase_in_electric_vehicles()
                            - initial_year_increase_in_electric_vehicles()
                        )
                    ),
                    lambda: xr.DataArray(
                        0,
                        {
                            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
                        },
                        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
                    ),
                ),
            ),
        )
    )


@component.add(
    name="INCREASE RATIO IRRIGATED CROP NEW",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_increase_irrigated_crop": 3,
        "time": 2,
        "water_security": 1,
        "increase_irrigated_crop_objective": 1,
        "initial_ratio_irrigated_crop_area": 1,
        "initial_year_increase_irrigated_crop": 2,
        "final_year_increase_irrigated_crop": 2,
    },
)
def increase_ratio_irrigated_crop_new():
    return if_then_else(
        switch_increase_irrigated_crop() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_increase_irrigated_crop() == 1,
                np.logical_or(
                    time() < initial_year_increase_irrigated_crop(),
                    time() >= final_year_increase_irrigated_crop(),
                ),
            ),
            lambda: 0,
            lambda: if_then_else(
                np.logical_and(
                    switch_increase_irrigated_crop() == 1, water_security() == 0
                ),
                lambda: 0,
                lambda: (
                    increase_irrigated_crop_objective()
                    - initial_ratio_irrigated_crop_area()
                )
                / (
                    final_year_increase_irrigated_crop()
                    - initial_year_increase_irrigated_crop()
                ),
            ),
        ),
    )


@component.add(
    name="WATER DEMAND FROM AGRICULTURE",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_efficient_water_demand_from_agriculture": 3,
        "water_intensity_in_2018": 4,
        "irrigated_crop_area": 4,
        "cs_population": 4,
        "time": 3,
        "initial_year_efficient_water_demand_from_agriculture": 3,
        "efficient_water_demand_from_agriculture_objective": 2,
        "final_year_efficient_water_demand_from_agriculture": 2,
    },
)
def water_demand_from_agriculture():
    return if_then_else(
        switch_efficient_water_demand_from_agriculture() == 0,
        lambda: water_intensity_in_2018() * cs_population() * irrigated_crop_area(),
        lambda: if_then_else(
            np.logical_and(
                switch_efficient_water_demand_from_agriculture() == 1,
                time() < initial_year_efficient_water_demand_from_agriculture(),
            ),
            lambda: water_intensity_in_2018() * cs_population() * irrigated_crop_area(),
            lambda: if_then_else(
                np.logical_and(
                    switch_efficient_water_demand_from_agriculture() == 1,
                    time() > final_year_efficient_water_demand_from_agriculture(),
                ),
                lambda: (
                    water_intensity_in_2018() * cs_population() * irrigated_crop_area()
                )
                * (1 - efficient_water_demand_from_agriculture_objective()),
                lambda: (
                    water_intensity_in_2018() * cs_population() * irrigated_crop_area()
                )
                * (
                    1
                    - efficient_water_demand_from_agriculture_objective()
                    * (
                        (
                            time()
                            - initial_year_efficient_water_demand_from_agriculture()
                        )
                        / (
                            final_year_efficient_water_demand_from_agriculture()
                            - initial_year_efficient_water_demand_from_agriculture()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="WATER INTENSITY IN 2018",
    units="Hm3/capita*km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agriculture": 2,
        "initial_ratio_irrigated_crop_area": 2,
        "water_demand_from_agriculture_per_capita": 1,
    },
)
def water_intensity_in_2018():
    return if_then_else(
        agriculture() * initial_ratio_irrigated_crop_area() == 0,
        lambda: 0,
        lambda: water_demand_from_agriculture_per_capita()
        / (agriculture() * initial_ratio_irrigated_crop_area()),
    )


@component.add(
    name="VEHICLES FROM 15 TO 19 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles_from_15_to_19_years": 1},
    other_deps={
        "_integ_vehicles_from_15_to_19_years": {
            "initial": {
                "ratio_of_initial_vehicles_based_on_age": 1,
                "initial_vehicles_tot": 1,
            },
            "step": {
                "vehicle_growth_15_to_19_years": 1,
                "drop_out_vehicles_from_15_to_19_years": 1,
                "vehicle_growth_more_than_20_years": 1,
            },
        }
    },
)
def vehicles_from_15_to_19_years():
    return _integ_vehicles_from_15_to_19_years()


_integ_vehicles_from_15_to_19_years = Integ(
    lambda: vehicle_growth_15_to_19_years()
    - drop_out_vehicles_from_15_to_19_years()
    - vehicle_growth_more_than_20_years(),
    lambda: float(
        ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 15 TO 19 YEARS"]
    )
    * initial_vehicles_tot(),
    "_integ_vehicles_from_15_to_19_years",
)


@component.add(
    name="VEHICLES FROM 5 TO 9 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles_from_5_to_9_years": 1},
    other_deps={
        "_integ_vehicles_from_5_to_9_years": {
            "initial": {
                "ratio_of_initial_vehicles_based_on_age": 1,
                "initial_vehicles_tot": 1,
            },
            "step": {
                "vehicle_growth_5_to_9_years": 1,
                "drop_out_vehicles_from_5_to_9_years": 1,
                "vehicle_growth_10_to_14_years": 1,
            },
        }
    },
)
def vehicles_from_5_to_9_years():
    return _integ_vehicles_from_5_to_9_years()


_integ_vehicles_from_5_to_9_years = Integ(
    lambda: vehicle_growth_5_to_9_years()
    - drop_out_vehicles_from_5_to_9_years()
    - vehicle_growth_10_to_14_years(),
    lambda: float(
        ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 5 TO 9 YEARS"]
    )
    * initial_vehicles_tot(),
    "_integ_vehicles_from_5_to_9_years",
)


@component.add(
    name="CHANGE INITIAL VEHICLE LESS THAN 4 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_vehicles_tot": 1,
        "ratio_of_initial_vehicles_based_on_age": 1,
    },
)
def change_initial_vehicle_less_than_4_years():
    return if_then_else(
        time() < 2028,
        lambda: initial_vehicles_tot()
        * float(
            ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "LESS THAN 4 YEARS"]
        )
        * 0.1,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
    )


@component.add(
    name="VEHICLES LESS THAN 4 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles_less_than_4_years": 1},
    other_deps={
        "_integ_vehicles_less_than_4_years": {
            "initial": {
                "ratio_of_initial_vehicles_based_on_age": 1,
                "initial_vehicles_tot": 1,
            },
            "step": {
                "vehicle_growth_less_than_4_years": 1,
                "drop_out_vehicles_less_than_4_years": 1,
                "vehicle_growth_5_to_9_years": 1,
            },
        }
    },
)
def vehicles_less_than_4_years():
    return _integ_vehicles_less_than_4_years()


_integ_vehicles_less_than_4_years = Integ(
    lambda: vehicle_growth_less_than_4_years()
    - drop_out_vehicles_less_than_4_years()
    - vehicle_growth_5_to_9_years(),
    lambda: float(
        ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "LESS THAN 4 YEARS"]
    )
    * initial_vehicles_tot(),
    "_integ_vehicles_less_than_4_years",
)


@component.add(
    name="VEHICLES MORE THAN 20 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles_more_than_20_years": 1},
    other_deps={
        "_integ_vehicles_more_than_20_years": {
            "initial": {
                "ratio_of_initial_vehicles_based_on_age": 1,
                "initial_vehicles_tot": 1,
            },
            "step": {
                "vehicle_growth_more_than_20_years": 1,
                "drop_out_vehicles_more_than_20_years": 1,
                "end_of_vehicle_life": 1,
            },
        }
    },
)
def vehicles_more_than_20_years():
    return _integ_vehicles_more_than_20_years()


_integ_vehicles_more_than_20_years = Integ(
    lambda: vehicle_growth_more_than_20_years()
    - drop_out_vehicles_more_than_20_years()
    - end_of_vehicle_life(),
    lambda: float(
        ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "MORE THAN 20 YEARS"]
    )
    * initial_vehicles_tot(),
    "_integ_vehicles_more_than_20_years",
)


@component.add(
    name="CHANGE INITIAL VEHICLE 5 TO 9 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_vehicles_tot": 1,
        "ratio_of_initial_vehicles_based_on_age": 1,
    },
)
def change_initial_vehicle_5_to_9_years():
    return if_then_else(
        time() < 2028,
        lambda: initial_vehicles_tot()
        * float(
            ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 5 TO 9 YEARS"]
        )
        * 0.1,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
    )


@component.add(
    name="VEHICLES FROM 10 TO 14 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles_from_10_to_14_years": 1},
    other_deps={
        "_integ_vehicles_from_10_to_14_years": {
            "initial": {
                "ratio_of_initial_vehicles_based_on_age": 1,
                "initial_vehicles_tot": 1,
            },
            "step": {
                "vehicle_growth_10_to_14_years": 1,
                "drop_out_vehicles_from_10_to_14_years": 1,
                "vehicle_growth_15_to_19_years": 1,
            },
        }
    },
)
def vehicles_from_10_to_14_years():
    return _integ_vehicles_from_10_to_14_years()


_integ_vehicles_from_10_to_14_years = Integ(
    lambda: vehicle_growth_10_to_14_years()
    - drop_out_vehicles_from_10_to_14_years()
    - vehicle_growth_15_to_19_years(),
    lambda: float(
        ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 10 TO 14 YEARS"]
    )
    * initial_vehicles_tot(),
    "_integ_vehicles_from_10_to_14_years",
)


@component.add(
    name="CHANGE INITIAL VEHICLE 15 TO 19 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_vehicles_tot": 1,
        "ratio_of_initial_vehicles_based_on_age": 1,
    },
)
def change_initial_vehicle_15_to_19_years():
    return if_then_else(
        time() < 2028,
        lambda: initial_vehicles_tot()
        * float(
            ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 15 TO 19 YEARS"]
        )
        * 0.1,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
    )


@component.add(
    name="CHANGE INITIAL VEHICLE MORE THAN 20 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_vehicles_tot": 1,
        "ratio_of_initial_vehicles_based_on_age": 1,
    },
)
def change_initial_vehicle_more_than_20_years():
    return if_then_else(
        time() < 2028,
        lambda: initial_vehicles_tot()
        * float(
            ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "MORE THAN 20 YEARS"]
        )
        * 0.1,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
    )


@component.add(
    name="CHANGE INITIAL VEHICLE 10 TO 14 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_vehicles_tot": 1,
        "ratio_of_initial_vehicles_based_on_age": 1,
    },
)
def change_initial_vehicle_10_to_14_years():
    return if_then_else(
        time() < 2028,
        lambda: initial_vehicles_tot()
        * float(
            ratio_of_initial_vehicles_based_on_age().loc["TRUCK", "FROM 10 TO 14 YEARS"]
        )
        * 0.1,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
    )


@component.add(
    name="VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vehicles": 1},
    other_deps={
        "_integ_vehicles": {
            "initial": {"initial_vehicles_tot": 1},
            "step": {
                "vehicles_growth": 1,
                "vehicles_dropout": 1,
                "value_for_subscript_electric": 1,
            },
        }
    },
)
def vehicles():
    return _integ_vehicles()


_integ_vehicles = Integ(
    lambda: vehicles_growth() - vehicles_dropout() + value_for_subscript_electric(),
    lambda: initial_vehicles_tot(),
    "_integ_vehicles",
)


@component.add(
    name="TOTAL DROPOUT VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "end_of_vehicle_life": 1,
        "drop_out_vehicles_less_than_4_years": 1,
        "drop_out_vehicles_from_5_to_9_years": 1,
        "drop_out_vehicles_from_10_to_14_years": 1,
        "drop_out_vehicles_from_15_to_19_years": 1,
        "drop_out_vehicles_more_than_20_years": 1,
    },
)
def total_dropout_vehicles():
    return (
        end_of_vehicle_life()
        + drop_out_vehicles_less_than_4_years()
        + drop_out_vehicles_from_5_to_9_years()
        + drop_out_vehicles_from_10_to_14_years()
        + drop_out_vehicles_from_15_to_19_years()
        + drop_out_vehicles_more_than_20_years()
    )


@component.add(
    name="VEHICLES DELAY",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="Delay",
    depends_on={"_delay_vehicles_delay": 1},
    other_deps={
        "_delay_vehicles_delay": {
            "initial": {"vehicles_corrected": 1},
            "step": {"vehicles_corrected": 1},
        }
    },
)
def vehicles_delay():
    return _delay_vehicles_delay()


_delay_vehicles_delay = Delay(
    lambda: vehicles_corrected(),
    lambda: xr.DataArray(
        1,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    lambda: vehicles_corrected(),
    lambda: 1,
    time_step,
    "_delay_vehicles_delay",
)


@component.add(
    name="VEHICLES DIFFERENCE",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vehicles_corrected": 1, "vehicles_delay": 1},
)
def vehicles_difference():
    return vehicles_corrected() - vehicles_delay()


@component.add(
    name="VEHICLES DROPOUT",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"real_dropout": 1, "dropout_distributed_from_electric_vehicles": 1},
)
def vehicles_dropout():
    """
    IF THEN ELSE((Time-INITIAL TIME)<5,TOTAL VEHICLES GROWTH*0.02,IF THEN ELSE((Time-INITIAL TIME)<10,DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02 , IF THEN ELSE ( (Time -INITIAL TIME)<15,DELAY 10 YEARS AGE*0.2+DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02,IF THEN ELSE ((Time-INITIAL TIME)<20,DELAY 15 YEARS AGE*0.4+DELAY 10 YEARS AGE*0.2+DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02,DELAY 20 YEARS AGE*0.8+DELAY 15 YEARS AGE*0.4+DELAY 10 YEARS AGE*0.2+DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02))))+INITIAL VEHICLES DROPOUT
    """
    return real_dropout() + dropout_distributed_from_electric_vehicles()


@component.add(
    name="RENOVATED VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_dropout_vehicles": 1},
)
def renovated_vehicles():
    return integer(total_dropout_vehicles() * 0.8)


@component.add(
    name="VEHICLES EMISSIONS",
    units="tCO2",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "vehicles_corrected": 1,
        "vehicles_emission_coefficient": 1,
        "distance_covered": 1,
        "vehicle_age_coefficient": 1,
    },
)
def vehicles_emissions():
    return (
        vehicles_corrected()
        * vehicles_emission_coefficient()
        * distance_covered()
        * vehicle_age_coefficient()
    )


@component.add(
    name="VEHICLE AGE COEFFICIENT",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "vehicles_corrected": 2,
        "vehicles_from_10_to_14_years": 1,
        "vehicles_less_than_4_years": 1,
        "vehicles_from_15_to_19_years": 1,
        "vehicles_from_5_to_9_years": 1,
        "vehicles_more_than_20_years": 1,
    },
)
def vehicle_age_coefficient():
    """
    IF THEN ELSE((Time-INITIAL TIME)<5,(TOTAL VEHICLES GROWTH-(TOTAL VEHICLES GROWTH*0.02)),IF THEN ELSE((Time-INITIAL TIME )<10,(1+0.15)*(TOTAL VEHICLES GROWTH-(DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02)) , IF THEN ELSE ( (Time -INITIAL TIME)<15,(1+0.4)*(TOTAL VEHICLES GROWTH-(DELAY 10 YEARS AGE*0.2+DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02 )),IF THEN ELSE ((Time-INITIAL TIME)<20,(1+0.8)*(TOTAL VEHICLES GROWTH-(DELAY 15 YEARS AGE*0.4+DELAY 10 YEARS AGE*0.2+DELAY 5 YEAR AGE *0.08+TOTAL VEHICLES GROWTH*0.02)),2*(TOTAL VEHICLES GROWTH-(DELAY 20 YEARS AGE*0.8+DELAY 15 YEARS AGE*0.4+DELAY 10 YEARS AGE *0.2+DELAY 5 YEAR AGE*0.08+TOTAL VEHICLES GROWTH*0.02))))))/VEHICLES
    """
    return if_then_else(
        vehicles_corrected() <= 0,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
        lambda: (
            vehicles_less_than_4_years()
            + 1.2 * vehicles_from_5_to_9_years()
            + 1.4 * vehicles_from_10_to_14_years()
            + 1.8 * vehicles_from_15_to_19_years()
            + 2 * vehicles_more_than_20_years()
        )
        / vehicles_corrected(),
    )


@component.add(
    name="DROPOUT DISTRIBUTED FROM ELECTRIC VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vehicles_difference": 16},
)
def dropout_distributed_from_electric_vehicles():
    value = xr.DataArray(
        np.nan,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    )
    value.loc[["TRUCK"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
                lambda: xr.DataArray(
                    integer(
                        float(vehicles_difference().loc["TRUCK", "ELECTRICITY"]) * 0.99
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
                lambda: xr.DataArray(
                    float(vehicles_difference().loc["TRUCK", "ELECTRICITY"])
                    - integer(
                        float(vehicles_difference().loc["TRUCK", "ELECTRICITY"]) * 0.01
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["TRUCK"]}, 0)
        .values
    )
    value.loc[["VAN"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: float(vehicles_difference().loc["VAN", "ELECTRICITY"]) * 0.5
            + if_then_else(
                modulo(float(vehicles_difference().loc["VAN", "ELECTRICITY"]), 2) != 0,
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    )
                    == 2,
                    lambda: xr.DataArray(
                        0.5,
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    ),
                    lambda: xr.DataArray(
                        -0.5,
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    ),
                ),
                lambda: xr.DataArray(
                    0,
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["VAN"]}, 0)
        .values
    )
    value.loc[["BUS"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
                lambda: xr.DataArray(
                    integer(
                        float(vehicles_difference().loc["BUS", "ELECTRICITY"]) * 0.99
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
                lambda: xr.DataArray(
                    float(vehicles_difference().loc["BUS", "ELECTRICITY"])
                    - integer(
                        float(vehicles_difference().loc["BUS", "ELECTRICITY"]) * 0.01
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["BUS"]}, 0)
        .values
    )
    value.loc[["CAR"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: float(vehicles_difference().loc["CAR", "ELECTRICITY"]) * 0.5
            + if_then_else(
                modulo(float(vehicles_difference().loc["CAR", "ELECTRICITY"]), 2) != 0,
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    )
                    == 2,
                    lambda: xr.DataArray(
                        0.5,
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    ),
                    lambda: xr.DataArray(
                        -0.5,
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    ),
                ),
                lambda: xr.DataArray(
                    0,
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["CAR"]}, 0)
        .values
    )
    value.loc[["MOTORCYCLE"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
                lambda: xr.DataArray(
                    integer(
                        float(vehicles_difference().loc["MOTORCYCLE", "ELECTRICITY"])
                        * 0.01
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
                lambda: xr.DataArray(
                    float(vehicles_difference().loc["MOTORCYCLE", "ELECTRICITY"])
                    - integer(
                        float(vehicles_difference().loc["MOTORCYCLE", "ELECTRICITY"])
                        * 0.01
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["MOTORCYCLE"]}, 0)
        .values
    )
    value.loc[["TRACTOR"], :] = (
        if_then_else(
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 1,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 2,
                lambda: xr.DataArray(
                    integer(
                        float(vehicles_difference().loc["TRACTOR", "ELECTRICITY"])
                        * 0.99
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
                lambda: xr.DataArray(
                    float(vehicles_difference().loc["TRACTOR", "ELECTRICITY"])
                    - integer(
                        float(vehicles_difference().loc["TRACTOR", "ELECTRICITY"])
                        * 0.01
                    ),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                ),
            ),
            lambda: xr.DataArray(
                0, {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, ["TYPE OF FUEL"]
            ),
        )
        .expand_dims({"TYPE OF VEHICLE": ["TRACTOR"]}, 0)
        .values
    )
    return value


@component.add(
    name="CHANGE TO ELECTRIC VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_dropout_vehicles": 1},
)
def change_to_electric_vehicles():
    return sum(
        integer(
            total_dropout_vehicles().rename({"TYPE OF FUEL": "TYPE OF FUEL!"}) * 0.01
        ),
        dim=["TYPE OF FUEL!"],
    )


@component.add(
    name="VEHICLES CORRECTED",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vehicles": 2},
)
def vehicles_corrected():
    return if_then_else(
        vehicles() < 0,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
        lambda: vehicles(),
    )


@component.add(
    name="REAL DROPOUT",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_dropout_vehicles": 1, "renovated_vehicles": 1},
)
def real_dropout():
    return total_dropout_vehicles() - renovated_vehicles()


@component.add(
    name="VALUE FOR SUBSCRIPT ELECTRIC",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"change_to_electric_vehicles": 1},
)
def value_for_subscript_electric():
    return if_then_else(
        np.logical_and(
            (
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                    {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                    ["TYPE OF FUEL"],
                )
                == 3
            ),
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["TYPE OF VEHICLE"]) + 1),
                    {"TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"]},
                    ["TYPE OF VEHICLE"],
                )
                == 4,
                np.logical_or(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF VEHICLE"]) + 1),
                        {"TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"]},
                        ["TYPE OF VEHICLE"],
                    )
                    == 2,
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF VEHICLE"]) + 1),
                        {"TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"]},
                        ["TYPE OF VEHICLE"],
                    )
                    == 5,
                ),
            ),
        ),
        lambda: change_to_electric_vehicles().expand_dims(
            {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]}, 0
        ),
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            },
            ["TYPE OF FUEL", "TYPE OF VEHICLE"],
        ),
    ).transpose("TYPE OF VEHICLE", "TYPE OF FUEL")


@component.add(
    name="TOTAL EMISSIONS FROM TRANSPORT",
    units="tCO2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vehicles_emissions": 1},
)
def total_emissions_from_transport():
    return sum(
        vehicles_emissions().rename(
            {"TYPE OF VEHICLE": "TYPE OF VEHICLE!", "TYPE OF FUEL": "TYPE OF FUEL!"}
        ),
        dim=["TYPE OF VEHICLE!", "TYPE OF FUEL!"],
    )


@component.add(
    name="VEHICLE GROWTH MORE THAN 20 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "delay_vehicle_growth_15_to_19_years": 1,
        "change_initial_vehicle_15_to_19_years": 1,
        "ratio_drop_out_from_15_to_19_years": 1,
    },
)
def vehicle_growth_more_than_20_years():
    return integer(
        (
            delay_vehicle_growth_15_to_19_years()
            + change_initial_vehicle_15_to_19_years()
        )
        * (1 - ratio_drop_out_from_15_to_19_years())
    )


@component.add(
    name="DELAY VEHICLE GROWTH 15 TO 19 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_vehicle_growth_15_to_19_years": 1},
    other_deps={
        "_delayfixed_delay_vehicle_growth_15_to_19_years": {
            "initial": {},
            "step": {"vehicle_growth_15_to_19_years": 1},
        }
    },
)
def delay_vehicle_growth_15_to_19_years():
    return _delayfixed_delay_vehicle_growth_15_to_19_years()


_delayfixed_delay_vehicle_growth_15_to_19_years = DelayFixed(
    lambda: vehicle_growth_15_to_19_years(),
    lambda: 5,
    lambda: xr.DataArray(
        0,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    time_step,
    "_delayfixed_delay_vehicle_growth_15_to_19_years",
)


@component.add(
    name="DROP OUT VEHICLES FROM 10 TO 14 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "ratio_drop_out_from_10_to_14_years": 1,
        "vehicle_growth_10_to_14_years": 1,
        "change_initial_vehicle_10_to_14_years": 1,
    },
)
def drop_out_vehicles_from_10_to_14_years():
    return integer(
        ratio_drop_out_from_10_to_14_years()
        * (change_initial_vehicle_10_to_14_years() + vehicle_growth_10_to_14_years())
    )


@component.add(
    name="DROP OUT VEHICLES FROM 15 TO 19 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "ratio_drop_out_from_15_to_19_years": 1,
        "change_initial_vehicle_15_to_19_years": 1,
        "vehicle_growth_15_to_19_years": 1,
    },
)
def drop_out_vehicles_from_15_to_19_years():
    return integer(
        ratio_drop_out_from_15_to_19_years()
        * (change_initial_vehicle_15_to_19_years() + vehicle_growth_15_to_19_years())
    )


@component.add(
    name="DROP OUT VEHICLES FROM 5 TO 9 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "ratio_drop_out_from_5_to_9_years": 1,
        "vehicle_growth_5_to_9_years": 1,
        "change_initial_vehicle_5_to_9_years": 1,
    },
)
def drop_out_vehicles_from_5_to_9_years():
    return integer(
        ratio_drop_out_from_5_to_9_years()
        * (change_initial_vehicle_5_to_9_years() + vehicle_growth_5_to_9_years())
    )


@component.add(
    name="DROP OUT VEHICLES LESS THAN 4 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "ratio_drop_out_vehicles_less_than_4_years": 1,
        "change_initial_vehicle_less_than_4_years": 1,
        "vehicle_growth_less_than_4_years": 1,
    },
)
def drop_out_vehicles_less_than_4_years():
    return integer(
        ratio_drop_out_vehicles_less_than_4_years()
        * (
            change_initial_vehicle_less_than_4_years()
            + vehicle_growth_less_than_4_years()
        )
    )


@component.add(
    name="DROP OUT VEHICLES MORE THAN 20 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "ratio_drop_out_more_than_20_years": 1,
        "change_initial_vehicle_more_than_20_years": 1,
        "vehicle_growth_more_than_20_years": 1,
    },
)
def drop_out_vehicles_more_than_20_years():
    return integer(
        ratio_drop_out_more_than_20_years()
        * (
            change_initial_vehicle_more_than_20_years()
            + vehicle_growth_more_than_20_years()
        )
    )


@component.add(
    name="VEHICLE GROWTH 15 TO 19 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "delay_vehicle_growth_10_to_14_years": 1,
        "change_initial_vehicle_10_to_14_years": 1,
        "ratio_drop_out_from_10_to_14_years": 1,
    },
)
def vehicle_growth_15_to_19_years():
    return integer(
        (
            delay_vehicle_growth_10_to_14_years()
            + change_initial_vehicle_10_to_14_years()
        )
        * (1 - ratio_drop_out_from_10_to_14_years())
    )


@component.add(
    name="VEHICLE GROWTH 5 TO 9 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "delay_vehicle_growth_less_than_4_years": 1,
        "change_initial_vehicle_less_than_4_years": 1,
        "ratio_drop_out_vehicles_less_than_4_years": 1,
    },
)
def vehicle_growth_5_to_9_years():
    return integer(
        (
            delay_vehicle_growth_less_than_4_years()
            + change_initial_vehicle_less_than_4_years()
        )
        * (1 - ratio_drop_out_vehicles_less_than_4_years())
    )


@component.add(
    name="VEHICLE GROWTH LESS THAN 4 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vehicles_growth": 2},
)
def vehicle_growth_less_than_4_years():
    return if_then_else(
        vehicles_growth() < 0,
        lambda: xr.DataArray(
            0,
            {
                "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
            },
            ["TYPE OF VEHICLE", "TYPE OF FUEL"],
        ),
        lambda: integer(vehicles_growth()),
    )


@component.add(
    name="DELAY VEHICLE GROWTH 10 TO 14 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_vehicle_growth_10_to_14_years": 1},
    other_deps={
        "_delayfixed_delay_vehicle_growth_10_to_14_years": {
            "initial": {},
            "step": {"vehicle_growth_10_to_14_years": 1},
        }
    },
)
def delay_vehicle_growth_10_to_14_years():
    return _delayfixed_delay_vehicle_growth_10_to_14_years()


_delayfixed_delay_vehicle_growth_10_to_14_years = DelayFixed(
    lambda: vehicle_growth_10_to_14_years(),
    lambda: 5,
    lambda: xr.DataArray(
        0,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    time_step,
    "_delayfixed_delay_vehicle_growth_10_to_14_years",
)


@component.add(
    name="END OF VEHICLE LIFE",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "delay_vehicle_growth_more_than_20_years": 1,
        "change_initial_vehicle_more_than_20_years": 1,
        "ratio_drop_out_more_than_20_years": 1,
    },
)
def end_of_vehicle_life():
    return integer(
        (
            delay_vehicle_growth_more_than_20_years()
            + change_initial_vehicle_more_than_20_years()
        )
        * (1 - ratio_drop_out_more_than_20_years())
    )


@component.add(
    name="DELAY VEHICLE GROWTH 5 TO 9 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_vehicle_growth_5_to_9_years": 1},
    other_deps={
        "_delayfixed_delay_vehicle_growth_5_to_9_years": {
            "initial": {},
            "step": {"vehicle_growth_5_to_9_years": 1},
        }
    },
)
def delay_vehicle_growth_5_to_9_years():
    return _delayfixed_delay_vehicle_growth_5_to_9_years()


_delayfixed_delay_vehicle_growth_5_to_9_years = DelayFixed(
    lambda: vehicle_growth_5_to_9_years(),
    lambda: 5,
    lambda: xr.DataArray(
        0,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    time_step,
    "_delayfixed_delay_vehicle_growth_5_to_9_years",
)


@component.add(
    name="DELAY VEHICLE GROWTH LESS THAN 4 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_vehicle_growth_less_than_4_years": 1},
    other_deps={
        "_delayfixed_delay_vehicle_growth_less_than_4_years": {
            "initial": {},
            "step": {"vehicle_growth_less_than_4_years": 1},
        }
    },
)
def delay_vehicle_growth_less_than_4_years():
    return _delayfixed_delay_vehicle_growth_less_than_4_years()


_delayfixed_delay_vehicle_growth_less_than_4_years = DelayFixed(
    lambda: vehicle_growth_less_than_4_years(),
    lambda: 5,
    lambda: xr.DataArray(
        0,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    time_step,
    "_delayfixed_delay_vehicle_growth_less_than_4_years",
)


@component.add(
    name="DELAY VEHICLE GROWTH MORE THAN 20 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_vehicle_growth_more_than_20_years": 1},
    other_deps={
        "_delayfixed_delay_vehicle_growth_more_than_20_years": {
            "initial": {},
            "step": {"vehicle_growth_more_than_20_years": 1},
        }
    },
)
def delay_vehicle_growth_more_than_20_years():
    return _delayfixed_delay_vehicle_growth_more_than_20_years()


_delayfixed_delay_vehicle_growth_more_than_20_years = DelayFixed(
    lambda: vehicle_growth_more_than_20_years(),
    lambda: 5,
    lambda: xr.DataArray(
        0,
        {
            "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
            "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
        },
        ["TYPE OF VEHICLE", "TYPE OF FUEL"],
    ),
    time_step,
    "_delayfixed_delay_vehicle_growth_more_than_20_years",
)


@component.add(
    name="VEHICLE GROWTH 10 TO 14 YEARS",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "delay_vehicle_growth_5_to_9_years": 1,
        "change_initial_vehicle_5_to_9_years": 1,
        "ratio_drop_out_from_5_to_9_years": 1,
    },
)
def vehicle_growth_10_to_14_years():
    return integer(
        (delay_vehicle_growth_5_to_9_years() + change_initial_vehicle_5_to_9_years())
        * (1 - ratio_drop_out_from_5_to_9_years())
    )


@component.add(
    name="INITIAL VEHICLES TOT",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_cs": 1,
        "ratio_electric_truck_over_van": 2,
        "initial_vehicles": 4,
    },
)
def initial_vehicles_tot():
    return if_then_else(
        switch_cs() == 4,
        lambda: if_then_else(
            np.logical_and(
                (
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF VEHICLE"]) + 1),
                        {"TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"]},
                        ["TYPE OF VEHICLE"],
                    )
                    == 1
                ),
                (
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                        {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                        ["TYPE OF FUEL"],
                    )
                    == 3
                ),
            ),
            lambda: xr.DataArray(
                float(initial_vehicles().loc["TRUCK", "ELECTRICITY"])
                * ratio_electric_truck_over_van(),
                {
                    "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                    "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
                },
                ["TYPE OF VEHICLE", "TYPE OF FUEL"],
            ),
            lambda: if_then_else(
                np.logical_and(
                    (
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["TYPE OF VEHICLE"]) + 1),
                            {"TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"]},
                            ["TYPE OF VEHICLE"],
                        )
                        == 2
                    ),
                    (
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
                            {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
                            ["TYPE OF FUEL"],
                        )
                        == 3
                    ),
                ),
                lambda: xr.DataArray(
                    float(initial_vehicles().loc["VAN", "ELECTRICITY"])
                    * (1 - ratio_electric_truck_over_van()),
                    {
                        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
                        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
                    },
                    ["TYPE OF VEHICLE", "TYPE OF FUEL"],
                ),
                lambda: initial_vehicles(),
            ),
        ),
        lambda: initial_vehicles(),
    )


@component.add(
    name="INITIAL VEHICLES",
    units="vehicle",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_vehicles"},
)
def initial_vehicles():
    return _ext_constant_initial_vehicles()


_ext_constant_initial_vehicles = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_initial_vehicles",
)

_ext_constant_initial_vehicles.add(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_initial_vehicles.add(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_initial_vehicles.add(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_initial_vehicles.add(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_initial_vehicles.add(
    r"Historical.xlsx",
    "Transport system",
    "INITIAL_VEHICLES_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="LOSSES WITH H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"production_with_h2": 1, "losses_factor": 1},
)
def losses_with_h2():
    return production_with_h2() * losses_factor()


@component.add(
    name="IMPORT WITH H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"balance_with_h2": 2},
)
def import_with_h2():
    return if_then_else(balance_with_h2() > 0, lambda: balance_with_h2(), lambda: 0)


@component.add(
    name="ENERGY SUPPLY CONSIDERING H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"production_with_h2": 1, "losses_with_h2": 1},
)
def energy_supply_considering_h2():
    return production_with_h2() - losses_with_h2()


@component.add(
    name="TOTAL PRODUCTION CONSIDERING H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "solar_production_after_h2": 1,
        "hydro_production_after_h2": 1,
        "biomass_production": 1,
        "non_renewable_production": 1,
        "rooftop_production": 1,
        "wind_production_after_h2": 1,
        "h2_production": 1,
    },
)
def total_production_considering_h2():
    return (
        solar_production_after_h2()
        + hydro_production_after_h2()
        + biomass_production()
        + non_renewable_production()
        + rooftop_production()
        + wind_production_after_h2()
        + h2_production()
    )


@component.add(
    name="BALANCE WITH H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption": 1, "energy_supply_considering_h2": 1},
)
def balance_with_h2():
    return energy_consumption() - energy_supply_considering_h2()


@component.add(
    name="EXPORT WITH H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"balance_with_h2": 2},
)
def export_with_h2():
    return if_then_else(
        balance_with_h2() < 0, lambda: -1 * balance_with_h2(), lambda: 0
    )


@component.add(
    name="PRODUCTION WITH H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_production_considering_h2": 1},
)
def production_with_h2():
    return total_production_considering_h2()


@component.add(
    name="H2 PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"request_from_energy_demand_with_policy": 1},
)
def h2_production():
    return request_from_energy_demand_with_policy()


@component.add(
    name="TOTAL PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "biomass_production": 1,
        "hydro_production": 1,
        "non_renewable_production": 1,
        "rooftop_production": 1,
        "solar_production": 1,
        "wind_production": 1,
    },
)
def total_production():
    return (
        biomass_production()
        + hydro_production()
        + non_renewable_production()
        + rooftop_production()
        + solar_production()
        + wind_production()
    )


@component.add(
    name="ENERGY DISTRIBUTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption": 1, "import_1": 1, "renewables_ratio": 1},
)
def energy_distribution():
    return (energy_consumption() - import_1()) * renewables_ratio()


@component.add(
    name="GREEN URBAN",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_green_urban": 1},
    other_deps={
        "_integ_green_urban": {
            "initial": {"initial_green_urban_area": 1},
            "step": {
                "switch_green_urban_planning": 2,
                "urbanization": 3,
                "ratio_green_area_to_water": 3,
                "urban": 3,
                "urban_expansion_from_wetlands": 3,
                "urban_expansion": 3,
                "urban_to_water": 3,
                "ratio_wetlands_to_green_area": 3,
                "ratio_agriculture_to_green_area": 3,
                "ratio_other_to_green_area": 3,
                "green_urban_planning_objective": 1,
                "initial_year_green_urban_planning": 2,
                "time": 2,
                "final_year_green_urban_planning": 2,
            },
        }
    },
)
def green_urban():
    return _integ_green_urban()


_integ_green_urban = Integ(
    lambda: if_then_else(
        switch_green_urban_planning() == 0,
        lambda: float(
            np.minimum(
                urban_expansion() * ratio_other_to_green_area()
                + urban_expansion_from_wetlands() * ratio_wetlands_to_green_area()
                + urbanization() * ratio_agriculture_to_green_area()
                - urban_to_water() * ratio_green_area_to_water(),
                urban() * 0.9,
            )
        ),
        lambda: if_then_else(
            np.logical_and(
                switch_green_urban_planning() == 1,
                np.logical_or(
                    time() < initial_year_green_urban_planning(),
                    time() > final_year_green_urban_planning(),
                ),
            ),
            lambda: float(
                np.minimum(
                    urban_expansion() * ratio_other_to_green_area()
                    + urban_expansion_from_wetlands() * ratio_wetlands_to_green_area()
                    + urbanization() * ratio_agriculture_to_green_area()
                    - urban_to_water() * ratio_green_area_to_water(),
                    urban() * 0.9,
                )
            ),
            lambda: float(
                np.minimum(
                    (
                        urban_expansion() * ratio_other_to_green_area()
                        + urban_expansion_from_wetlands()
                        * ratio_wetlands_to_green_area()
                        + urbanization() * ratio_agriculture_to_green_area()
                        - urban_to_water() * ratio_green_area_to_water()
                    )
                    * green_urban_planning_objective()
                    * (
                        1
                        / (
                            final_year_green_urban_planning()
                            - initial_year_green_urban_planning()
                        )
                    ),
                    urban() * 0.9,
                )
            ),
        ),
    ),
    lambda: initial_green_urban_area(),
    "_integ_green_urban",
)


@component.add(
    name="RENEWABLES RATIO",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption_share": 1, "energy_consumption": 1},
)
def renewables_ratio():
    return float(energy_consumption_share().loc["RENEWABLES"]) / energy_consumption()


@component.add(
    name="GDP DOLLARS",
    units="Dollars",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gdp_per_capita": 1, "euro_to_dollar": 1},
)
def gdp_dollars():
    return gdp_per_capita() / euro_to_dollar()


@component.add(
    name="URBAN EXPANSION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "other": 1,
        "initial_other_area": 2,
        "conversion_speed": 1,
        "expansion_ratio": 1,
        "urban": 1,
    },
)
def urban_expansion():
    return if_then_else(
        other() > initial_other_area() * 0.5,
        lambda: float(
            np.minimum(
                (urban() * (expansion_ratio() * conversion_speed()) / 100) * 0.6,
                (initial_other_area() * 0.5) / 1,
            )
        ),
        lambda: 0,
    )


@component.add(
    name="URBAN TO WATER",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"select_location_water_storage": 1, "water_area": 1},
)
def urban_to_water():
    return if_then_else(
        select_location_water_storage() == 2, lambda: water_area(), lambda: 0
    )


@component.add(
    name="URBANIZATION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"accesibility_restriction": 1, "agriculture": 1, "expansion_ratio": 1},
)
def urbanization():
    return accesibility_restriction() * agriculture() * expansion_ratio() * 0.3


@component.add(
    name="WETLANDS",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_wetlands": 1},
    other_deps={
        "_integ_wetlands": {
            "initial": {"initial_wetlands_area": 1},
            "step": {"urban_expansion_from_wetlands": 1, "wetlands_desertion": 1},
        }
    },
)
def wetlands():
    return _integ_wetlands()


_integ_wetlands = Integ(
    lambda: -urban_expansion_from_wetlands() - wetlands_desertion(),
    lambda: initial_wetlands_area(),
    "_integ_wetlands",
)


@component.add(
    name="BALANCE FOR H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption": 1, "energy_supply": 1},
)
def balance_for_h2():
    return energy_consumption() - energy_supply()


@component.add(
    name="TOTAL DAMAGES CURRENT VALUE",
    units="Euros",
    subscripts=["RETURN PERIOD"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_damages": 1, "inflation_rate_2018_to_2024": 1},
)
def total_damages_current_value():
    return total_damages() * (1 * (1 + inflation_rate_2018_to_2024()) ** 8)


@component.add(
    name="DAMAGES IN OTHER",
    units="Euros 2010",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"maximum_damage_other": 1, "damage_by_depth_in_other": 1},
)
def damages_in_other():
    return maximum_damage_other() * damage_by_depth_in_other()


@component.add(
    name="DAMAGES IN ROADS RAILWAYS",
    units="Euros 2010",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "maximum_damage_roads_railways": 1,
        "damage_by_depth_in_roads_railways": 1,
    },
)
def damages_in_roads_railways():
    return maximum_damage_roads_railways() * damage_by_depth_in_roads_railways()


@component.add(
    name="DAMAGES IN URBAN",
    units="Euros 2010",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"maximum_damage_urban": 1, "damage_by_depth_in_urban": 1},
)
def damages_in_urban():
    return maximum_damage_urban() * damage_by_depth_in_urban()


@component.add(
    name="FLOOD AREA ROADS RAILWAYS",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flood_area": 1},
)
def flood_area_roads_railways():
    return flood_area().loc[:, :, "RAIL RAILWAYS"].reset_coords(drop=True) * 1000000.0


@component.add(
    name="FLOOD AREA URBAN",
    units="m2",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flood_area": 1},
)
def flood_area_urban():
    return flood_area().loc[:, :, "URBAN LAND"].reset_coords(drop=True) * 1000000.0


@component.add(
    name="SWITCH POLICY FLOOD DEFENSES FOR URBAN AREAS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_policy_flood_defenses_for_urban_areas"
    },
)
def switch_policy_flood_defenses_for_urban_areas():
    """
    0 = Without flood defences 1 = With flood defences
    """
    return _ext_constant_switch_policy_flood_defenses_for_urban_areas()


_ext_constant_switch_policy_flood_defenses_for_urban_areas = ExtConstant(
    r"Policy.xlsx",
    "Floods",
    "SWITCH_POLICY_DEFENSES*",
    {},
    _root,
    {},
    "_ext_constant_switch_policy_flood_defenses_for_urban_areas",
)


@component.add(
    name="AGGREGATED DAMAGES IN OTHER",
    units="Euros 2010",
    subscripts=["RETURN PERIOD"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"damages_in_other": 1},
)
def aggregated_damages_in_other():
    return sum(
        damages_in_other().rename({"FLOOD DEPTH": "FLOOD DEPTH!"}), dim=["FLOOD DEPTH!"]
    )


@component.add(
    name="AGGREGATED DAMAGES IN ROADS RAILWAYS",
    units="Euros 2010",
    subscripts=["RETURN PERIOD"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"damages_in_roads_railways": 1},
)
def aggregated_damages_in_roads_railways():
    return sum(
        damages_in_roads_railways().rename({"FLOOD DEPTH": "FLOOD DEPTH!"}),
        dim=["FLOOD DEPTH!"],
    )


@component.add(
    name="AGGREGATED DAMAGES IN URBAN",
    units="Euros 2010",
    subscripts=["RETURN PERIOD"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"damages_in_urban": 1},
)
def aggregated_damages_in_urban():
    return sum(
        damages_in_urban().rename({"FLOOD DEPTH": "FLOOD DEPTH!"}), dim=["FLOOD DEPTH!"]
    )


@component.add(
    name="EURO TO DOLLAR", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def euro_to_dollar():
    return 0.86


@component.add(
    name="FLOOD AREA",
    units="km2",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH", "FLOODED LAND"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_flood_area"},
)
def flood_area():
    return _ext_constant_flood_area()


_ext_constant_flood_area = ExtConstant(
    r"Historical.xlsx",
    "Floods",
    "RP10_DATA",
    {
        "RETURN PERIOD": ["TEN"],
        "FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"],
        "FLOODED LAND": _subscript_dict["FLOODED LAND"],
    },
    _root,
    {
        "RETURN PERIOD": _subscript_dict["RETURN PERIOD"],
        "FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"],
        "FLOODED LAND": _subscript_dict["FLOODED LAND"],
    },
    "_ext_constant_flood_area",
)

_ext_constant_flood_area.add(
    r"Historical.xlsx",
    "Floods",
    "RP50_DATA",
    {
        "RETURN PERIOD": ["FIFTY"],
        "FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"],
        "FLOODED LAND": _subscript_dict["FLOODED LAND"],
    },
)

_ext_constant_flood_area.add(
    r"Historical.xlsx",
    "Floods",
    "RP100_DATA",
    {
        "RETURN PERIOD": ["HUNDRED"],
        "FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"],
        "FLOODED LAND": _subscript_dict["FLOODED LAND"],
    },
)


@component.add(
    name="FLOOD AREA OTHER",
    subscripts=["RETURN PERIOD", "FLOOD DEPTH"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"flood_area": 1},
)
def flood_area_other():
    return flood_area().loc[:, :, "OTHER LAND"].reset_coords(drop=True) * 1000000.0


@component.add(
    name="EXPECTED LOSSES BY FLOODS",
    units="Euros",
    subscripts=["RETURN PERIOD"],
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_expected_losses_by_floods": 1},
    other_deps={
        "_integ_expected_losses_by_floods": {
            "initial": {"total_damages_current_value": 1},
            "step": {"total_damages_current_value": 2, "inflation": 1},
        }
    },
)
def expected_losses_by_floods():
    return _integ_expected_losses_by_floods()


_integ_expected_losses_by_floods = Integ(
    lambda: total_damages_current_value() * (1 + inflation())
    - total_damages_current_value(),
    lambda: total_damages_current_value(),
    "_integ_expected_losses_by_floods",
)


@component.add(
    name="OTHER DAMAGE AT DEPTH",
    units="Dmnl",
    subscripts=["FLOOD DEPTH"],
    comp_type="Constant",
    comp_subtype="Normal",
)
def other_damage_at_depth():
    return xr.DataArray(
        [0.55, 0.75, 0.85, 0.95, 1.0],
        {"FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"]},
        ["FLOOD DEPTH"],
    )


@component.add(
    name="INFLATION RATE 2018 TO 2024",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_inflation_rate_2018_to_2024"},
)
def inflation_rate_2018_to_2024():
    return _ext_constant_inflation_rate_2018_to_2024()


_ext_constant_inflation_rate_2018_to_2024 = ExtConstant(
    r"Historical.xlsx",
    "Floods",
    "HISTORICAL_INFLATION_RATE",
    {},
    _root,
    {},
    "_ext_constant_inflation_rate_2018_to_2024",
)


@component.add(
    name="ROADS RAILWAYS DAMAGE AT DEPTH",
    units="Dmnl",
    subscripts=["FLOOD DEPTH"],
    comp_type="Constant",
    comp_subtype="Normal",
)
def roads_railways_damage_at_depth():
    return xr.DataArray(
        [0.42, 0.65, 0.8, 0.9, 1.0],
        {"FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"]},
        ["FLOOD DEPTH"],
    )


@component.add(
    name="MAXIMUM DAMAGE OTHER",
    units="Euros/m2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gdp_dollars": 1},
)
def maximum_damage_other():
    return 0.000585 * gdp_dollars() + 29.12


@component.add(
    name="MAXIMUM DAMAGE ROADS RAILWAYS",
    units="Euros/m2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gdp_dollars": 1},
)
def maximum_damage_roads_railways():
    return 0.000403 * gdp_dollars() + 26.2


@component.add(
    name="INFLATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_inflation"},
)
def inflation():
    return _ext_constant_inflation()


_ext_constant_inflation = ExtConstant(
    r"Historical.xlsx",
    "Floods",
    "INFLATION_RATE",
    {},
    _root,
    {},
    "_ext_constant_inflation",
)


@component.add(
    name="URBAN DAMAGE AT DEPTH CONSIDERING FLOOD DEFENSES",
    units="Dmnl",
    subscripts=["FLOOD DEPTH"],
    comp_type="Constant",
    comp_subtype="Normal",
)
def urban_damage_at_depth_considering_flood_defenses():
    return xr.DataArray(
        [0.08, 0.45, 0.68, 0.85, 0.95],
        {"FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"]},
        ["FLOOD DEPTH"],
    )


@component.add(
    name="TOTAL DAMAGES",
    units="Euros 2010",
    subscripts=["RETURN PERIOD"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "aggregated_damages_in_urban": 1,
        "aggregated_damages_in_other": 1,
        "aggregated_damages_in_roads_railways": 1,
    },
)
def total_damages():
    return (
        aggregated_damages_in_urban()
        + aggregated_damages_in_other()
        + aggregated_damages_in_roads_railways()
    )


@component.add(
    name="URBAN DAMAGE AT DEPTH",
    units="Dmnl",
    subscripts=["FLOOD DEPTH"],
    comp_type="Constant",
    comp_subtype="Normal",
)
def urban_damage_at_depth():
    return xr.DataArray(
        [0.4, 0.6, 0.75, 0.85, 0.95],
        {"FLOOD DEPTH": _subscript_dict["FLOOD DEPTH"]},
        ["FLOOD DEPTH"],
    )


@component.add(
    name="MAXIMUM DAMAGE URBAN",
    units="Euros/m2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gdp_dollars": 1},
)
def maximum_damage_urban():
    return 0.0034 * gdp_dollars() + 195.8


@component.add(
    name="RATIO DROP OUT FROM 10 TO 14 YEARS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_drop_out_from_10_to_14_years"},
)
def ratio_drop_out_from_10_to_14_years():
    """
    0.2
    """
    return _ext_constant_ratio_drop_out_from_10_to_14_years()


_ext_constant_ratio_drop_out_from_10_to_14_years = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "RATIO_DROP_OUT_VEHICLES_FROM_10_TO_14_YEARS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_drop_out_from_10_to_14_years",
)


@component.add(
    name="RATIO DROP OUT FROM 15 TO 19 YEARS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_drop_out_from_15_to_19_years"},
)
def ratio_drop_out_from_15_to_19_years():
    """
    0.4
    """
    return _ext_constant_ratio_drop_out_from_15_to_19_years()


_ext_constant_ratio_drop_out_from_15_to_19_years = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "RATIO_DROP_OUT_VEHICLES_FROM_15_TO_19_YEARS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_drop_out_from_15_to_19_years",
)


@component.add(
    name="RATIO DROP OUT FROM 5 TO 9 YEARS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_drop_out_from_5_to_9_years"},
)
def ratio_drop_out_from_5_to_9_years():
    """
    0.08
    """
    return _ext_constant_ratio_drop_out_from_5_to_9_years()


_ext_constant_ratio_drop_out_from_5_to_9_years = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "RATIO_DROP_OUT_VEHICLES_FROM_5_TO_9_YEARS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_drop_out_from_5_to_9_years",
)


@component.add(
    name="RATIO DROP OUT MORE THAN 20 YEARS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_drop_out_more_than_20_years"},
)
def ratio_drop_out_more_than_20_years():
    """
    0.8
    """
    return _ext_constant_ratio_drop_out_more_than_20_years()


_ext_constant_ratio_drop_out_more_than_20_years = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "RATIO_DROP_OUT_VEHICLES_MORE_THAN_20_YEARS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_drop_out_more_than_20_years",
)


@component.add(
    name="RATIO DROP OUT VEHICLES LESS THAN 4 YEARS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_ratio_drop_out_vehicles_less_than_4_years"
    },
)
def ratio_drop_out_vehicles_less_than_4_years():
    """
    0.02
    """
    return _ext_constant_ratio_drop_out_vehicles_less_than_4_years()


_ext_constant_ratio_drop_out_vehicles_less_than_4_years = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "RATIO_DROP_OUT_VEHICLES_LESS_THAN_4_YEARS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_drop_out_vehicles_less_than_4_years",
)


@component.add(
    name="LIMIT 2 VEHICLE GROWTH",
    units="Dmnl",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limit_2_vehicle_growth"},
)
def limit_2_vehicle_growth():
    return _ext_constant_limit_2_vehicle_growth()


_ext_constant_limit_2_vehicle_growth = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_limit_2_vehicle_growth",
)

_ext_constant_limit_2_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_2_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_2_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_2_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_limit_2_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_2_VEHICLE_GROWTH_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="RATIO OF INITIAL VEHICLES BASED ON AGE",
    units="Dmnl",
    subscripts=["TYPE OF VEHICLE", "VEHICLE AGE"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_of_initial_vehicles_based_on_age"},
)
def ratio_of_initial_vehicles_based_on_age():
    return _ext_constant_ratio_of_initial_vehicles_based_on_age()


_ext_constant_ratio_of_initial_vehicles_based_on_age = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "VEHICLE AGE": _subscript_dict["VEHICLE AGE"],
    },
    "_ext_constant_ratio_of_initial_vehicles_based_on_age",
)

_ext_constant_ratio_of_initial_vehicles_based_on_age.add(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
)

_ext_constant_ratio_of_initial_vehicles_based_on_age.add(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
)

_ext_constant_ratio_of_initial_vehicles_based_on_age.add(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
)

_ext_constant_ratio_of_initial_vehicles_based_on_age.add(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_MOTORCYCLE",
    {"TYPE OF VEHICLE": ["MOTORCYCLE"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
)

_ext_constant_ratio_of_initial_vehicles_based_on_age.add(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_OF_INITIAL_VEHICLES_BASED_ON_AGE_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "VEHICLE AGE": _subscript_dict["VEHICLE AGE"]},
)


@component.add(
    name="LIMIT 1 VEHICLE GROWTH",
    units="Dmnl",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limit_1_vehicle_growth"},
)
def limit_1_vehicle_growth():
    return _ext_constant_limit_1_vehicle_growth()


_ext_constant_limit_1_vehicle_growth = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_limit_1_vehicle_growth",
)

_ext_constant_limit_1_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_1_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_1_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_limit_1_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_limit_1_vehicle_growth.add(
    r"Historical.xlsx",
    "Transport system",
    "LIMIT_1_VEHICLE_GROWTH_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="WIND PRODUCTION AFTER H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"wind_production": 1, "wind_energy_request": 1},
)
def wind_production_after_h2():
    return wind_production() - wind_energy_request()


@component.add(
    name="LITERS TO HM3",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_liters_to_hm3"},
)
def liters_to_hm3():
    return _ext_constant_liters_to_hm3()


_ext_constant_liters_to_hm3 = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "LITERS_TO_HM3*",
    {},
    _root,
    {},
    "_ext_constant_liters_to_hm3",
)


@component.add(
    name="TRANSFORMATION COEFFICIENT ONE",
    units="kWh/kg",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_transformation_coefficient_one"},
)
def transformation_coefficient_one():
    return _ext_constant_transformation_coefficient_one()


_ext_constant_transformation_coefficient_one = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "TRANSFORMATION_COEFFICIENT_ONE*",
    {},
    _root,
    {},
    "_ext_constant_transformation_coefficient_one",
)


@component.add(
    name="TRANSFORMATION COEFFICIENT TWO",
    units="kg/kWh",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_transformation_coefficient_two"},
)
def transformation_coefficient_two():
    return _ext_constant_transformation_coefficient_two()


_ext_constant_transformation_coefficient_two = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "TRANSFORMATION_COEFFICIENT_TWO*",
    {},
    _root,
    {},
    "_ext_constant_transformation_coefficient_two",
)


@component.add(
    name="FINAL YEAR H2 POLICY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_h2_policy"},
)
def final_year_h2_policy():
    return _ext_constant_final_year_h2_policy()


_ext_constant_final_year_h2_policy = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "FINAL_YEAR_H2_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_h2_policy",
)


@component.add(
    name="HYDRO ENERGY REQUEST",
    units="kWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_requested_from_h2_production": 1,
        "energy_distribution_scenario_for_h2": 1,
    },
)
def hydro_energy_request():
    return energy_requested_from_h2_production() * float(
        energy_distribution_scenario_for_h2().loc["HYDRO"]
    )


@component.add(
    name="H2 OBJECTIVE IN RENEWABLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_h2_objective_in_renewables"},
)
def h2_objective_in_renewables():
    return _ext_constant_h2_objective_in_renewables()


_ext_constant_h2_objective_in_renewables = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "H2_OBJECTIVE_IN_RENEWABLES*",
    {},
    _root,
    {},
    "_ext_constant_h2_objective_in_renewables",
)


@component.add(
    name="HYDRO PRODUCTION AFTER H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"hydro_production": 1, "hydro_energy_request": 1},
)
def hydro_production_after_h2():
    return hydro_production() - hydro_energy_request()


@component.add(
    name="KWH TO MWH",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_kwh_to_mwh"},
)
def kwh_to_mwh():
    return _ext_constant_kwh_to_mwh()


_ext_constant_kwh_to_mwh = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "KWH_TO_MWH*",
    {},
    _root,
    {},
    "_ext_constant_kwh_to_mwh",
)


@component.add(
    name="SWITCH H2 POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_h2_policy"},
)
def switch_h2_policy():
    """
    0=OFF 1=ON
    """
    return _ext_constant_switch_h2_policy()


_ext_constant_switch_h2_policy = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "SWITCH_H2_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_switch_h2_policy",
)


@component.add(
    name="SOLAR PRODUCTION AFTER H2",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"solar_production": 1, "solar_energy_request": 1},
)
def solar_production_after_h2():
    return solar_production() - solar_energy_request()


@component.add(
    name="WATER REQUIREMENTS FROM H2 PRODUCTION",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "h2_production": 1,
        "kwh_to_mwh": 1,
        "transformation_coefficient_one": 1,
        "water_consumption": 1,
        "liters_to_hm3": 1,
    },
)
def water_requirements_from_h2_production():
    return (
        ((h2_production() / kwh_to_mwh()) / transformation_coefficient_one())
        * water_consumption()
    ) / liters_to_hm3()


@component.add(
    name="ENERGY DISTRIBUTION SCENARIO FOR H2",
    units="Dmnl",
    subscripts=["TYPE OF RENEWABLES"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_energy_distribution_scenario_for_h2"},
)
def energy_distribution_scenario_for_h2():
    return _ext_constant_energy_distribution_scenario_for_h2()


_ext_constant_energy_distribution_scenario_for_h2 = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "ENERGY_DISTRIBUTION_SCENARIO_FOR_H2*",
    {"TYPE OF RENEWABLES": _subscript_dict["TYPE OF RENEWABLES"]},
    _root,
    {"TYPE OF RENEWABLES": _subscript_dict["TYPE OF RENEWABLES"]},
    "_ext_constant_energy_distribution_scenario_for_h2",
)


@component.add(
    name="INITIAL ROLE OF H2 IN RENEWABLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_role_of_h2_in_renewables"},
)
def initial_role_of_h2_in_renewables():
    return _ext_constant_initial_role_of_h2_in_renewables()


_ext_constant_initial_role_of_h2_in_renewables = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "INITIAL_ROLE_OF_H2_IN_RENEWABLES*",
    {},
    _root,
    {},
    "_ext_constant_initial_role_of_h2_in_renewables",
)


@component.add(
    name="REQUEST FROM ENERGY DEMAND WITH POLICY",
    units="MWh",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_request_from_energy_demand_with_policy": 1},
    other_deps={
        "_integ_request_from_energy_demand_with_policy": {
            "initial": {},
            "step": {
                "switch_h2_policy": 2,
                "energy_distribution": 1,
                "time": 2,
                "final_year_h2_policy": 2,
                "initial_year_h2_policy": 2,
                "h2_objective_in_renewables": 1,
                "initial_role_of_h2_in_renewables": 1,
            },
        }
    },
)
def request_from_energy_demand_with_policy():
    return _integ_request_from_energy_demand_with_policy()


_integ_request_from_energy_demand_with_policy = Integ(
    lambda: if_then_else(
        switch_h2_policy() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_h2_policy() == 1,
                np.logical_or(
                    time() < initial_year_h2_policy(),
                    time() > final_year_h2_policy() - 1,
                ),
            ),
            lambda: 0,
            lambda: (
                (h2_objective_in_renewables() - initial_role_of_h2_in_renewables())
                / (final_year_h2_policy() - initial_year_h2_policy())
            )
            * energy_distribution(),
        ),
    ),
    lambda: 0,
    "_integ_request_from_energy_demand_with_policy",
)


@component.add(
    name="INITIAL YEAR H2 POLICY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_h2_policy"},
)
def initial_year_h2_policy():
    return _ext_constant_initial_year_h2_policy()


_ext_constant_initial_year_h2_policy = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "INITIAL_YEAR_H2_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_h2_policy",
)


@component.add(
    name="WATER DEMAND",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "water_demand_from_agriculture": 1,
        "water_demand_from_industry": 1,
        "water_demand_from_urban": 1,
        "water_used_for_snow_production": 1,
        "water_requirements_from_h2_production": 1,
    },
)
def water_demand():
    return (
        water_demand_from_agriculture()
        + water_demand_from_industry()
        + water_demand_from_urban()
        + water_used_for_snow_production()
        + water_requirements_from_h2_production()
    )


@component.add(
    name="WIND ENERGY REQUEST",
    units="kWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_requested_from_h2_production": 1,
        "energy_distribution_scenario_for_h2": 1,
    },
)
def wind_energy_request():
    return energy_requested_from_h2_production() * float(
        energy_distribution_scenario_for_h2().loc["WIND"]
    )


@component.add(
    name="SOLAR ENERGY REQUEST",
    units="kWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_requested_from_h2_production": 1,
        "energy_distribution_scenario_for_h2": 1,
    },
)
def solar_energy_request():
    return energy_requested_from_h2_production() * float(
        energy_distribution_scenario_for_h2().loc["SOLAR"]
    )


@component.add(
    name="WATER CONSUMPTION",
    units="l/kg",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_water_consumption"},
)
def water_consumption():
    return _ext_constant_water_consumption()


_ext_constant_water_consumption = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "WATER_CONSUMPTION*",
    {},
    _root,
    {},
    "_ext_constant_water_consumption",
)


@component.add(
    name="ENERGY REQUESTED FROM H2 PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "h2_production": 1,
        "kwh_to_mwh": 2,
        "transformation_coefficient_one": 1,
        "transformation_coefficient_two": 1,
    },
)
def energy_requested_from_h2_production():
    return (h2_production() / (transformation_coefficient_one() * kwh_to_mwh())) * (
        transformation_coefficient_two() * kwh_to_mwh()
    )


@component.add(
    name="RATIO OF GREEN URBAN AREA",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"green_urban": 1, "urban": 1},
)
def ratio_of_green_urban_area():
    return 1 - green_urban() / urban()


@component.add(
    name="RATIO ELECTRIC TRUCK OVER VAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_electric_truck_over_van"},
)
def ratio_electric_truck_over_van():
    return _ext_constant_ratio_electric_truck_over_van()


_ext_constant_ratio_electric_truck_over_van = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "RATIO_ELECTRIC_TRUCK_OVER_VAN*",
    {},
    _root,
    {},
    "_ext_constant_ratio_electric_truck_over_van",
)


@component.add(
    name="MEDIUM INCENTIVES ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_incentives_electric_vehicles():
    return 0.5


@component.add(
    name="INITIAL YEAR GREEN URBAN PLANNING",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_green_urban_planning"},
)
def initial_year_green_urban_planning():
    return _ext_constant_initial_year_green_urban_planning()


_ext_constant_initial_year_green_urban_planning = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INITIAL_YEAR_GREEN_URBAN_PLANNING*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_green_urban_planning",
)


@component.add(
    name="FINAL YEAR GREEN URBAN PLANNING",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_green_urban_planning"},
)
def final_year_green_urban_planning():
    return _ext_constant_final_year_green_urban_planning()


_ext_constant_final_year_green_urban_planning = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "FINAL_YEAR_GREEN_URBAN_PLANNING*",
    {},
    _root,
    {},
    "_ext_constant_final_year_green_urban_planning",
)


@component.add(
    name="ALPHA VEHICLE",
    units="Dmnl",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alpha_vehicle"},
)
def alpha_vehicle():
    return _ext_constant_alpha_vehicle()


_ext_constant_alpha_vehicle = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_alpha_vehicle",
)

_ext_constant_alpha_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_alpha_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_alpha_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_alpha_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_alpha_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "ALPHA_VEHICLE_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="SUBSIDIES ELECTRIC VEHICLES OBJECTIVE",
    units="Dmnl",
    subscripts=["TYPE OF FUEL"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_incentives_electric_vehicles": 8,
        "low_incentives_electric_vehicles": 4,
        "very_high_incentives_electric_vehicles": 2,
        "high_incentives_electric_vehicles": 2,
        "medium_incentives_electric_vehicles": 2,
    },
)
def subsidies_electric_vehicles_objective():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["TYPE OF FUEL"]) + 1),
            {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
            ["TYPE OF FUEL"],
        )
        == 3,
        lambda: xr.DataArray(
            if_then_else(
                select_incentives_electric_vehicles() == 0,
                lambda: low_incentives_electric_vehicles(),
                lambda: if_then_else(
                    select_incentives_electric_vehicles() == 1,
                    lambda: medium_incentives_electric_vehicles(),
                    lambda: if_then_else(
                        select_incentives_electric_vehicles() == 2,
                        lambda: high_incentives_electric_vehicles(),
                        lambda: if_then_else(
                            select_incentives_electric_vehicles() == 3,
                            lambda: very_high_incentives_electric_vehicles(),
                            lambda: low_incentives_electric_vehicles(),
                        ),
                    ),
                ),
            ),
            {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
            ["TYPE OF FUEL"],
        ),
        lambda: xr.DataArray(
            if_then_else(
                select_incentives_electric_vehicles() == 0,
                lambda: -low_incentives_electric_vehicles(),
                lambda: if_then_else(
                    select_incentives_electric_vehicles() == 1,
                    lambda: -medium_incentives_electric_vehicles(),
                    lambda: if_then_else(
                        select_incentives_electric_vehicles() == 2,
                        lambda: -high_incentives_electric_vehicles(),
                        lambda: if_then_else(
                            select_incentives_electric_vehicles() == 3,
                            lambda: -very_high_incentives_electric_vehicles(),
                            lambda: -low_incentives_electric_vehicles(),
                        ),
                    ),
                ),
            ),
            {"TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
            ["TYPE OF FUEL"],
        ),
    )


@component.add(
    name="HIGH INCENTIVES ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_incentives_electric_vehicles():
    return 0.75


@component.add(
    name="SWITCH INCREASE IN ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_increase_in_electric_vehicles"},
)
def switch_increase_in_electric_vehicles():
    """
    0 = OFF 1 = ON GET DIRECT CONSTANTS ('Policy.xlsx', 'Transport system', 'SWITCH_SUBSIDIES_ELECTRIC_VEHICLES*')
    """
    return _ext_constant_switch_increase_in_electric_vehicles()


_ext_constant_switch_increase_in_electric_vehicles = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "SWITCH_SUBSIDIES_ELECTRIC_VEHICLES*",
    {},
    _root,
    {},
    "_ext_constant_switch_increase_in_electric_vehicles",
)


@component.add(
    name="SWITCH GREEN URBAN PLANNING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_green_urban_planning"},
)
def switch_green_urban_planning():
    """
    0=OFF 1=ON
    """
    return _ext_constant_switch_green_urban_planning()


_ext_constant_switch_green_urban_planning = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SWITCH_GREEN_URBAN_PLANNING*",
    {},
    _root,
    {},
    "_ext_constant_switch_green_urban_planning",
)


@component.add(
    name="RATIO GREEN AREA TO WATER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_green_area_to_water"},
)
def ratio_green_area_to_water():
    return _ext_constant_ratio_green_area_to_water()


_ext_constant_ratio_green_area_to_water = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_GREEN_AREA_TO_WATER*",
    {},
    _root,
    {},
    "_ext_constant_ratio_green_area_to_water",
)


@component.add(
    name="BETA VEHICLE",
    units="Dmnl",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_vehicle"},
)
def beta_vehicle():
    return _ext_constant_beta_vehicle()


_ext_constant_beta_vehicle = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_beta_vehicle",
)

_ext_constant_beta_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_beta_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_beta_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_beta_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_beta_vehicle.add(
    r"Historical.xlsx",
    "Transport system",
    "BETA_VEHICLE_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="GREEN URBAN PLANNING OBJECTIVE",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_green_urban_planning_objective"},
)
def green_urban_planning_objective():
    return _ext_constant_green_urban_planning_objective()


_ext_constant_green_urban_planning_objective = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "GREEN_URBAN_PLANNING_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_green_urban_planning_objective",
)


@component.add(
    name="RATIO AGRICULTURE TO GREEN AREA",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_agriculture_to_green_area"},
)
def ratio_agriculture_to_green_area():
    return _ext_constant_ratio_agriculture_to_green_area()


_ext_constant_ratio_agriculture_to_green_area = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_AGRICULTURE_TO_GREEN_AREA*",
    {},
    _root,
    {},
    "_ext_constant_ratio_agriculture_to_green_area",
)


@component.add(
    name="FINAL YEAR INCREASE IN ELECTRIC VEHICLES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_increase_in_electric_vehicles"
    },
)
def final_year_increase_in_electric_vehicles():
    return _ext_constant_final_year_increase_in_electric_vehicles()


_ext_constant_final_year_increase_in_electric_vehicles = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "FINAL_YEAR_SUBSIDIES_ELECTRIC_VEHICLES*",
    {},
    _root,
    {},
    "_ext_constant_final_year_increase_in_electric_vehicles",
)


@component.add(
    name="VERY HIGH INCENTIVES ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_incentives_electric_vehicles():
    return 1


@component.add(
    name="INITIAL YEAR INCREASE IN ELECTRIC VEHICLES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_increase_in_electric_vehicles"
    },
)
def initial_year_increase_in_electric_vehicles():
    return _ext_constant_initial_year_increase_in_electric_vehicles()


_ext_constant_initial_year_increase_in_electric_vehicles = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "INITIAL_YEAR_SUBSIDIES_ELECTRIC_VEHICLES*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_increase_in_electric_vehicles",
)


@component.add(
    name="SELECT INCENTIVES ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_incentives_electric_vehicles"},
)
def select_incentives_electric_vehicles():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_incentives_electric_vehicles()


_ext_constant_select_incentives_electric_vehicles = ExtConstant(
    r"Policy.xlsx",
    "Transport system",
    "SELECT_SUBSIDIES_ELECTRIC_VEHICLES*",
    {},
    _root,
    {},
    "_ext_constant_select_incentives_electric_vehicles",
)


@component.add(
    name="RATIO OTHER TO GREEN AREA",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_other_to_green_area"},
)
def ratio_other_to_green_area():
    return _ext_constant_ratio_other_to_green_area()


_ext_constant_ratio_other_to_green_area = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_OTHER_TO_GREEN_AREA*",
    {},
    _root,
    {},
    "_ext_constant_ratio_other_to_green_area",
)


@component.add(
    name="RATIO WETLANDS TO GREEN AREA",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_wetlands_to_green_area"},
)
def ratio_wetlands_to_green_area():
    return _ext_constant_ratio_wetlands_to_green_area()


_ext_constant_ratio_wetlands_to_green_area = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "RATIO_WETLANDS_TO_GREEN_AREA*",
    {},
    _root,
    {},
    "_ext_constant_ratio_wetlands_to_green_area",
)


@component.add(
    name="INITIAL GREEN URBAN AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_green_urban_area"},
)
def initial_green_urban_area():
    return _ext_constant_initial_green_urban_area()


_ext_constant_initial_green_urban_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_GREEN_URBAN_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_green_urban_area",
)


@component.add(
    name="BUILT UP URBAN",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"urban": 1, "green_urban": 1},
)
def built_up_urban():
    return urban() - green_urban()


@component.add(
    name="LOW INCENTIVES ELECTRIC VEHICLES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_incentives_electric_vehicles():
    return 0.25


@component.add(
    name="GDP",
    units="Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"cs_population": 1, "gdp_per_capita": 1},
)
def gdp():
    return cs_population() * gdp_per_capita()


@component.add(
    name="DISTANCE COVERED",
    units="km",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_distance_covered"},
)
def distance_covered():
    return _ext_constant_distance_covered()


_ext_constant_distance_covered = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "DISTANCE_COVERED*",
    {},
    _root,
    {},
    "_ext_constant_distance_covered",
)


@component.add(
    name="VEHICLES EMISSION COEFFICIENT",
    units="tCO2/km",
    subscripts=["TYPE OF VEHICLE", "TYPE OF FUEL"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_vehicles_emission_coefficient"},
)
def vehicles_emission_coefficient():
    return _ext_constant_vehicles_emission_coefficient()


_ext_constant_vehicles_emission_coefficient = ExtConstant(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_TRUCK",
    {"TYPE OF VEHICLE": ["TRUCK"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
    _root,
    {
        "TYPE OF VEHICLE": _subscript_dict["TYPE OF VEHICLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
    "_ext_constant_vehicles_emission_coefficient",
)

_ext_constant_vehicles_emission_coefficient.add(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_VAN",
    {"TYPE OF VEHICLE": ["VAN"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_vehicles_emission_coefficient.add(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_BUS",
    {"TYPE OF VEHICLE": ["BUS"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_vehicles_emission_coefficient.add(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_CAR",
    {"TYPE OF VEHICLE": ["CAR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)

_ext_constant_vehicles_emission_coefficient.add(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_MOTORCYCLE",
    {
        "TYPE OF VEHICLE": ["MOTORCYCLE"],
        "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"],
    },
)

_ext_constant_vehicles_emission_coefficient.add(
    r"Historical.xlsx",
    "Transport system",
    "VEHICLES_EMISSION_COEFFICIENT_TRACTOR",
    {"TYPE OF VEHICLE": ["TRACTOR"], "TYPE OF FUEL": _subscript_dict["TYPE OF FUEL"]},
)


@component.add(
    name="ARTIFICIAL SNOW PRODUCTION YEARLY",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"artificial_snow_production_monthly": 1},
)
def artificial_snow_production_yearly():
    return sum(
        artificial_snow_production_monthly().rename({"MONTHS": "MONTHS!"}),
        dim=["MONTHS!"],
    )


@component.add(
    name="RATIO 2018 AND 2045",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_energy_and_gdp_2045": 1, "ratio_energy_and_gdp_2018": 1},
)
def ratio_2018_and_2045():
    return 1 - zidz(ratio_energy_and_gdp_2045(), ratio_energy_and_gdp_2018())


@component.add(
    name="RATIO 2018 AND 2030",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_energy_and_gdp_2030": 1, "ratio_energy_and_gdp_2018": 1},
)
def ratio_2018_and_2030():
    return 1 - zidz(ratio_energy_and_gdp_2030(), ratio_energy_and_gdp_2018())


@component.add(
    name="RATIO ENERGY AND GDP 2045",
    units="MWh/euro per capita",
    comp_type="Stateful",
    comp_subtype="SampleIfTrue",
    depends_on={"_sampleiftrue_ratio_energy_and_gdp_2045": 1},
    other_deps={
        "_sampleiftrue_ratio_energy_and_gdp_2045": {
            "initial": {},
            "step": {"time": 1, "ratio_energy_consumption_and_gdp_per_capita": 1},
        }
    },
)
def ratio_energy_and_gdp_2045():
    return _sampleiftrue_ratio_energy_and_gdp_2045()


_sampleiftrue_ratio_energy_and_gdp_2045 = SampleIfTrue(
    lambda: time() == 2045,
    lambda: ratio_energy_consumption_and_gdp_per_capita(),
    lambda: 0,
    "_sampleiftrue_ratio_energy_and_gdp_2045",
)


@component.add(
    name="RATIO ENERGY CONSUMPTION AND GDP PER CAPITA",
    units="MWh/euro per capita",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption": 1, "gdp_per_capita": 1},
)
def ratio_energy_consumption_and_gdp_per_capita():
    return energy_consumption() / gdp_per_capita()


@component.add(
    name="RATIO ENERGY AND GDP 2030",
    units="MWh/euro per capita",
    comp_type="Stateful",
    comp_subtype="SampleIfTrue",
    depends_on={"_sampleiftrue_ratio_energy_and_gdp_2030": 1},
    other_deps={
        "_sampleiftrue_ratio_energy_and_gdp_2030": {
            "initial": {},
            "step": {"time": 1, "ratio_energy_consumption_and_gdp_per_capita": 1},
        }
    },
)
def ratio_energy_and_gdp_2030():
    return _sampleiftrue_ratio_energy_and_gdp_2030()


_sampleiftrue_ratio_energy_and_gdp_2030 = SampleIfTrue(
    lambda: time() == 2030,
    lambda: ratio_energy_consumption_and_gdp_per_capita(),
    lambda: 0,
    "_sampleiftrue_ratio_energy_and_gdp_2030",
)


@component.add(
    name="ENERGY CONSUMPTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_energy_efficiency_increase": 3,
        "energy_consumption_domestic_and_industry": 4,
        "energy_consumption_water_treatment": 4,
        "energy_used_for_snow_production": 4,
        "energy_consumption_desalination": 4,
        "energy_efficiency_objective": 2,
        "initial_year_energy_efficiency_increase": 3,
        "time": 3,
        "final_year_energy_efficiency_increase": 2,
    },
)
def energy_consumption():
    return if_then_else(
        switch_energy_efficiency_increase() == 0,
        lambda: energy_consumption_domestic_and_industry()
        + energy_consumption_desalination()
        + energy_consumption_water_treatment()
        + energy_used_for_snow_production(),
        lambda: if_then_else(
            np.logical_and(
                switch_energy_efficiency_increase() == 1,
                time() < initial_year_energy_efficiency_increase(),
            ),
            lambda: energy_consumption_domestic_and_industry()
            + energy_consumption_desalination()
            + energy_consumption_water_treatment()
            + energy_used_for_snow_production(),
            lambda: if_then_else(
                np.logical_and(
                    switch_energy_efficiency_increase() == 1,
                    time() > final_year_energy_efficiency_increase(),
                ),
                lambda: (
                    energy_used_for_snow_production()
                    + energy_consumption_desalination()
                    + energy_consumption_water_treatment()
                    + energy_consumption_domestic_and_industry()
                )
                * (1 - energy_efficiency_objective()),
                lambda: (
                    energy_used_for_snow_production()
                    + energy_consumption_desalination()
                    + energy_consumption_water_treatment()
                    + energy_consumption_domestic_and_industry()
                )
                * (
                    1
                    - energy_efficiency_objective()
                    * (time() - initial_year_energy_efficiency_increase())
                    / (
                        final_year_energy_efficiency_increase()
                        - initial_year_energy_efficiency_increase()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY CONSUMPTION DOMESTIC AND INDUSTRY",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"consumption_per_capita": 1, "cs_population": 1},
)
def energy_consumption_domestic_and_industry():
    return consumption_per_capita() * cs_population()


@component.add(
    name="RATIO ENERGY AND GDP 2018",
    units="MWh/euro per capita",
    comp_type="Stateful",
    comp_subtype="SampleIfTrue",
    depends_on={"_sampleiftrue_ratio_energy_and_gdp_2018": 1},
    other_deps={
        "_sampleiftrue_ratio_energy_and_gdp_2018": {
            "initial": {"ratio_energy_consumption_and_gdp_per_capita": 1},
            "step": {"time": 1, "ratio_energy_consumption_and_gdp_per_capita": 1},
        }
    },
)
def ratio_energy_and_gdp_2018():
    return _sampleiftrue_ratio_energy_and_gdp_2018()


_sampleiftrue_ratio_energy_and_gdp_2018 = SampleIfTrue(
    lambda: time() == 2018,
    lambda: ratio_energy_consumption_and_gdp_per_capita(),
    lambda: ratio_energy_consumption_and_gdp_per_capita(),
    "_sampleiftrue_ratio_energy_and_gdp_2018",
)


@component.add(
    name="MAX BIOMASS INCREASE TOT",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"max_biomass_increase_per_year": 1},
)
def max_biomass_increase_tot():
    return max_biomass_increase_per_year() * (2060 - 2018)


@component.add(
    name="NEW BIOMASS CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_biomass_capacity": 2,
        "time": 2,
        "final_year_biomass_policy_capacity": 2,
        "biomass_capacity": 1,
        "max_biomass_capacity": 1,
        "biomass_capacity_objective": 1,
        "initial_year_biomass_policy_capacity": 2,
    },
)
def new_biomass_capacity():
    """
    IF THEN ELSE(SWITCH BIOMASS CAPACITY=0,MIN(MARKET DRIVEN INVESTMENTS FOR BIOMASS CAPACITY,MAX BIOMASS CAPACITY-BIOMASS CAPACITY ), IF THEN ELSE(SWITCH BIOMASS CAPACITY=1:AND:(Time<INITIAL YEAR BIOMASS POLICY CAPACITY:OR:Time>FINAL YEAR BIOMASS POLICY CAPACITY ),MIN(MARKET DRIVEN INVESTMENTS FOR BIOMASS CAPACITY,MAX BIOMASS CAPACITY-BIOMASS CAPACITY), IF THEN ELSE( BIOMASS CAPACITY < MAX BIOMASS CAPACITY, IF THEN ELSE(((BIOMASS CAPACITY OBJECTIVE)/(FINAL YEAR BIOMASS POLICY CAPACITY - INITIAL YEAR BIOMASS POLICY CAPACITY))>MARKET DRIVEN INVESTMENTS FOR BIOMASS CAPACITY,MIN((BIOMASS CAPACITY OBJECTIVE )/(FINAL YEAR BIOMASS POLICY CAPACITY - INITIAL YEAR BIOMASS POLICY CAPACITY), MAX BIOMASS CAPACITY-BIOMASS CAPACITY),MIN (MARKET DRIVEN INVESTMENTS FOR BIOMASS CAPACITY,MAX BIOMASS CAPACITY-BIOMASS CAPACITY)), 0 ) ) )
    """
    return if_then_else(
        switch_biomass_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_biomass_capacity() == 1,
                np.logical_or(
                    time() < initial_year_biomass_policy_capacity(),
                    time() > final_year_biomass_policy_capacity(),
                ),
            ),
            lambda: 0,
            lambda: float(
                np.minimum(
                    biomass_capacity_objective()
                    / (
                        final_year_biomass_policy_capacity()
                        - initial_year_biomass_policy_capacity()
                    ),
                    max_biomass_capacity() - biomass_capacity(),
                )
            ),
        ),
    )


@component.add(
    name="MAX BIOMASS CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"initial_biomass_capacity": 1, "max_biomass_increase_tot": 1},
)
def max_biomass_capacity():
    return initial_biomass_capacity() + max_biomass_increase_tot()


@component.add(
    name="VOLUME SNOW ACCUMULATION",
    units="m3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tot_snow_accumulation_cs2": 1, "ski_slope_area": 1},
)
def volume_snow_accumulation():
    return tot_snow_accumulation_cs2() * ski_slope_area() * 0.01


@component.add(
    name="TOT SNOW ACCUMULATION CS2",
    units="cm/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"switch_cs": 1, "snow_accumulation_monthly": 28},
)
def tot_snow_accumulation_cs2():
    return if_then_else(
        switch_cs() == 2,
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 1,
            lambda: xr.DataArray(
                float(snow_accumulation_monthly().loc["OCTOBER"]) * 0.7,
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 2,
                lambda: xr.DataArray(
                    float(snow_accumulation_monthly().loc["NOVEMBER"]) * 0.7
                    + float(snow_accumulation_monthly().loc["OCTOBER"]) * 0.7 * 0.6,
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                ),
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 3,
                    lambda: xr.DataArray(
                        float(snow_accumulation_monthly().loc["DECEMBER"]) * 0.7
                        + (
                            float(snow_accumulation_monthly().loc["NOVEMBER"]) * 0.7
                            + float(snow_accumulation_monthly().loc["OCTOBER"])
                            * 0.7
                            * 0.6
                        )
                        * 0.6,
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    ),
                    lambda: if_then_else(
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        )
                        == 4,
                        lambda: xr.DataArray(
                            float(snow_accumulation_monthly().loc["JANUARY"]) * 0.7
                            + (
                                float(snow_accumulation_monthly().loc["DECEMBER"])
                                + (
                                    float(snow_accumulation_monthly().loc["NOVEMBER"])
                                    + float(snow_accumulation_monthly().loc["OCTOBER"])
                                    * 0.6
                                )
                                * 0.6
                            )
                            * 0.7
                            * 0.6,
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        ),
                        lambda: if_then_else(
                            xr.DataArray(
                                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            )
                            == 5,
                            lambda: xr.DataArray(
                                float(snow_accumulation_monthly().loc["FEBRUARY"]) * 0.7
                                + (
                                    float(snow_accumulation_monthly().loc["JANUARY"])
                                    + (
                                        float(
                                            snow_accumulation_monthly().loc["DECEMBER"]
                                        )
                                        + (
                                            float(
                                                snow_accumulation_monthly().loc[
                                                    "NOVEMBER"
                                                ]
                                            )
                                            + float(
                                                snow_accumulation_monthly().loc[
                                                    "OCTOBER"
                                                ]
                                            )
                                            * 0.6
                                        )
                                        * 0.6
                                    )
                                    * 0.6
                                )
                                * 0.7
                                * 0.6,
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            ),
                            lambda: if_then_else(
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 6,
                                lambda: xr.DataArray(
                                    float(snow_accumulation_monthly().loc["MARCH"])
                                    * 0.7
                                    + (
                                        float(
                                            snow_accumulation_monthly().loc["FEBRUARY"]
                                        )
                                        + (
                                            float(
                                                snow_accumulation_monthly().loc[
                                                    "JANUARY"
                                                ]
                                            )
                                            + (
                                                float(
                                                    snow_accumulation_monthly().loc[
                                                        "DECEMBER"
                                                    ]
                                                )
                                                + (
                                                    float(
                                                        snow_accumulation_monthly().loc[
                                                            "NOVEMBER"
                                                        ]
                                                    )
                                                    + float(
                                                        snow_accumulation_monthly().loc[
                                                            "OCTOBER"
                                                        ]
                                                    )
                                                    * 0.6
                                                )
                                                * 0.6
                                            )
                                            * 0.6
                                        )
                                        * 0.6
                                    )
                                    * 0.7
                                    * 0.5,
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                ),
                                lambda: if_then_else(
                                    xr.DataArray(
                                        np.arange(
                                            1, len(_subscript_dict["MONTHS"]) + 1
                                        ),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    )
                                    == 7,
                                    lambda: xr.DataArray(
                                        float(snow_accumulation_monthly().loc["APRIL"])
                                        * 0.7
                                        + (
                                            float(
                                                snow_accumulation_monthly().loc["MARCH"]
                                            )
                                            + (
                                                float(
                                                    snow_accumulation_monthly().loc[
                                                        "FEBRUARY"
                                                    ]
                                                )
                                                + (
                                                    float(
                                                        snow_accumulation_monthly().loc[
                                                            "JANUARY"
                                                        ]
                                                    )
                                                    + (
                                                        float(
                                                            snow_accumulation_monthly().loc[
                                                                "DECEMBER"
                                                            ]
                                                        )
                                                        + (
                                                            float(
                                                                snow_accumulation_monthly().loc[
                                                                    "NOVEMBER"
                                                                ]
                                                            )
                                                            + float(
                                                                snow_accumulation_monthly().loc[
                                                                    "OCTOBER"
                                                                ]
                                                            )
                                                            * 0.6
                                                        )
                                                        * 0.6
                                                    )
                                                    * 0.6
                                                )
                                                * 0.6
                                            )
                                            * 0.6
                                        )
                                        * 0.7
                                        * 0.4,
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    ),
                                    lambda: xr.DataArray(
                                        0,
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="SNOW VOLUME FOR SKIING",
    units="m3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"minimum_ski_depth": 1, "ski_slope_area": 1},
)
def snow_volume_for_skiing():
    return minimum_ski_depth() * ski_slope_area()


@component.add(
    name="BUILDING AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_building_area"},
)
def building_area():
    return _ext_constant_building_area()


_ext_constant_building_area = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "BUILDING_AREA*",
    {},
    _root,
    {},
    "_ext_constant_building_area",
)


@component.add(
    name="MAX BIOMASS INCREASE PER YEAR",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "boreal_broadleaved_summergreen": 1,
        "bbs": 1,
        "bne": 1,
        "boreal_needleleaved_evergreen": 1,
        "temperate_broadleaved_evergreen": 1,
        "tbe": 1,
        "temperate_broadleaved_summergreen": 1,
        "tbs": 1,
        "temperate_needleleaved_evergreen": 1,
        "tne": 1,
        "factor_ha_to_km2": 1,
        "kg_to_tonnes": 1,
        "biomass_energy_content": 1,
        "biomass_performance": 1,
        "hours_in_a_year": 1,
    },
)
def max_biomass_increase_per_year():
    """
    (((BIOMASS STOCK/FOREST)*BIOMASS ENERGY CONTENT)/(BIOMASS PERFORMANCE*HOURS IN A YEAR))*(AVAILABLE FOREST LAND)
    """
    return (
        (
            boreal_broadleaved_summergreen() * bbs()
            + boreal_needleleaved_evergreen() * bne()
            + temperate_broadleaved_evergreen() * tbe()
            + temperate_broadleaved_summergreen() * tbs()
            + temperate_needleleaved_evergreen() * tne()
        )
        * factor_ha_to_km2()
        * kg_to_tonnes()
        * biomass_energy_content()
    ) / (biomass_performance() * hours_in_a_year())


@component.add(
    name="WATER FLOW YEARLY TO SEA",
    units="Hm3/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"runoff_to_river_flow": 1, "available_water_from_runoff_yearly": 1},
)
def water_flow_yearly_to_sea():
    return runoff_to_river_flow() - available_water_from_runoff_yearly()


@component.add(
    name="EFFECTIVE SNOWMELT",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "cumulative_snowmelt_capacity": 4,
        "total_snow_accumulation_corrected": 4,
        "snowmelt": 2,
        "snow_output": 1,
    },
)
def effective_snowmelt():
    return (
        if_then_else(
            cumulative_snowmelt_capacity() >= total_snow_accumulation_corrected(),
            lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
            lambda: if_then_else(
                cumulative_snowmelt_capacity() < total_snow_accumulation_corrected(),
                lambda: snowmelt(),
                lambda: if_then_else(
                    cumulative_snowmelt_capacity() + snowmelt()
                    < total_snow_accumulation_corrected(),
                    lambda: total_snow_accumulation_corrected()
                    - cumulative_snowmelt_capacity(),
                    lambda: xr.DataArray(
                        0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]
                    ),
                ),
            ),
        )
        + snow_output()
    )


@component.add(
    name="TOTAL SNOW ACCUMULATION CORRECTED",
    units="cm",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "snow_accumulation_monthly": 9,
        "snow_accumulation_monthly_previous_year": 3,
    },
)
def total_snow_accumulation_corrected():
    return (
        float(snow_accumulation_monthly().loc["JANUARY"])
        + float(snow_accumulation_monthly().loc["FEBRUARY"])
        + float(snow_accumulation_monthly().loc["MARCH"])
        + float(snow_accumulation_monthly().loc["APRIL"])
        + float(snow_accumulation_monthly().loc["MAY"])
        + float(snow_accumulation_monthly().loc["JUNE"])
        + float(snow_accumulation_monthly().loc["JULY"])
        + float(snow_accumulation_monthly().loc["AUGUST"])
        + float(snow_accumulation_monthly().loc["SEPTEMBER"])
        + float(snow_accumulation_monthly_previous_year().loc["OCTOBER"])
        + float(snow_accumulation_monthly_previous_year().loc["NOVEMBER"])
        + float(snow_accumulation_monthly_previous_year().loc["DECEMBER"])
    )


@component.add(
    name="WATER SUPPLY",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "available_water_from_runoff_yearly": 1,
        "groundwater_inputs": 1,
        "desalination": 1,
        "transboundary_water": 1,
        "water_treatment": 1,
        "storage_outputs": 1,
    },
)
def water_supply():
    return (
        available_water_from_runoff_yearly()
        + groundwater_inputs()
        + desalination()
        + sum(transboundary_water().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        + water_treatment()
        - storage_outputs()
    )


@component.add(
    name="RUNOFF TO RIVER FLOW",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"available_runoff": 1},
)
def runoff_to_river_flow():
    return sum(available_runoff().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])


@component.add(
    name="AVAILABLE WATER FROM RUNOFF YEARLY",
    units="Hm3/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_reservoir_capacity": 3, "available_runoff": 3},
)
def available_water_from_runoff_yearly():
    return if_then_else(
        total_reservoir_capacity() == 0,
        lambda: sum(available_runoff().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        * 0.1,
        lambda: if_then_else(
            total_reservoir_capacity()
            > sum(available_runoff().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]),
            lambda: sum(
                available_runoff().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]
            ),
            lambda: total_reservoir_capacity() * 1.25,
        ),
    )


@component.add(
    name="DIFFERENCE",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "cumulative_snowmelt_capacity": 2,
        "total_snow_accumulation_corrected": 2,
    },
)
def difference():
    return if_then_else(
        cumulative_snowmelt_capacity() - total_snow_accumulation_corrected() > 0,
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: -1
        * (cumulative_snowmelt_capacity() - total_snow_accumulation_corrected()),
    )


@component.add(
    name="SNOW ACCUMULATION MONTHLY PREVIOUS YEAR",
    units="cm/Month",
    subscripts=["MONTHS"],
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_snow_accumulation_monthly_previous_year": 1},
    other_deps={
        "_delayfixed_snow_accumulation_monthly_previous_year": {
            "initial": {},
            "step": {"snow_accumulation_monthly": 1},
        }
    },
)
def snow_accumulation_monthly_previous_year():
    return _delayfixed_snow_accumulation_monthly_previous_year()


_delayfixed_snow_accumulation_monthly_previous_year = DelayFixed(
    lambda: snow_accumulation_monthly(),
    lambda: 1,
    lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    time_step,
    "_delayfixed_snow_accumulation_monthly_previous_year",
)


@component.add(
    name="SNOW VALIDATION",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"difference": 2, "total_snow_accumulation_corrected": 1},
)
def snow_validation():
    return if_then_else(
        difference() >= 0.8 * total_snow_accumulation_corrected(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: difference(),
    )


@component.add(
    name="SWITCH SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_sustainable_snowmaking_and_conservation_practices"
    },
)
def switch_sustainable_snowmaking_and_conservation_practices():
    """
    0 = OFF 1 = ON GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SWITCH_SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES*')
    """
    return _ext_constant_switch_sustainable_snowmaking_and_conservation_practices()


_ext_constant_switch_sustainable_snowmaking_and_conservation_practices = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES*",
    {},
    _root,
    {},
    "_ext_constant_switch_sustainable_snowmaking_and_conservation_practices",
)


@component.add(
    name="INITIAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_sustainable_snowmaking_and_conservation_practices"
    },
)
def initial_year_sustainable_snowmaking_and_conservation_practices():
    return (
        _ext_constant_initial_year_sustainable_snowmaking_and_conservation_practices()
    )


_ext_constant_initial_year_sustainable_snowmaking_and_conservation_practices = (
    ExtConstant(
        r"Policy.xlsx",
        "Water system",
        "INITIAL_YEAR_SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES*",
        {},
        _root,
        {},
        "_ext_constant_initial_year_sustainable_snowmaking_and_conservation_practices",
    )
)


@component.add(
    name="SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_sustainable_snowmaking_and_conservation_practices_objective"
    },
)
def sustainable_snowmaking_and_conservation_practices_objective():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES_OBJECTIVE*')
    """
    return _ext_constant_sustainable_snowmaking_and_conservation_practices_objective()


_ext_constant_sustainable_snowmaking_and_conservation_practices_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_sustainable_snowmaking_and_conservation_practices_objective",
)


@component.add(
    name="FINAL YEAR SUSTAINABLE SNOWMAKING AND CONSERVATION PRACTICES",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_sustainable_snowmaking_and_conservation_practices"
    },
)
def final_year_sustainable_snowmaking_and_conservation_practices():
    return _ext_constant_final_year_sustainable_snowmaking_and_conservation_practices()


_ext_constant_final_year_sustainable_snowmaking_and_conservation_practices = (
    ExtConstant(
        r"Policy.xlsx",
        "Water system",
        "FINAL_YEAR_SUSTAINABLE_SNOWMAKING_AND_CONSERVATION_PRACTICES*",
        {},
        _root,
        {},
        "_ext_constant_final_year_sustainable_snowmaking_and_conservation_practices",
    )
)


@component.add(
    name="INITIAL RATIO ENERGY PER SNOW PRODUCTION",
    units="MWh/m3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_ratio_energy_per_snow_production"
    },
)
def initial_ratio_energy_per_snow_production():
    return _ext_constant_initial_ratio_energy_per_snow_production()


_ext_constant_initial_ratio_energy_per_snow_production = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_RATIO_ENERGY_PER_SNOW_PRODUCTION*",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_energy_per_snow_production",
)


@component.add(
    name="SKI SLOPE AREA",
    units="m2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ski_slope_area"},
)
def ski_slope_area():
    return _ext_constant_ski_slope_area()


_ext_constant_ski_slope_area = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "SKI_SLOPE_AREA*",
    {},
    _root,
    {},
    "_ext_constant_ski_slope_area",
)


@component.add(
    name="MINIMUM SKI DEPTH",
    units="m",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_ski_depth"},
)
def minimum_ski_depth():
    return _ext_constant_minimum_ski_depth()


_ext_constant_minimum_ski_depth = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "MINIMUM_SKI_DEPTH*",
    {},
    _root,
    {},
    "_ext_constant_minimum_ski_depth",
)


@component.add(
    name="MAX HYDRO CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "hours_of_hydro_production_per_year": 2,
        "max_potential_hydro": 1,
        "hydro_performance": 1,
    },
)
def max_hydro_capacity():
    return if_then_else(
        hours_of_hydro_production_per_year() <= 0,
        lambda: 0,
        lambda: max_potential_hydro()
        / (hours_of_hydro_production_per_year() * hydro_performance()),
    )


@component.add(
    name="WIND PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"wind_capacity": 1, "wind_performance": 1, "hours_in_a_year": 1},
)
def wind_production():
    return wind_capacity() * wind_performance() * hours_in_a_year()


@component.add(
    name="NON RENEWABLE PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "non_renewable_performance": 1,
        "non_renewable_capacity": 1,
        "hours_in_a_year": 1,
    },
)
def non_renewable_production():
    return non_renewable_performance() * non_renewable_capacity() * hours_in_a_year()


@component.add(
    name="MAX POTENTIAL ROOFTOP",
    units="MW/km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_max_potential_rooftop"},
)
def max_potential_rooftop():
    return _ext_constant_max_potential_rooftop()


_ext_constant_max_potential_rooftop = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MAX_POTENTIAL_ROOFTOP*",
    {},
    _root,
    {},
    "_ext_constant_max_potential_rooftop",
)


@component.add(
    name="ROOFTOP PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"rooftop_capacity": 1, "rooftop_performance": 1, "hours_in_a_year": 1},
)
def rooftop_production():
    return rooftop_capacity() * rooftop_performance() * hours_in_a_year()


@component.add(
    name="MAX WIND CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"max_potential_wind": 1},
)
def max_wind_capacity():
    return max_potential_wind()


@component.add(
    name="BIOMASS PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"biomass_capacity": 1, "biomass_performance": 1, "hours_in_a_year": 1},
)
def biomass_production():
    return biomass_capacity() * biomass_performance() * hours_in_a_year()


@component.add(
    name="REAL UNEMPLOYED",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_real_unemployed": 1},
    other_deps={
        "_smooth_real_unemployed": {
            "initial": {"unemployed": 2, "cs_population": 2},
            "step": {"unemployed": 2, "cs_population": 2},
        }
    },
)
def real_unemployed():
    return _smooth_real_unemployed()


_smooth_real_unemployed = Smooth(
    lambda: if_then_else(
        unemployed() < 0.01 * cs_population(),
        lambda: 0.01 * cs_population(),
        lambda: unemployed(),
    ),
    lambda: 2,
    lambda: if_then_else(
        unemployed() < 0.01 * cs_population(),
        lambda: 0.01 * cs_population(),
        lambda: unemployed(),
    ),
    lambda: 1,
    "_smooth_real_unemployed",
)


@component.add(
    name="UNEMPLOYMENT RATE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"real_unemployed": 1, "cs_population": 1},
)
def unemployment_rate():
    return real_unemployed() / cs_population()


@component.add(
    name="NEW NON RENEWABLE CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_increase_non_renewable_capacity": 2,
        "time": 1,
        "final_year_increase_non_renewable_capacity": 2,
        "increase_non_renewable_capacity_objective": 1,
        "initial_time": 1,
    },
)
def new_non_renewable_capacity():
    return if_then_else(
        switch_increase_non_renewable_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_increase_non_renewable_capacity() == 1,
                time() > final_year_increase_non_renewable_capacity(),
            ),
            lambda: 0,
            lambda: increase_non_renewable_capacity_objective()
            / (final_year_increase_non_renewable_capacity() - initial_time()),
        ),
    )


@component.add(
    name="RATIO IRRIGATED CROP AREA",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_irrigated_crop_area_not_normalized": 1},
)
def ratio_irrigated_crop_area():
    return float(np.minimum(ratio_irrigated_crop_area_not_normalized(), 1))


@component.add(
    name="RATIO IRRIGATED CROP AREA NOT NORMALIZED",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_ratio_irrigated_crop_area_not_normalized": 1},
    other_deps={
        "_integ_ratio_irrigated_crop_area_not_normalized": {
            "initial": {"initial_ratio_irrigated_crop_area": 1},
            "step": {"increase_ratio_irrigated_crop_new": 1},
        }
    },
)
def ratio_irrigated_crop_area_not_normalized():
    return _integ_ratio_irrigated_crop_area_not_normalized()


_integ_ratio_irrigated_crop_area_not_normalized = Integ(
    lambda: increase_ratio_irrigated_crop_new(),
    lambda: initial_ratio_irrigated_crop_area(),
    "_integ_ratio_irrigated_crop_area_not_normalized",
)


@component.add(
    name="RATIO RAINFED CROP AREA",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio_irrigated_crop_area": 1},
)
def ratio_rainfed_crop_area():
    return 1 - ratio_irrigated_crop_area()


@component.add(
    name="SWITCH INCREASE IRRIGATED CROP",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_increase_irrigated_crop"},
)
def switch_increase_irrigated_crop():
    """
    OFF=0 ON=1
    """
    return _ext_constant_switch_increase_irrigated_crop()


_ext_constant_switch_increase_irrigated_crop = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "SWITCH_INCREASE_IRRIGATED_CROP*",
    {},
    _root,
    {},
    "_ext_constant_switch_increase_irrigated_crop",
)


@component.add(
    name="SWITCH INCREASE NON RENEWABLE CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_increase_non_renewable_capacity"},
)
def switch_increase_non_renewable_capacity():
    """
    OFF=0 ON=1 GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_INCREASE_NON_RENEWABLE_CAPACITY*')
    """
    return _ext_constant_switch_increase_non_renewable_capacity()


_ext_constant_switch_increase_non_renewable_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_INCREASE_NON_RENEWABLE_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_increase_non_renewable_capacity",
)


@component.add(
    name="INITIAL YEAR INCREASE IRRIGATED CROP",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_increase_irrigated_crop"},
)
def initial_year_increase_irrigated_crop():
    return _ext_constant_initial_year_increase_irrigated_crop()


_ext_constant_initial_year_increase_irrigated_crop = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "INITIAL_YEAR_INCREASE_IRRIGATED_CROP*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_increase_irrigated_crop",
)


@component.add(
    name="INCREASE IRRIGATED CROP OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_increase_irrigated_crop_objective"},
)
def increase_irrigated_crop_objective():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Crop system', 'INCREASE_IRRIGATED_CROP_OBJECTIVE*')
    """
    return _ext_constant_increase_irrigated_crop_objective()


_ext_constant_increase_irrigated_crop_objective = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "INCREASE_IRRIGATED_CROP_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_increase_irrigated_crop_objective",
)


@component.add(
    name="FINAL YEAR INCREASE IRRIGATED CROP",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_increase_irrigated_crop"},
)
def final_year_increase_irrigated_crop():
    return _ext_constant_final_year_increase_irrigated_crop()


_ext_constant_final_year_increase_irrigated_crop = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "FINAL_YEAR_INCREASE_IRRIGATED_CROP*",
    {},
    _root,
    {},
    "_ext_constant_final_year_increase_irrigated_crop",
)


@component.add(
    name="FINAL YEAR INCREASE NON RENEWABLE CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_increase_non_renewable_capacity"
    },
)
def final_year_increase_non_renewable_capacity():
    return _ext_constant_final_year_increase_non_renewable_capacity()


_ext_constant_final_year_increase_non_renewable_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_INCREASE_NON_RENEWABLE_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_increase_non_renewable_capacity",
)


@component.add(
    name="AGRICULTURE TO WATER",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"select_location_water_storage": 1, "water_area": 1},
)
def agriculture_to_water():
    return if_then_else(
        select_location_water_storage() == 3, lambda: water_area(), lambda: 0
    )


@component.add(
    name="FOREST TO WATER",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"select_location_water_storage": 1, "water_area": 1},
)
def forest_to_water():
    return if_then_else(
        select_location_water_storage() == 4, lambda: water_area(), lambda: 0
    )


@component.add(
    name="WATER TREATMENT",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_water_treatment_increase": 3,
        "initial_water_treatment": 4,
        "initial_year_water_treatment_increase": 3,
        "time": 3,
        "water_treatment_increase_objective": 2,
        "final_year_water_treatment_increase": 2,
    },
)
def water_treatment():
    return if_then_else(
        switch_water_treatment_increase() == 0,
        lambda: initial_water_treatment(),
        lambda: if_then_else(
            np.logical_and(
                switch_water_treatment_increase() == 1,
                time() <= initial_year_water_treatment_increase(),
            ),
            lambda: initial_water_treatment(),
            lambda: if_then_else(
                np.logical_and(
                    switch_water_treatment_increase() == 1,
                    time() > final_year_water_treatment_increase(),
                ),
                lambda: initial_water_treatment()
                + water_treatment_increase_objective(),
                lambda: initial_water_treatment()
                + water_treatment_increase_objective()
                * (
                    (time() - initial_year_water_treatment_increase())
                    / (
                        final_year_water_treatment_increase()
                        - initial_year_water_treatment_increase()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="DESALINATION",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_cs": 1,
        "time": 3,
        "initial_year_desalination_increase": 3,
        "initial_desalination": 4,
        "switch_desalination_increase_policy": 3,
        "final_year_desalination_increase": 2,
        "desalination_objective": 2,
    },
)
def desalination():
    return if_then_else(
        switch_cs() == 2,
        lambda: 0,
        lambda: if_then_else(
            switch_desalination_increase_policy() == 0,
            lambda: initial_desalination(),
            lambda: if_then_else(
                np.logical_and(
                    switch_desalination_increase_policy() == 1,
                    time() < initial_year_desalination_increase(),
                ),
                lambda: initial_desalination(),
                lambda: if_then_else(
                    np.logical_and(
                        switch_desalination_increase_policy() == 1,
                        time() > final_year_desalination_increase(),
                    ),
                    lambda: initial_desalination() + desalination_objective(),
                    lambda: initial_desalination()
                    + desalination_objective()
                    * (
                        (time() - initial_year_desalination_increase())
                        / (
                            final_year_desalination_increase()
                            - initial_year_desalination_increase()
                        )
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY CONSUMPTION WATER TREATMENT URBAN",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption_water_treatment": 1, "ratio_water_demand_urban": 1},
)
def energy_consumption_water_treatment_urban():
    return energy_consumption_water_treatment() * ratio_water_demand_urban()


@component.add(
    name="ENERGY CONSUMPTION WATER TREATMENT AGRICULTURE",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_consumption_water_treatment": 1,
        "ratio_water_demand_agriculture": 1,
    },
)
def energy_consumption_water_treatment_agriculture():
    return energy_consumption_water_treatment() * ratio_water_demand_agriculture()


@component.add(
    name="ENERGY CONSUMPTION DESALINATION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"desalination": 1, "mwh_per_hm3_desalinated_water": 1},
)
def energy_consumption_desalination():
    return desalination() * mwh_per_hm3_desalinated_water()


@component.add(
    name="ENERGY CONSUMPTION DESALINATION URBAN",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption_desalination": 1, "ratio_water_demand_urban": 1},
)
def energy_consumption_desalination_urban():
    return energy_consumption_desalination() * ratio_water_demand_urban()


@component.add(
    name="ENERGY CONSUMPTION WATER TREATMENT INDUSTRY",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_consumption_water_treatment": 1,
        "ratio_water_demand_industry": 1,
    },
)
def energy_consumption_water_treatment_industry():
    return energy_consumption_water_treatment() * ratio_water_demand_industry()


@component.add(
    name="ENERGY CONSUMPTION DESALINATION INDUSTRY",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption_desalination": 1, "ratio_water_demand_industry": 1},
)
def energy_consumption_desalination_industry():
    return energy_consumption_desalination() * ratio_water_demand_industry()


@component.add(
    name="ENERGY CONSUMPTION DESALINATION AGRICULTURE",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "energy_consumption_desalination": 1,
        "ratio_water_demand_agriculture": 1,
    },
)
def energy_consumption_desalination_agriculture():
    return energy_consumption_desalination() * ratio_water_demand_agriculture()


@component.add(
    name="HYDRO PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "hydropower_capacity": 1,
        "hydro_performance": 1,
        "hours_of_hydro_production_per_year": 1,
    },
)
def hydro_production():
    return (
        hydropower_capacity()
        * hydro_performance()
        * hours_of_hydro_production_per_year()
    )


@component.add(
    name="TOTAL FLOW",
    units="m3/s",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_cs": 1,
        "month_to_hours": 3,
        "available_runoff": 2,
        "transboundary_water": 1,
    },
)
def total_flow():
    """
    IF THEN ELSE(SWITCH CS=4,MONTH TO HOURS[MONTHS]*RIVER FLOW[MONTHS]+(1e+06)*TRANSBOUNDARY WATER[MONTHS]/(MONTH TO HOURS[MONTHS]*60*60), MONTH TO HOURS[MONTHS]*RIVER FLOW[MONTHS])
    """
    return if_then_else(
        switch_cs() == 4,
        lambda: month_to_hours() * available_runoff()
        + 1000000.0 * transboundary_water() / (month_to_hours() * 60 * 60),
        lambda: month_to_hours() * available_runoff(),
    )


@component.add(
    name="MONTH TO HOURS",
    units="Hours",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"month_days": 1},
)
def month_to_hours():
    return month_days() * 24


@component.add(
    name="HOURS OF HYDRO PRODUCTION PER YEAR",
    units="Hours",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"hours_of_production": 1},
)
def hours_of_hydro_production_per_year():
    return sum(hours_of_production().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])


@component.add(
    name="HOURS OF PRODUCTION",
    units="Hours",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_flow": 1, "month_to_hours": 1},
)
def hours_of_production():
    return if_then_else(
        total_flow() < 2,
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: month_to_hours(),
    )


@component.add(
    name="RATIO WATER DEMAND INDUSTRY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_demand_from_industry": 1, "water_demand": 1},
)
def ratio_water_demand_industry():
    return water_demand_from_industry() / water_demand()


@component.add(
    name="RATIO WATER DEMAND URBAN",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_demand_from_urban": 1, "water_demand": 1},
)
def ratio_water_demand_urban():
    return water_demand_from_urban() / water_demand()


@component.add(
    name="RATIO WATER DEMAND AGRICULTURE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_demand_from_agriculture": 1, "water_demand": 1},
)
def ratio_water_demand_agriculture():
    return water_demand_from_agriculture() / water_demand()


@component.add(
    name="TOTAL SNOW ACCUMULATION",
    units="cm",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"snow_accumulation_monthly": 1},
)
def total_snow_accumulation():
    return sum(
        snow_accumulation_monthly().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]
    )


@component.add(
    name="FINAL YEAR PROMOTE DOMESTIC TOURISM",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_promote_domestic_tourism"},
)
def final_year_promote_domestic_tourism():
    return _ext_constant_final_year_promote_domestic_tourism()


_ext_constant_final_year_promote_domestic_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "FINAL_YEAR_PROMOTE_DOMESTIC_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_final_year_promote_domestic_tourism",
)


@component.add(
    name="ENERGY CONSUMPTION WATER TREATMENT",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"mwh_per_hm3_treated_water": 1, "water_treatment": 1},
)
def energy_consumption_water_treatment():
    return mwh_per_hm3_treated_water() * water_treatment()


@component.add(
    name="MWH PER HM3 DESALINATED WATER",
    units="MWh/Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_mwh_per_hm3_desalinated_water"},
)
def mwh_per_hm3_desalinated_water():
    return _ext_constant_mwh_per_hm3_desalinated_water()


_ext_constant_mwh_per_hm3_desalinated_water = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "MWH_PER_HM3_DESALINATED_WATER*",
    {},
    _root,
    {},
    "_ext_constant_mwh_per_hm3_desalinated_water",
)


@component.add(
    name="DAYS INTERNATIONAL VISITORS",
    units="Days",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_days_international_visitors"},
)
def days_international_visitors():
    return _ext_constant_days_international_visitors()


_ext_constant_days_international_visitors = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "DAYS_INTERNATIONAL_VISITORS*",
    {},
    _root,
    {},
    "_ext_constant_days_international_visitors",
)


@component.add(
    name="ADAPTIVE CAPACITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "investment_in_tourism_adaptation_and_resilience": 1,
        "promote_all_year_tourism": 1,
        "promote_domestic_tourism": 1,
    },
)
def adaptive_capacity():
    return (
        investment_in_tourism_adaptation_and_resilience()
        + promote_all_year_tourism()
        + promote_domestic_tourism()
    ) / 3


@component.add(
    name="DAYS NATIONAL VISITORS",
    units="Days",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_days_national_visitors"},
)
def days_national_visitors():
    return _ext_constant_days_national_visitors()


_ext_constant_days_national_visitors = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "DAYS_NATIONAL_VISITORS*",
    {},
    _root,
    {},
    "_ext_constant_days_national_visitors",
)


@component.add(
    name="INITIAL WATER TREATMENT",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_water_treatment"},
)
def initial_water_treatment():
    return _ext_constant_initial_water_treatment()


_ext_constant_initial_water_treatment = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_WATER_TREATMENT*",
    {},
    _root,
    {},
    "_ext_constant_initial_water_treatment",
)


@component.add(
    name="WATER DEMAND FROM URBAN",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_efficient_water_demand_from_urban": 3,
        "days_national_visitors": 4,
        "days_international_visitors": 4,
        "national_visitors": 4,
        "international_visitors": 4,
        "cs_population": 4,
        "water_demand_from_urban_per_capita": 4,
        "efficient_water_demand_from_urban_objective": 2,
        "time": 3,
        "initial_year_efficient_water_demand_from_urban": 3,
        "final_year_efficient_water_demand_from_urban": 2,
    },
)
def water_demand_from_urban():
    return if_then_else(
        switch_efficient_water_demand_from_urban() == 0,
        lambda: water_demand_from_urban_per_capita()
        * (
            cs_population()
            + sum(national_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"])
            * (days_national_visitors() / 365)
            + sum(
                international_visitors().rename({"SEASONS": "SEASONS!"}),
                dim=["SEASONS!"],
            )
            * (days_international_visitors() / 365)
        ),
        lambda: if_then_else(
            np.logical_and(
                switch_efficient_water_demand_from_urban() == 1,
                time() < initial_year_efficient_water_demand_from_urban(),
            ),
            lambda: water_demand_from_urban_per_capita()
            * (
                cs_population()
                + sum(
                    national_visitors().rename({"SEASONS": "SEASONS!"}),
                    dim=["SEASONS!"],
                )
                * (days_national_visitors() / 365)
                + sum(
                    international_visitors().rename({"SEASONS": "SEASONS!"}),
                    dim=["SEASONS!"],
                )
                * (days_international_visitors() / 365)
            ),
            lambda: if_then_else(
                np.logical_and(
                    switch_efficient_water_demand_from_urban() == 1,
                    time() > final_year_efficient_water_demand_from_urban(),
                ),
                lambda: (
                    (
                        cs_population()
                        + sum(
                            national_visitors().rename({"SEASONS": "SEASONS!"}),
                            dim=["SEASONS!"],
                        )
                        * (days_national_visitors() / 365)
                        + sum(
                            international_visitors().rename({"SEASONS": "SEASONS!"}),
                            dim=["SEASONS!"],
                        )
                        * (days_international_visitors() / 365)
                    )
                    * water_demand_from_urban_per_capita()
                )
                * (1 - efficient_water_demand_from_urban_objective()),
                lambda: (
                    (
                        cs_population()
                        + sum(
                            national_visitors().rename({"SEASONS": "SEASONS!"}),
                            dim=["SEASONS!"],
                        )
                        * (days_national_visitors() / 365)
                        + sum(
                            international_visitors().rename({"SEASONS": "SEASONS!"}),
                            dim=["SEASONS!"],
                        )
                        * (days_international_visitors() / 365)
                    )
                    * water_demand_from_urban_per_capita()
                )
                * (
                    1
                    - efficient_water_demand_from_urban_objective()
                    * (
                        (time() - initial_year_efficient_water_demand_from_urban())
                        / (
                            final_year_efficient_water_demand_from_urban()
                            - initial_year_efficient_water_demand_from_urban()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="SWITCH PROMOTE DOMESTIC TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_promote_domestic_tourism"},
)
def switch_promote_domestic_tourism():
    """
    0=OFF 1=ON
    """
    return _ext_constant_switch_promote_domestic_tourism()


_ext_constant_switch_promote_domestic_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "SWITCH_PROMOTE_DOMESTIC_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_switch_promote_domestic_tourism",
)


@component.add(
    name="FINAL YEAR WATER TREATMENT INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_water_treatment_increase"},
)
def final_year_water_treatment_increase():
    return _ext_constant_final_year_water_treatment_increase()


_ext_constant_final_year_water_treatment_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_WATER_TREATMENT_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_water_treatment_increase",
)


@component.add(
    name="INITIAL YEAR WATER TREATMENT INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_water_treatment_increase"},
)
def initial_year_water_treatment_increase():
    return _ext_constant_initial_year_water_treatment_increase()


_ext_constant_initial_year_water_treatment_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_WATER_TREATMENT_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_water_treatment_increase",
)


@component.add(
    name="EFFECTIVE PRECIPITATION",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature": 1, "precipitation": 1, "effective_snowmelt": 1},
)
def effective_precipitation():
    return (
        if_then_else(
            temperature() < 0,
            lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
            lambda: precipitation(),
        )
        + effective_snowmelt()
    )


@component.add(
    name="SWITCH WATER TREATMENT INCREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_water_treatment_increase"},
)
def switch_water_treatment_increase():
    return _ext_constant_switch_water_treatment_increase()


_ext_constant_switch_water_treatment_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_WATER_TREATMENT_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_switch_water_treatment_increase",
)


@component.add(
    name="WATER TREATMENT INCREASE OBJECTIVE",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_water_treatment_increase_objective"},
)
def water_treatment_increase_objective():
    return _ext_constant_water_treatment_increase_objective()


_ext_constant_water_treatment_increase_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "WATER_TREATMENT_INCREASE_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_water_treatment_increase_objective",
)


@component.add(
    name="INITIAL NATIONAL VISITORS AUTUMN",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_domestic_tourism": 3,
        "mean_tci_autumn": 4,
        "alfa_autumn": 4,
        "beta_autumn": 4,
        "initial_year_promote_domestic_tourism": 3,
        "final_year_promote_domestic_tourism": 2,
        "time": 3,
        "promote_domestic_tourism_objective": 2,
    },
)
def initial_national_visitors_autumn():
    return if_then_else(
        switch_promote_domestic_tourism() == 0,
        lambda: alfa_autumn() * mean_tci_autumn() + beta_autumn(),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_domestic_tourism() == 1,
                time() < initial_year_promote_domestic_tourism(),
            ),
            lambda: alfa_autumn() * mean_tci_autumn() + beta_autumn(),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_domestic_tourism() == 1,
                    time() > final_year_promote_domestic_tourism(),
                ),
                lambda: (alfa_autumn() * mean_tci_autumn() + beta_autumn())
                * (1 + float(promote_domestic_tourism_objective().loc["AUTUMN"])),
                lambda: (alfa_autumn() * mean_tci_autumn() + beta_autumn())
                * (
                    1
                    + float(promote_domestic_tourism_objective().loc["AUTUMN"])
                    * (
                        (time() - initial_year_promote_domestic_tourism())
                        / (
                            final_year_promote_domestic_tourism()
                            - initial_year_promote_domestic_tourism()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="INITIAL NATIONAL VISITORS SPRING",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_domestic_tourism": 3,
        "mean_tci_spring": 4,
        "alfa_spring": 4,
        "beta_spring": 4,
        "initial_year_promote_domestic_tourism": 3,
        "final_year_promote_domestic_tourism": 2,
        "time": 3,
        "promote_domestic_tourism_objective": 2,
    },
)
def initial_national_visitors_spring():
    return if_then_else(
        switch_promote_domestic_tourism() == 0,
        lambda: alfa_spring() * mean_tci_spring() + beta_spring(),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_domestic_tourism() == 1,
                time() < initial_year_promote_domestic_tourism(),
            ),
            lambda: alfa_spring() * mean_tci_spring() + beta_spring(),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_domestic_tourism() == 1,
                    time() > final_year_promote_domestic_tourism(),
                ),
                lambda: (alfa_spring() * mean_tci_spring() + beta_spring())
                * (1 + float(promote_domestic_tourism_objective().loc["SPRING"])),
                lambda: (alfa_spring() * mean_tci_spring() + beta_spring())
                * (
                    1
                    + float(promote_domestic_tourism_objective().loc["SPRING"])
                    * (
                        (time() - initial_year_promote_domestic_tourism())
                        / (
                            final_year_promote_domestic_tourism()
                            - initial_year_promote_domestic_tourism()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="INITIAL NATIONAL VISITORS SUMMER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_domestic_tourism": 3,
        "mean_tci_summer": 4,
        "alfa_summer": 4,
        "beta_summer": 4,
        "initial_year_promote_domestic_tourism": 3,
        "final_year_promote_domestic_tourism": 2,
        "time": 3,
        "promote_domestic_tourism_objective": 2,
    },
)
def initial_national_visitors_summer():
    return if_then_else(
        switch_promote_domestic_tourism() == 0,
        lambda: alfa_summer() * mean_tci_summer() + beta_summer(),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_domestic_tourism() == 1,
                time() < initial_year_promote_domestic_tourism(),
            ),
            lambda: alfa_summer() * mean_tci_summer() + beta_summer(),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_domestic_tourism() == 1,
                    time() > final_year_promote_domestic_tourism(),
                ),
                lambda: (alfa_summer() * mean_tci_summer() + beta_summer())
                * (1 + float(promote_domestic_tourism_objective().loc["SUMMER"])),
                lambda: (alfa_summer() * mean_tci_summer() + beta_summer())
                * (
                    1
                    + float(promote_domestic_tourism_objective().loc["SUMMER"])
                    * (
                        (time() - initial_year_promote_domestic_tourism())
                        / (
                            final_year_promote_domestic_tourism()
                            - initial_year_promote_domestic_tourism()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="REQUEST FROM GROUNDWATER EXTRACTION",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "water_demand": 2,
        "storage_outputs": 2,
        "desalination": 2,
        "transboundary_water": 2,
        "water_treatment": 2,
    },
)
def request_from_groundwater_extraction():
    """
    WATER DEMAND-STORAGE OUTPUTS-DESALINATION-SUM(TRANSBOUNDARY WATER[MONTHS!])-WATER TREATMENT
    """
    return if_then_else(
        water_demand()
        - storage_outputs()
        - desalination()
        - sum(transboundary_water().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        - water_treatment()
        < 0,
        lambda: 0,
        lambda: water_demand()
        - storage_outputs()
        - desalination()
        - sum(transboundary_water().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        - water_treatment(),
    )


@component.add(
    name="MWH PER HM3 TREATED WATER",
    units="MWh/Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_mwh_per_hm3_treated_water"},
)
def mwh_per_hm3_treated_water():
    return _ext_constant_mwh_per_hm3_treated_water()


_ext_constant_mwh_per_hm3_treated_water = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "MWH_PER_HM3_TREATED_WATER*",
    {},
    _root,
    {},
    "_ext_constant_mwh_per_hm3_treated_water",
)


@component.add(
    name="INITIAL YEAR PROMOTE DOMESTIC TOURISM",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_promote_domestic_tourism"},
)
def initial_year_promote_domestic_tourism():
    return _ext_constant_initial_year_promote_domestic_tourism()


_ext_constant_initial_year_promote_domestic_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "INITIAL_YEAR_PROMOTE_DOMESTIC_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_promote_domestic_tourism",
)


@component.add(
    name="PROMOTE DOMESTIC TOURISM OBJECTIVE",
    units="Dmnl",
    subscripts=["SEASONS"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_promote_domestic_tourism_objective"},
)
def promote_domestic_tourism_objective():
    return _ext_constant_promote_domestic_tourism_objective()


_ext_constant_promote_domestic_tourism_objective = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "PROMOTE_DOMESTIC_TOURISM_OBJECTIVE*",
    {"SEASONS": _subscript_dict["SEASONS"]},
    _root,
    {"SEASONS": _subscript_dict["SEASONS"]},
    "_ext_constant_promote_domestic_tourism_objective",
)


@component.add(
    name="PROMOTE DOMESTIC TOURISM",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"national_visitors": 1, "total_visitors": 1},
)
def promote_domestic_tourism():
    return float(
        np.minimum(
            1,
            sum(national_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"])
            / total_visitors(),
        )
    )


@component.add(
    name="DIRECT RUNOFF",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"effective_precipitation": 3, "pet": 3, "previous_moisture": 2},
)
def direct_runoff():
    """
    IF THEN ELSE( EFFECTIVE PRECIPITATION[MONTHS]>=PET[MONTHS] , EFFECTIVE PRECIPITATION[MONTHS]-PET[MONTHS]-MOISTURE CHANGE [MONTHS] , 0 )
    """
    return if_then_else(
        np.logical_and(
            effective_precipitation() > pet(),
            previous_moisture() + effective_precipitation() - pet() > 150,
        ),
        lambda: (previous_moisture() + effective_precipitation() - pet()) - 150,
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="SNOW ACCUMULATION MONTHLY",
    units="cm/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature": 1, "precipitation": 1},
)
def snow_accumulation_monthly():
    return if_then_else(
        temperature() < 0,
        lambda: precipitation(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="CUMULATIVE SNOWMELT CAPACITY",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"snowmelt": 78},
)
def cumulative_snowmelt_capacity():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        )
        == 5,
        lambda: xr.DataArray(
            float(snowmelt().loc["FEBRUARY"]),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 6,
            lambda: xr.DataArray(
                float(snowmelt().loc["FEBRUARY"]) + float(snowmelt().loc["MARCH"]),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 7,
                lambda: xr.DataArray(
                    float(snowmelt().loc["FEBRUARY"])
                    + float(snowmelt().loc["MARCH"])
                    + float(snowmelt().loc["APRIL"]),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                ),
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 8,
                    lambda: xr.DataArray(
                        float(snowmelt().loc["FEBRUARY"])
                        + float(snowmelt().loc["MARCH"])
                        + float(snowmelt().loc["APRIL"])
                        + float(snowmelt().loc["MAY"]),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    ),
                    lambda: if_then_else(
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        )
                        == 9,
                        lambda: xr.DataArray(
                            float(snowmelt().loc["FEBRUARY"])
                            + float(snowmelt().loc["MARCH"])
                            + float(snowmelt().loc["APRIL"])
                            + float(snowmelt().loc["MAY"])
                            + float(snowmelt().loc["JUNE"]),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        ),
                        lambda: if_then_else(
                            xr.DataArray(
                                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            )
                            == 10,
                            lambda: xr.DataArray(
                                float(snowmelt().loc["FEBRUARY"])
                                + float(snowmelt().loc["MARCH"])
                                + float(snowmelt().loc["APRIL"])
                                + float(snowmelt().loc["MAY"])
                                + float(snowmelt().loc["JUNE"])
                                + float(snowmelt().loc["JULY"]),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            ),
                            lambda: if_then_else(
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 11,
                                lambda: xr.DataArray(
                                    float(snowmelt().loc["FEBRUARY"])
                                    + float(snowmelt().loc["MARCH"])
                                    + float(snowmelt().loc["APRIL"])
                                    + float(snowmelt().loc["MAY"])
                                    + float(snowmelt().loc["JUNE"])
                                    + float(snowmelt().loc["JULY"])
                                    + float(snowmelt().loc["AUGUST"]),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                ),
                                lambda: if_then_else(
                                    xr.DataArray(
                                        np.arange(
                                            1, len(_subscript_dict["MONTHS"]) + 1
                                        ),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    )
                                    == 12,
                                    lambda: xr.DataArray(
                                        float(snowmelt().loc["FEBRUARY"])
                                        + float(snowmelt().loc["MARCH"])
                                        + float(snowmelt().loc["APRIL"])
                                        + float(snowmelt().loc["MAY"])
                                        + float(snowmelt().loc["JUNE"])
                                        + float(snowmelt().loc["JULY"])
                                        + float(snowmelt().loc["AUGUST"])
                                        + float(snowmelt().loc["SEPTEMBER"]),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    ),
                                    lambda: if_then_else(
                                        xr.DataArray(
                                            np.arange(
                                                1, len(_subscript_dict["MONTHS"]) + 1
                                            ),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        )
                                        == 1,
                                        lambda: xr.DataArray(
                                            float(snowmelt().loc["FEBRUARY"])
                                            + float(snowmelt().loc["MARCH"])
                                            + float(snowmelt().loc["APRIL"])
                                            + float(snowmelt().loc["MAY"])
                                            + float(snowmelt().loc["JUNE"])
                                            + float(snowmelt().loc["JULY"])
                                            + float(snowmelt().loc["AUGUST"])
                                            + float(snowmelt().loc["SEPTEMBER"])
                                            + float(snowmelt().loc["OCTOBER"]),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        ),
                                        lambda: if_then_else(
                                            xr.DataArray(
                                                np.arange(
                                                    1,
                                                    len(_subscript_dict["MONTHS"]) + 1,
                                                ),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            )
                                            == 2,
                                            lambda: xr.DataArray(
                                                float(snowmelt().loc["FEBRUARY"])
                                                + float(snowmelt().loc["MARCH"])
                                                + float(snowmelt().loc["APRIL"])
                                                + float(snowmelt().loc["MAY"])
                                                + float(snowmelt().loc["JUNE"])
                                                + float(snowmelt().loc["JULY"])
                                                + float(snowmelt().loc["AUGUST"])
                                                + float(snowmelt().loc["SEPTEMBER"])
                                                + float(snowmelt().loc["OCTOBER"])
                                                + float(snowmelt().loc["NOVEMBER"]),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            ),
                                            lambda: if_then_else(
                                                xr.DataArray(
                                                    np.arange(
                                                        1,
                                                        len(_subscript_dict["MONTHS"])
                                                        + 1,
                                                    ),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                )
                                                == 3,
                                                lambda: xr.DataArray(
                                                    float(snowmelt().loc["FEBRUARY"])
                                                    + float(snowmelt().loc["MARCH"])
                                                    + float(snowmelt().loc["APRIL"])
                                                    + float(snowmelt().loc["MAY"])
                                                    + float(snowmelt().loc["JUNE"])
                                                    + float(snowmelt().loc["JULY"])
                                                    + float(snowmelt().loc["AUGUST"])
                                                    + float(snowmelt().loc["SEPTEMBER"])
                                                    + float(snowmelt().loc["OCTOBER"])
                                                    + float(snowmelt().loc["NOVEMBER"])
                                                    + float(snowmelt().loc["DECEMBER"]),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                ),
                                                lambda: if_then_else(
                                                    xr.DataArray(
                                                        np.arange(
                                                            1,
                                                            len(
                                                                _subscript_dict[
                                                                    "MONTHS"
                                                                ]
                                                            )
                                                            + 1,
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    )
                                                    == 4,
                                                    lambda: xr.DataArray(
                                                        float(
                                                            snowmelt().loc["FEBRUARY"]
                                                        )
                                                        + float(snowmelt().loc["MARCH"])
                                                        + float(snowmelt().loc["APRIL"])
                                                        + float(snowmelt().loc["MAY"])
                                                        + float(snowmelt().loc["JUNE"])
                                                        + float(snowmelt().loc["JULY"])
                                                        + float(
                                                            snowmelt().loc["AUGUST"]
                                                        )
                                                        + float(
                                                            snowmelt().loc["SEPTEMBER"]
                                                        )
                                                        + float(
                                                            snowmelt().loc["OCTOBER"]
                                                        )
                                                        + float(
                                                            snowmelt().loc["NOVEMBER"]
                                                        )
                                                        + float(
                                                            snowmelt().loc["DECEMBER"]
                                                        )
                                                        + float(
                                                            snowmelt().loc["JANUARY"]
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                    lambda: xr.DataArray(
                                                        0,
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="SNOWMELT",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature": 2, "snowmelting_factor": 1, "month_days": 1},
)
def snowmelt():
    return if_then_else(
        temperature() > 0,
        lambda: snowmelting_factor() * temperature() * month_days(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="AET",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"effective_precipitation": 3, "pet": 3, "previous_moisture": 1},
)
def aet():
    return if_then_else(
        effective_precipitation() >= pet(),
        lambda: pet(),
        lambda: effective_precipitation()
        + np.minimum(previous_moisture(), pet() - effective_precipitation()),
    )


@component.add(
    name="SOIL MOISTURE",
    units="mm/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"initial_soil_moisture": 1, "effective_precipitation": 1, "pet": 1},
)
def soil_moisture():
    return initial_soil_moisture() + effective_precipitation() - pet()


@component.add(
    name="SOIL MOISTURE REAL",
    units="mm/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "effective_precipitation": 3,
        "pet": 3,
        "soil_water_capacity": 1,
        "previous_moisture": 2,
    },
)
def soil_moisture_real():
    return if_then_else(
        effective_precipitation() >= pet(),
        lambda: np.minimum(
            previous_moisture() + effective_precipitation() - pet(),
            soil_water_capacity(),
        ),
        lambda: np.maximum(
            previous_moisture() - (pet() - effective_precipitation()), 0
        ),
    )


@component.add(
    name="INFILTRATION",
    units="Hm3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "soil_moisture_real": 2,
        "soil_water_capacity": 1,
        "percolation_coefficient": 1,
        "total_cs_area": 1,
        "mm_per_km2_to_m3": 1,
        "m3_to_hm3": 1,
    },
)
def infiltration():
    return (
        if_then_else(
            soil_moisture_real() >= soil_water_capacity(),
            lambda: percolation_coefficient() * soil_moisture_real(),
            lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        )
        * total_cs_area()
        * mm_per_km2_to_m3()
    ) / m3_to_hm3()


@component.add(
    name="SNOW OUTPUT",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"snow_validation": 12},
)
def snow_output():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        )
        == 4,
        lambda: xr.DataArray(
            float(snow_validation().loc["DECEMBER"]),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 5,
            lambda: xr.DataArray(
                float(snow_validation().loc["JANUARY"]),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 6,
                lambda: xr.DataArray(
                    float(snow_validation().loc["FEBRUARY"]),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                ),
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 7,
                    lambda: xr.DataArray(
                        float(snow_validation().loc["MARCH"]),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    ),
                    lambda: if_then_else(
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        )
                        == 8,
                        lambda: xr.DataArray(
                            float(snow_validation().loc["APRIL"]),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        ),
                        lambda: if_then_else(
                            xr.DataArray(
                                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            )
                            == 9,
                            lambda: xr.DataArray(
                                float(snow_validation().loc["MAY"]),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            ),
                            lambda: if_then_else(
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 10,
                                lambda: xr.DataArray(
                                    float(snow_validation().loc["JUNE"]),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                ),
                                lambda: if_then_else(
                                    xr.DataArray(
                                        np.arange(
                                            1, len(_subscript_dict["MONTHS"]) + 1
                                        ),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    )
                                    == 11,
                                    lambda: xr.DataArray(
                                        float(snow_validation().loc["JULY"]),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    ),
                                    lambda: if_then_else(
                                        xr.DataArray(
                                            np.arange(
                                                1, len(_subscript_dict["MONTHS"]) + 1
                                            ),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        )
                                        == 12,
                                        lambda: xr.DataArray(
                                            float(snow_validation().loc["AUGUST"]),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        ),
                                        lambda: if_then_else(
                                            xr.DataArray(
                                                np.arange(
                                                    1,
                                                    len(_subscript_dict["MONTHS"]) + 1,
                                                ),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            )
                                            == 1,
                                            lambda: xr.DataArray(
                                                float(
                                                    snow_validation().loc["SEPTEMBER"]
                                                ),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            ),
                                            lambda: if_then_else(
                                                xr.DataArray(
                                                    np.arange(
                                                        1,
                                                        len(_subscript_dict["MONTHS"])
                                                        + 1,
                                                    ),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                )
                                                == 2,
                                                lambda: xr.DataArray(
                                                    float(
                                                        snow_validation().loc["OCTOBER"]
                                                    ),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                ),
                                                lambda: if_then_else(
                                                    xr.DataArray(
                                                        np.arange(
                                                            1,
                                                            len(
                                                                _subscript_dict[
                                                                    "MONTHS"
                                                                ]
                                                            )
                                                            + 1,
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    )
                                                    == 3,
                                                    lambda: xr.DataArray(
                                                        float(
                                                            snow_validation().loc[
                                                                "NOVEMBER"
                                                            ]
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                    lambda: xr.DataArray(
                                                        0,
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="PREVIOUS MOISTURE",
    units="mm/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"soil_moisture": 12},
)
def previous_moisture():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        )
        == 4,
        lambda: xr.DataArray(
            float(soil_moisture().loc["DECEMBER"]),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 5,
            lambda: xr.DataArray(
                float(soil_moisture().loc["JANUARY"]),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 6,
                lambda: xr.DataArray(
                    float(soil_moisture().loc["FEBRUARY"]),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                ),
                lambda: if_then_else(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 7,
                    lambda: xr.DataArray(
                        float(soil_moisture().loc["MARCH"]),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    ),
                    lambda: if_then_else(
                        xr.DataArray(
                            np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        )
                        == 8,
                        lambda: xr.DataArray(
                            float(soil_moisture().loc["APRIL"]),
                            {"MONTHS": _subscript_dict["MONTHS"]},
                            ["MONTHS"],
                        ),
                        lambda: if_then_else(
                            xr.DataArray(
                                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            )
                            == 9,
                            lambda: xr.DataArray(
                                float(soil_moisture().loc["MAY"]),
                                {"MONTHS": _subscript_dict["MONTHS"]},
                                ["MONTHS"],
                            ),
                            lambda: if_then_else(
                                xr.DataArray(
                                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                )
                                == 10,
                                lambda: xr.DataArray(
                                    float(soil_moisture().loc["JUNE"]),
                                    {"MONTHS": _subscript_dict["MONTHS"]},
                                    ["MONTHS"],
                                ),
                                lambda: if_then_else(
                                    xr.DataArray(
                                        np.arange(
                                            1, len(_subscript_dict["MONTHS"]) + 1
                                        ),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    )
                                    == 11,
                                    lambda: xr.DataArray(
                                        float(soil_moisture().loc["JULY"]),
                                        {"MONTHS": _subscript_dict["MONTHS"]},
                                        ["MONTHS"],
                                    ),
                                    lambda: if_then_else(
                                        xr.DataArray(
                                            np.arange(
                                                1, len(_subscript_dict["MONTHS"]) + 1
                                            ),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        )
                                        == 12,
                                        lambda: xr.DataArray(
                                            float(soil_moisture().loc["AUGUST"]),
                                            {"MONTHS": _subscript_dict["MONTHS"]},
                                            ["MONTHS"],
                                        ),
                                        lambda: if_then_else(
                                            xr.DataArray(
                                                np.arange(
                                                    1,
                                                    len(_subscript_dict["MONTHS"]) + 1,
                                                ),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            )
                                            == 1,
                                            lambda: xr.DataArray(
                                                float(soil_moisture().loc["SEPTEMBER"]),
                                                {"MONTHS": _subscript_dict["MONTHS"]},
                                                ["MONTHS"],
                                            ),
                                            lambda: if_then_else(
                                                xr.DataArray(
                                                    np.arange(
                                                        1,
                                                        len(_subscript_dict["MONTHS"])
                                                        + 1,
                                                    ),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                )
                                                == 2,
                                                lambda: xr.DataArray(
                                                    float(
                                                        soil_moisture().loc["OCTOBER"]
                                                    ),
                                                    {
                                                        "MONTHS": _subscript_dict[
                                                            "MONTHS"
                                                        ]
                                                    },
                                                    ["MONTHS"],
                                                ),
                                                lambda: if_then_else(
                                                    xr.DataArray(
                                                        np.arange(
                                                            1,
                                                            len(
                                                                _subscript_dict[
                                                                    "MONTHS"
                                                                ]
                                                            )
                                                            + 1,
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    )
                                                    == 3,
                                                    lambda: xr.DataArray(
                                                        float(
                                                            soil_moisture().loc[
                                                                "NOVEMBER"
                                                            ]
                                                        ),
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                    lambda: xr.DataArray(
                                                        0,
                                                        {
                                                            "MONTHS": _subscript_dict[
                                                                "MONTHS"
                                                            ]
                                                        },
                                                        ["MONTHS"],
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="PERCOLATION RUNOFF",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "soil_moisture_real": 2,
        "soil_water_capacity": 1,
        "percolation_coefficient": 1,
    },
)
def percolation_runoff():
    return (
        if_then_else(
            soil_moisture_real() >= soil_water_capacity(),
            lambda: percolation_coefficient() * soil_moisture_real(),
            lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        )
        * 0.1
    )


@component.add(
    name="INITIAL SOIL MOISTURE",
    units="mm/Month",
    subscripts=["MONTHS"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_soil_moisture"},
)
def initial_soil_moisture():
    return _ext_constant_initial_soil_moisture()


_ext_constant_initial_soil_moisture = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_SOIL_MOISTURE*",
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_constant_initial_soil_moisture",
)


@component.add(
    name="TOTAL INTERNATIONAL VISITORS",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"international_visitors": 1},
)
def total_international_visitors():
    return sum(
        international_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"]
    )


@component.add(
    name="TOTAL NATIONAL VISITORS",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"national_visitors": 1},
)
def total_national_visitors():
    return sum(national_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"])


@component.add(
    name="DEVIATION TOURISTS",
    units="Dmnl",
    subscripts=["SEASONS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "international_visitors": 1,
        "national_visitors": 1,
        "total_visitors": 1,
    },
)
def deviation_tourists():
    return np.abs(
        (international_visitors() + national_visitors()) / total_visitors() - 1 / 4
    )


@component.add(
    name="TOTAL VISITORS",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"national_visitors": 1, "international_visitors": 1},
)
def total_visitors():
    return sum(
        national_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"]
    ) + sum(international_visitors().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"])


@component.add(
    name="INTERNATIONAL VISITORS",
    units="Tourists",
    subscripts=["SEASONS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "international_visitors_spring": 1,
        "international_visitors_autumn": 1,
        "international_visitors_winter": 1,
        "international_visitors_summer": 1,
    },
)
def international_visitors():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        )
        == 1,
        lambda: xr.DataArray(
            international_visitors_spring(),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            )
            == 2,
            lambda: xr.DataArray(
                international_visitors_summer(),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                )
                == 3,
                lambda: xr.DataArray(
                    international_visitors_autumn(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
                lambda: xr.DataArray(
                    international_visitors_winter(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
            ),
        ),
    )


@component.add(
    name="NATIONAL VISITORS",
    units="Tourists",
    subscripts=["SEASONS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "national_visitors_spring": 1,
        "national_visitors_summer": 1,
        "national_visitors_winter": 1,
        "national_visitors_autumn": 1,
    },
)
def national_visitors():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        )
        == 1,
        lambda: xr.DataArray(
            national_visitors_spring(),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            )
            == 2,
            lambda: xr.DataArray(
                national_visitors_summer(),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                )
                == 3,
                lambda: xr.DataArray(
                    national_visitors_autumn(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
                lambda: xr.DataArray(
                    national_visitors_winter(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
            ),
        ),
    )


@component.add(
    name="BETA AUTUMN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_autumn"},
)
def beta_autumn():
    return _ext_constant_beta_autumn()


_ext_constant_beta_autumn = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_AUTUMN*",
    {},
    _root,
    {},
    "_ext_constant_beta_autumn",
)


@component.add(
    name="TOURISM INCOME",
    units="Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "risk_index_tourism": 2,
        "average_expenditure_increase": 6,
        "international_expenditure": 3,
        "total_national_visitors": 3,
        "total_international_visitors": 3,
        "national_expenditure": 3,
    },
)
def tourism_income():
    return if_then_else(
        risk_index_tourism() > 0.7,
        lambda: 0.02
        * (
            total_international_visitors()
            * (international_expenditure() * (1 * average_expenditure_increase()))
        )
        + total_national_visitors()
        * (national_expenditure() * (1 + average_expenditure_increase())),
        lambda: if_then_else(
            risk_index_tourism() < 0.1,
            lambda: 0.1
            * (
                total_international_visitors()
                * (international_expenditure() * (1 * average_expenditure_increase()))
            )
            + total_national_visitors()
            * (national_expenditure() * (1 + average_expenditure_increase())),
            lambda: 0.05
            * (
                total_international_visitors()
                * (international_expenditure() * (1 * average_expenditure_increase()))
            )
            + total_national_visitors()
            * (national_expenditure() * (1 + average_expenditure_increase())),
        ),
    )


@component.add(
    name="ALFA 1 AUTUMN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_1_autumn"},
)
def alfa_1_autumn():
    return _ext_constant_alfa_1_autumn()


_ext_constant_alfa_1_autumn = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_1_AUTUMN*",
    {},
    _root,
    {},
    "_ext_constant_alfa_1_autumn",
)


@component.add(
    name="ALFA 1 SPRING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_1_spring"},
)
def alfa_1_spring():
    return _ext_constant_alfa_1_spring()


_ext_constant_alfa_1_spring = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_1_SPRING*",
    {},
    _root,
    {},
    "_ext_constant_alfa_1_spring",
)


@component.add(
    name="NATIONAL VISITORS AUTUMN",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_national_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def national_visitors_autumn():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_national_visitors().loc["AUTUMN"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_national_visitors().loc["AUTUMN"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_national_visitors().loc["AUTUMN"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_national_visitors().loc["AUTUMN"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="NATIONAL VISITORS SPRING",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_national_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def national_visitors_spring():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_national_visitors().loc["SPRING"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_national_visitors().loc["SPRING"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_national_visitors().loc["SPRING"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_national_visitors().loc["SPRING"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="NATIONAL VISITORS SUMMER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_national_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def national_visitors_summer():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_national_visitors().loc["SUMMER"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_national_visitors().loc["SUMMER"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_national_visitors().loc["SUMMER"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_national_visitors().loc["SUMMER"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_national_visitors().rename({"SEASONS": "SEASONS!"}),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="FINAL YEAR PROMOTE ALL YEAR TOURISM",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_promote_all_year_tourism"},
)
def final_year_promote_all_year_tourism():
    return _ext_constant_final_year_promote_all_year_tourism()


_ext_constant_final_year_promote_all_year_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "FINAL_YEAR_PROMOTE_ALL_YEAR_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_final_year_promote_all_year_tourism",
)


@component.add(
    name="BETA WINTER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_winter"},
)
def beta_winter():
    return _ext_constant_beta_winter()


_ext_constant_beta_winter = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_WINTER*",
    {},
    _root,
    {},
    "_ext_constant_beta_winter",
)


@component.add(
    name="ALFA AUTUMN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_autumn"},
)
def alfa_autumn():
    return _ext_constant_alfa_autumn()


_ext_constant_alfa_autumn = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_AUTUMN*",
    {},
    _root,
    {},
    "_ext_constant_alfa_autumn",
)


@component.add(
    name="INITIAL INTERNATIONAL VISITORS SPRING",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"alfa_1_spring": 1, "mean_tci_spring": 1, "beta_1_spring": 1},
)
def initial_international_visitors_spring():
    return alfa_1_spring() * mean_tci_spring() + beta_1_spring()


@component.add(
    name="INITIAL INTERNATIONAL VISITORS SUMMER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"alfa_1_summer": 1, "mean_tci_summer": 1, "beta_1_summer": 1},
)
def initial_international_visitors_summer():
    return alfa_1_summer() * mean_tci_summer() + beta_1_summer()


@component.add(
    name="INITIAL NATIONAL VISITORS",
    units="Tourists",
    subscripts=["SEASONS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "initial_national_visitors_spring": 1,
        "initial_national_visitors_winter": 1,
        "initial_national_visitors_autumn": 1,
        "initial_national_visitors_summer": 1,
    },
)
def initial_national_visitors():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        )
        == 1,
        lambda: xr.DataArray(
            initial_national_visitors_spring(),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            )
            == 2,
            lambda: xr.DataArray(
                initial_national_visitors_summer(),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                )
                == 3,
                lambda: xr.DataArray(
                    initial_national_visitors_autumn(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
                lambda: xr.DataArray(
                    initial_national_visitors_winter(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
            ),
        ),
    )


@component.add(
    name="PROMOTE ALL YEAR TOURISM",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"deviation_tourists": 1},
)
def promote_all_year_tourism():
    """
    1-((ABS(((INTERNATIONAL VISITORS AUTUMN+NATIONAL VISITORS AUTUMN)/TOTAL VISITORS)-(1/4))+ABS(((INTERNATIONAL VISITORS SPRING +NATIONAL VISITORS SPRING )/TOTAL VISITORS)-(1/4))+ABS(((INTERNATIONAL VISITORS SUMMER+NATIONAL VISITORS SUMMER)/TOTAL VISITORS)-(1/4))+ABS(((INTERNATIONAL VISITORS WINTER +NATIONAL VISITORS WINTER )/TOTAL VISITORS)-(1/4)))/(2*(1-(1/4))))
    """
    return 1 - sum(
        deviation_tourists().rename({"SEASONS": "SEASONS!"}), dim=["SEASONS!"]
    ) / (2 * (1 - 1 / 4))


@component.add(
    name="ALFA SPRING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_spring"},
)
def alfa_spring():
    return _ext_constant_alfa_spring()


_ext_constant_alfa_spring = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_SPRING*",
    {},
    _root,
    {},
    "_ext_constant_alfa_spring",
)


@component.add(
    name="ALFA SUMMER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_summer"},
)
def alfa_summer():
    return _ext_constant_alfa_summer()


_ext_constant_alfa_summer = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_SUMMER*",
    {},
    _root,
    {},
    "_ext_constant_alfa_summer",
)


@component.add(
    name="ALFA WINTER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_winter"},
)
def alfa_winter():
    return _ext_constant_alfa_winter()


_ext_constant_alfa_winter = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_WINTER*",
    {},
    _root,
    {},
    "_ext_constant_alfa_winter",
)


@component.add(
    name="INTERNATIONAL VISITORS AUTUMN",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_international_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def international_visitors_autumn():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_international_visitors().loc["AUTUMN"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_international_visitors().loc["AUTUMN"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_international_visitors().loc["AUTUMN"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_international_visitors().loc["AUTUMN"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="INTERNATIONAL VISITORS SPRING",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_international_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def international_visitors_spring():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_international_visitors().loc["SPRING"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_international_visitors().loc["SPRING"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_international_visitors().loc["SPRING"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_international_visitors().loc["SPRING"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="ALFA 1 SUMMER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_1_summer"},
)
def alfa_1_summer():
    return _ext_constant_alfa_1_summer()


_ext_constant_alfa_1_summer = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_1_SUMMER*",
    {},
    _root,
    {},
    "_ext_constant_alfa_1_summer",
)


@component.add(
    name="ALFA 1 WINTER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_1_winter"},
)
def alfa_1_winter():
    return _ext_constant_alfa_1_winter()


_ext_constant_alfa_1_winter = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "ALFA_1_WINTER*",
    {},
    _root,
    {},
    "_ext_constant_alfa_1_winter",
)


@component.add(
    name="BETA SPRING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_spring"},
)
def beta_spring():
    return _ext_constant_beta_spring()


_ext_constant_beta_spring = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_SPRING*",
    {},
    _root,
    {},
    "_ext_constant_beta_spring",
)


@component.add(
    name="PROMOTE ALL YEAR TOURISM OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_promote_all_year_tourism_objective"},
)
def promote_all_year_tourism_objective():
    return _ext_constant_promote_all_year_tourism_objective()


_ext_constant_promote_all_year_tourism_objective = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "PROMOTE_ALL_YEAR_TOURISM_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_promote_all_year_tourism_objective",
)


@component.add(
    name="BETA SUMMER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_summer"},
)
def beta_summer():
    return _ext_constant_beta_summer()


_ext_constant_beta_summer = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_SUMMER*",
    {},
    _root,
    {},
    "_ext_constant_beta_summer",
)


@component.add(
    name="RISK INDEX TOURISM",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tourism_exposure_and_probabilities": 3, "vulnerability": 3},
)
def risk_index_tourism():
    return if_then_else(
        tourism_exposure_and_probabilities() * vulnerability() < 0,
        lambda: 0,
        lambda: if_then_else(
            tourism_exposure_and_probabilities() * vulnerability() > 1,
            lambda: 1,
            lambda: tourism_exposure_and_probabilities() * vulnerability(),
        ),
    )


@component.add(
    name="BETA 1 AUTUMN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_1_autumn"},
)
def beta_1_autumn():
    return _ext_constant_beta_1_autumn()


_ext_constant_beta_1_autumn = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_1_AUTUMN*",
    {},
    _root,
    {},
    "_ext_constant_beta_1_autumn",
)


@component.add(
    name="INITIAL YEAR PROMOTE ALL YEAR TOURISM",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_promote_all_year_tourism"},
)
def initial_year_promote_all_year_tourism():
    return _ext_constant_initial_year_promote_all_year_tourism()


_ext_constant_initial_year_promote_all_year_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "INITIAL_YEAR_PROMOTE_ALL_YEAR_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_promote_all_year_tourism",
)


@component.add(
    name="BETA 1 SPRING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_1_spring"},
)
def beta_1_spring():
    return _ext_constant_beta_1_spring()


_ext_constant_beta_1_spring = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_1_SPRING*",
    {},
    _root,
    {},
    "_ext_constant_beta_1_spring",
)


@component.add(
    name="INTERNATIONAL VISITORS SUMMER",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_promote_all_year_tourism": 3,
        "initial_international_visitors": 6,
        "final_year_promote_all_year_tourism": 3,
        "time": 4,
        "promote_all_year_tourism_objective": 4,
        "initial_year_promote_all_year_tourism": 5,
    },
)
def international_visitors_summer():
    return if_then_else(
        switch_promote_all_year_tourism() == 0,
        lambda: float(initial_international_visitors().loc["SUMMER"]),
        lambda: if_then_else(
            np.logical_and(
                switch_promote_all_year_tourism() == 1,
                time() < initial_year_promote_all_year_tourism(),
            ),
            lambda: float(initial_international_visitors().loc["SUMMER"]),
            lambda: if_then_else(
                np.logical_and(
                    switch_promote_all_year_tourism() == 1,
                    time() > final_year_promote_all_year_tourism(),
                ),
                lambda: float(initial_international_visitors().loc["SUMMER"])
                * (1 - promote_all_year_tourism_objective())
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective(),
                lambda: float(initial_international_visitors().loc["SUMMER"])
                * (
                    1
                    - promote_all_year_tourism_objective()
                    * (
                        (time() - initial_year_promote_all_year_tourism())
                        / (
                            final_year_promote_all_year_tourism()
                            - initial_year_promote_all_year_tourism()
                        )
                    )
                )
                + (
                    sum(
                        initial_international_visitors().rename(
                            {"SEASONS": "SEASONS!"}
                        ),
                        dim=["SEASONS!"],
                    )
                    / 4
                )
                * promote_all_year_tourism_objective()
                * (
                    (time() - initial_year_promote_all_year_tourism())
                    / (
                        final_year_promote_all_year_tourism()
                        - initial_year_promote_all_year_tourism()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="SWITCH PROMOTE ALL YEAR TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_promote_all_year_tourism"},
)
def switch_promote_all_year_tourism():
    """
    0=ON 1=OFF
    """
    return _ext_constant_switch_promote_all_year_tourism()


_ext_constant_switch_promote_all_year_tourism = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "SWITCH_PROMOTE_ALL_YEAR_TOURISM*",
    {},
    _root,
    {},
    "_ext_constant_switch_promote_all_year_tourism",
)


@component.add(
    name="INITIAL INTERNATIONAL VISITORS",
    units="Tourists",
    subscripts=["SEASONS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "initial_international_visitors_spring": 1,
        "initial_international_visitors_summer": 1,
        "initial_international_visitors_winter": 1,
        "initial_international_visitors_autumn": 1,
    },
)
def initial_international_visitors():
    return if_then_else(
        xr.DataArray(
            np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        )
        == 1,
        lambda: xr.DataArray(
            initial_international_visitors_spring(),
            {"SEASONS": _subscript_dict["SEASONS"]},
            ["SEASONS"],
        ),
        lambda: if_then_else(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            )
            == 2,
            lambda: xr.DataArray(
                initial_international_visitors_summer(),
                {"SEASONS": _subscript_dict["SEASONS"]},
                ["SEASONS"],
            ),
            lambda: if_then_else(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["SEASONS"]) + 1),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                )
                == 3,
                lambda: xr.DataArray(
                    initial_international_visitors_autumn(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
                lambda: xr.DataArray(
                    initial_international_visitors_winter(),
                    {"SEASONS": _subscript_dict["SEASONS"]},
                    ["SEASONS"],
                ),
            ),
        ),
    )


@component.add(
    name="BETA 1 WINTER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_1_winter"},
)
def beta_1_winter():
    return _ext_constant_beta_1_winter()


_ext_constant_beta_1_winter = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_1_WINTER*",
    {},
    _root,
    {},
    "_ext_constant_beta_1_winter",
)


@component.add(
    name="BETA 1 SUMMER",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_1_summer"},
)
def beta_1_summer():
    return _ext_constant_beta_1_summer()


_ext_constant_beta_1_summer = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BETA_1_SUMMER*",
    {},
    _root,
    {},
    "_ext_constant_beta_1_summer",
)


@component.add(
    name="INITIAL INTERNATIONAL VISITORS AUTUMN",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"alfa_1_autumn": 1, "mean_tci_autumn": 1, "beta_1_autumn": 1},
)
def initial_international_visitors_autumn():
    return alfa_1_autumn() * mean_tci_autumn() + beta_1_autumn()


@component.add(
    name="TCI WINTER",
    units="Dmnl",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_index_ssp585": 1},
)
def tci_winter():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 3,
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 4,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 5,
            ),
        ),
        lambda: tci_index_ssp585(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="TCI SPRING",
    units="Dmnl",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_index_ssp585": 1},
)
def tci_spring():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 6,
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 7,
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 8,
            ),
        ),
        lambda: tci_index_ssp585(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="MEAN TCI SUMMER",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_summer": 2},
)
def mean_tci_summer():
    return sum(tci_summer().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]) / sum(
        if_then_else(
            tci_summer().rename({"MONTHS": "MONTHS!"}) != 0,
            lambda: xr.DataArray(
                1,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
            lambda: xr.DataArray(
                0,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
        ),
        dim=["MONTHS!"],
    )


@component.add(
    name="MEAN TCI WINTER",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_winter": 2},
)
def mean_tci_winter():
    return sum(tci_winter().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]) / sum(
        if_then_else(
            tci_winter().rename({"MONTHS": "MONTHS!"}) != 0,
            lambda: xr.DataArray(
                1,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
            lambda: xr.DataArray(
                0,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
        ),
        dim=["MONTHS!"],
    )


@component.add(
    name="MEAN TCI",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "mean_tci_autumn": 1,
        "mean_tci_spring": 1,
        "mean_tci_summer": 1,
        "mean_tci_winter": 1,
    },
)
def mean_tci():
    return (
        mean_tci_autumn() + mean_tci_spring() + mean_tci_summer() + mean_tci_winter()
    ) / 4


@component.add(
    name="TCI SUMMER",
    units="Dmnl",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_index_ssp585": 1},
)
def tci_summer():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 9,
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 10,
                np.logical_or(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 11,
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 12,
                ),
            ),
        ),
        lambda: tci_index_ssp585(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="MEAN TCI AUTUMN",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_autumn": 2},
)
def mean_tci_autumn():
    return sum(tci_autumn().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]) / sum(
        if_then_else(
            tci_autumn().rename({"MONTHS": "MONTHS!"}) != 0,
            lambda: xr.DataArray(
                1,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
            lambda: xr.DataArray(
                0,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
        ),
        dim=["MONTHS!"],
    )


@component.add(
    name="MEAN TCI SPRING",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_spring": 2},
)
def mean_tci_spring():
    return sum(tci_spring().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"]) / sum(
        if_then_else(
            tci_spring().rename({"MONTHS": "MONTHS!"}) != 0,
            lambda: xr.DataArray(
                1,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
            lambda: xr.DataArray(
                0,
                {
                    "MONTHS!": [
                        "OCTOBER",
                        "NOVEMBER",
                        "DECEMBER",
                        "JANUARY",
                        "FEBRUARY",
                        "MARCH",
                        "APRIL",
                        "MAY",
                        "JUNE",
                        "JULY",
                        "AUGUST",
                        "SEPTEMBER",
                    ]
                },
                ["MONTHS!"],
            ),
        ),
        dim=["MONTHS!"],
    )


@component.add(
    name="TCI AUTUMN",
    units="Dmnl",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tci_index_ssp585": 1},
)
def tci_autumn():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 1,
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 2,
        ),
        lambda: tci_index_ssp585(),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="NEW WIND CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_wind_capacity": 2,
        "time": 2,
        "max_wind_capacity": 1,
        "final_year_wind_policy_capacity": 2,
        "wind_capacity": 1,
        "wind_capacity_objective": 1,
        "initial_year_wind_policy_capacity": 2,
    },
)
def new_wind_capacity():
    """
    IF THEN ELSE(SWITCH WIND CAPACITY=0,MIN(MARKET DRIVEN INVESTMENTS FOR WIND CAPACITY,MAX WIND CAPACITY-WIND CAPACITY), IF THEN ELSE(SWITCH WIND CAPACITY=1:AND:(Time<INITIAL YEAR WIND POLICY CAPACITY:OR:Time>FINAL YEAR WIND POLICY CAPACITY ),MIN(MARKET DRIVEN INVESTMENTS FOR WIND CAPACITY,MAX WIND CAPACITY-WIND CAPACITY), IF THEN ELSE( WIND CAPACITY < MAX WIND CAPACITY, IF THEN ELSE(((WIND CAPACITY OBJECTIVE)/(FINAL YEAR WIND POLICY CAPACITY - INITIAL YEAR WIND POLICY CAPACITY))>MARKET DRIVEN INVESTMENTS FOR WIND CAPACITY,MIN((WIND CAPACITY OBJECTIVE)/(FINAL YEAR WIND POLICY CAPACITY - INITIAL YEAR WIND POLICY CAPACITY), MAX WIND CAPACITY-WIND CAPACITY),MIN(MARKET DRIVEN INVESTMENTS FOR WIND CAPACITY, MAX WIND CAPACITY-WIND CAPACITY)), 0 ) ) )
    """
    return if_then_else(
        switch_wind_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_wind_capacity() == 1,
                np.logical_or(
                    time() < initial_year_wind_policy_capacity(),
                    time() > final_year_wind_policy_capacity(),
                ),
            ),
            lambda: 0,
            lambda: float(
                np.minimum(
                    wind_capacity_objective()
                    / (
                        final_year_wind_policy_capacity()
                        - initial_year_wind_policy_capacity()
                    ),
                    max_wind_capacity() - wind_capacity(),
                )
            ),
        ),
    )


@component.add(
    name="NEW ROOFTOP CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_rooftop_capacity": 2,
        "time": 2,
        "rooftop_capacity_objective": 1,
        "rooftop_capacity": 1,
        "initial_year_rooftop_policy_capacity": 2,
        "final_year_rooftop_policy_capacity": 2,
        "max_rooftop_capacity": 1,
    },
)
def new_rooftop_capacity():
    """
    IF THEN ELSE(SWITCH ROOFTOP CAPACITY=0,MIN(MARKET DRIVEN INVESTMENTS FOR ROOFTOP CAPACITY,MAX ROOFTOP CAPACITY-ROOFTOP CAPACITY ), IF THEN ELSE(SWITCH ROOFTOP CAPACITY=1:AND:(Time<INITIAL YEAR ROOFTOP POLICY CAPACITY:OR:Time>FINAL YEAR ROOFTOP POLICY CAPACITY ),MIN(MARKET DRIVEN INVESTMENTS FOR ROOFTOP CAPACITY,MAX ROOFTOP CAPACITY-ROOFTOP CAPACITY), IF THEN ELSE( ROOFTOP CAPACITY < MAX ROOFTOP CAPACITY, IF THEN ELSE(((ROOFTOP CAPACITY OBJECTIVE)/(FINAL YEAR ROOFTOP POLICY CAPACITY - INITIAL YEAR ROOFTOP POLICY CAPACITY))>MARKET DRIVEN INVESTMENTS FOR ROOFTOP CAPACITY,MIN((ROOFTOP CAPACITY OBJECTIVE )/(FINAL YEAR ROOFTOP POLICY CAPACITY - INITIAL YEAR ROOFTOP POLICY CAPACITY), MAX ROOFTOP CAPACITY-ROOFTOP CAPACITY),MIN (MARKET DRIVEN INVESTMENTS FOR ROOFTOP CAPACITY,MAX ROOFTOP CAPACITY-ROOFTOP CAPACITY)), 0 ) ) )
    """
    return if_then_else(
        switch_rooftop_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_rooftop_capacity() == 1,
                np.logical_or(
                    time() < initial_year_rooftop_policy_capacity(),
                    time() > final_year_rooftop_policy_capacity(),
                ),
            ),
            lambda: 0,
            lambda: float(
                np.minimum(
                    rooftop_capacity_objective()
                    / (
                        final_year_rooftop_policy_capacity()
                        - initial_year_rooftop_policy_capacity()
                    ),
                    max_rooftop_capacity() - rooftop_capacity(),
                )
            ),
        ),
    )


@component.add(
    name="NEW HYDRO CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_hydro_capacity": 2,
        "time": 2,
        "hydropower_capacity": 1,
        "max_hydro_capacity": 1,
        "final_year_hydro_policy_capacity": 2,
        "initial_year_hydro_policy_capacity": 2,
        "hydro_capacity_objective": 1,
    },
)
def new_hydro_capacity():
    """
    IF THEN ELSE(SWITCH HYDRO CAPACITY=0,MIN(MARKET DRIVEN INVESTMENTS FOR HYDRO CAPACITY,MAX HYDRO CAPACITY-HYDROPOWER CAPACITY ), IF THEN ELSE(SWITCH HYDRO CAPACITY=1:AND:(Time<INITIAL YEAR HYDRO POLICY CAPACITY:OR:Time>FINAL YEAR HYDRO POLICY CAPACITY ),MIN(MARKET DRIVEN INVESTMENTS FOR HYDRO CAPACITY,MAX HYDRO CAPACITY-HYDROPOWER CAPACITY), IF THEN ELSE( HYDROPOWER CAPACITY < MAX HYDRO CAPACITY, IF THEN ELSE(((HYDRO CAPACITY OBJECTIVE)/(FINAL YEAR HYDRO POLICY CAPACITY - INITIAL YEAR HYDRO POLICY CAPACITY))>MARKET DRIVEN INVESTMENTS FOR HYDRO CAPACITY,MIN((HYDRO CAPACITY OBJECTIVE)/(FINAL YEAR HYDRO POLICY CAPACITY - INITIAL YEAR HYDRO POLICY CAPACITY), MAX HYDRO CAPACITY-HYDROPOWER CAPACITY),MIN(MARKET DRIVEN INVESTMENTS FOR HYDRO CAPACITY ,MAX HYDRO CAPACITY-HYDROPOWER CAPACITY)), 0 ) ) )
    """
    return if_then_else(
        switch_hydro_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_hydro_capacity() == 1,
                np.logical_or(
                    time() < initial_year_hydro_policy_capacity(),
                    time() > final_year_hydro_policy_capacity(),
                ),
            ),
            lambda: 0,
            lambda: float(
                np.minimum(
                    hydro_capacity_objective()
                    / (
                        final_year_hydro_policy_capacity()
                        - initial_year_hydro_policy_capacity()
                    ),
                    max_hydro_capacity() - hydropower_capacity(),
                )
            ),
        ),
    )


@component.add(
    name="SWITCH PROTECTED WETLANDS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_protected_wetlands"},
)
def switch_protected_wetlands():
    """
    1 = ON 0 = OFF GET DIRECT CONSTANTS ('Policy.xlsx', 'Land system', 'SWITCH_WETLANDS_PROTECTED*')
    """
    return _ext_constant_switch_protected_wetlands()


_ext_constant_switch_protected_wetlands = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SWITCH_WETLANDS_PROTECTED*",
    {},
    _root,
    {},
    "_ext_constant_switch_protected_wetlands",
)


@component.add(
    name="FINAL YEAR FOR PROTECTED WETLANDS",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_for_protected_wetlands"},
)
def final_year_for_protected_wetlands():
    return _ext_constant_final_year_for_protected_wetlands()


_ext_constant_final_year_for_protected_wetlands = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "FINAL_YEAR_FOR_WETLANDS_PROTECTED*",
    {},
    _root,
    {},
    "_ext_constant_final_year_for_protected_wetlands",
)


@component.add(
    name="INITIAL YEAR FOR PROTECTED WETLANDS",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_for_protected_wetlands"},
)
def initial_year_for_protected_wetlands():
    return _ext_constant_initial_year_for_protected_wetlands()


_ext_constant_initial_year_for_protected_wetlands = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INITIAL_YEAR_FOR_WETLANDS_PROTECTED*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_for_protected_wetlands",
)


@component.add(
    name="INITIAL PROTECTED WETLANDS",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_protected_wetlands"},
)
def initial_protected_wetlands():
    return _ext_constant_initial_protected_wetlands()


_ext_constant_initial_protected_wetlands = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_PROTECTED_WETLANDS*",
    {},
    _root,
    {},
    "_ext_constant_initial_protected_wetlands",
)


@component.add(
    name="INCREASE OBJECTIVE FOR PROTECTED WETLANDS",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_increase_objective_for_protected_wetlands"
    },
)
def increase_objective_for_protected_wetlands():
    return _ext_constant_increase_objective_for_protected_wetlands()


_ext_constant_increase_objective_for_protected_wetlands = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INCREASE_OBJECTIVE_FOR_WETLANDS_PROTECTED*",
    {},
    _root,
    {},
    "_ext_constant_increase_objective_for_protected_wetlands",
)


@component.add(
    name='"WATER-CROP RATIO"',
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "irrigated_crop_area": 2,
        "groundwater_inputs": 1,
        "storage_inputs": 1,
        "storage_water": 1,
    },
)
def watercrop_ratio():
    """
    Deleted the multiplication factor of 365 applied to IRRIGATED CROP AREA
    """
    return if_then_else(
        irrigated_crop_area() == 0,
        lambda: 1,
        lambda: (storage_inputs() + storage_water() + groundwater_inputs())
        / irrigated_crop_area(),
    )


@component.add(
    name="AVAILABLE RUNOFF",
    units="Hm3/Month",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "direct_runoff": 1,
        "percolation_runoff": 1,
        "total_cs_area": 1,
        "mm_per_km2_to_m3": 1,
    },
)
def available_runoff():
    return (
        (direct_runoff() + percolation_runoff()) * total_cs_area()
    ) / mm_per_km2_to_m3()


@component.add(
    name="SWITCH TRANSBOUNDARY POLICY OTHER CS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_transboundary_policy_other_cs"},
)
def switch_transboundary_policy_other_cs():
    """
    ON = 1 OFF = 0
    """
    return _ext_constant_switch_transboundary_policy_other_cs()


_ext_constant_switch_transboundary_policy_other_cs = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_TRANSBOUNDARY_POLICY_OTHER_CS*",
    {},
    _root,
    {},
    "_ext_constant_switch_transboundary_policy_other_cs",
)


@component.add(
    name="TRANSBOUNDARY WATER",
    units="Hm3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_cs": 1,
        "transboundary_objective_murcia": 1,
        "time": 3,
        "final_year_transboundary_policy_other_cs": 2,
        "switch_transboundary_policy_other_cs": 2,
        "initial_year_transboundary_policy_other_cs": 3,
        "initial_transboundary_objective_other_cs_monthly": 1,
    },
)
def transboundary_water():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 2,
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 3,
                np.logical_or(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 4,
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 5,
                ),
            ),
        ),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: if_then_else(
            switch_cs() == 4,
            lambda: transboundary_objective_murcia(),
            lambda: if_then_else(
                switch_transboundary_policy_other_cs() == 0,
                lambda: xr.DataArray(
                    0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]
                ),
                lambda: if_then_else(
                    np.logical_and(
                        switch_transboundary_policy_other_cs() == 1,
                        np.logical_or(
                            time() < initial_year_transboundary_policy_other_cs(),
                            time() >= final_year_transboundary_policy_other_cs(),
                        ),
                    ),
                    lambda: xr.DataArray(
                        0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]
                    ),
                    lambda: initial_transboundary_objective_other_cs_monthly()
                    * (
                        (time() - initial_year_transboundary_policy_other_cs())
                        / (
                            final_year_transboundary_policy_other_cs()
                            - initial_year_transboundary_policy_other_cs()
                        )
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="LIMIT ENERGY CONSUMPTION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limit_energy_consumption"},
)
def limit_energy_consumption():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Energy', 'LIMIT_ENERGY_CONSUMPTION*')
    """
    return _ext_constant_limit_energy_consumption()


_ext_constant_limit_energy_consumption = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "LIMIT_ENERGY_CONSUMPTION*",
    {},
    _root,
    {},
    "_ext_constant_limit_energy_consumption",
)


@component.add(
    name="ENERGY CONSUMPTION VARIATION FACTOR",
    units="Hm3 per capita",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_energy_consumption_variation_factor"},
)
def energy_consumption_variation_factor():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Energy', 'ENERGY_CONSUMPTION_VARIATION_FACTOR*')
    """
    return _ext_constant_energy_consumption_variation_factor()


_ext_constant_energy_consumption_variation_factor = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "ENERGY_CONSUMPTION_VARIATION_FACTOR*",
    {},
    _root,
    {},
    "_ext_constant_energy_consumption_variation_factor",
)


@component.add(
    name="ENERGY CONSUMPTION VARIATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "energy_consumption_variation_factor": 2,
        "limit_energy_consumption": 1,
    },
)
def energy_consumption_variation():
    return if_then_else(
        time() < 2027,
        lambda: energy_consumption_variation_factor(),
        lambda: energy_consumption_variation_factor() * limit_energy_consumption(),
    )


@component.add(
    name="INITIAL CONSUMPTION PER CAPITA",
    units="MWh per capita",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_consumption_per_capita"},
)
def initial_consumption_per_capita():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Energy', 'INITIAL_CONSUMPTION_PER_CAPITA*')
    """
    return _ext_constant_initial_consumption_per_capita()


_ext_constant_initial_consumption_per_capita = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "INITIAL_CONSUMPTION_PER_CAPITA*",
    {},
    _root,
    {},
    "_ext_constant_initial_consumption_per_capita",
)


@component.add(
    name="CONSUMPTION PER CAPITA",
    units="MWh",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_consumption_per_capita": 1},
    other_deps={
        "_integ_consumption_per_capita": {
            "initial": {"initial_consumption_per_capita": 1},
            "step": {"variation_consumption_per_capita": 1},
        }
    },
)
def consumption_per_capita():
    return _integ_consumption_per_capita()


_integ_consumption_per_capita = Integ(
    lambda: variation_consumption_per_capita(),
    lambda: initial_consumption_per_capita(),
    "_integ_consumption_per_capita",
)


@component.add(
    name="VARIATION CONSUMPTION PER CAPITA",
    units="MWh per capita",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_variation_consumption_per_capita": 1},
    other_deps={
        "_smooth_variation_consumption_per_capita": {
            "initial": {"consumption_per_capita": 1, "energy_consumption_variation": 1},
            "step": {"consumption_per_capita": 1, "energy_consumption_variation": 1},
        }
    },
)
def variation_consumption_per_capita():
    return _smooth_variation_consumption_per_capita()


_smooth_variation_consumption_per_capita = Smooth(
    lambda: consumption_per_capita() * energy_consumption_variation(),
    lambda: 2,
    lambda: consumption_per_capita() * energy_consumption_variation(),
    lambda: 1,
    "_smooth_variation_consumption_per_capita",
)


@component.add(
    name="WATER AREA",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_water_storage_increase": 3,
        "time": 3,
        "water_storage_increase_area_objective": 1,
        "initial_year_water_storage_increase": 3,
        "final_year_water_storage_increase": 2,
    },
)
def water_area():
    """
    IF THEN ELSE(SWITCH WATER STORAGE INCREASE=0,0,IF THEN ELSE(SWITCH WATER STORAGE INCREASE=1:AND:Time<INITIAL YEAR WATER STORAGE INCREASE,0,IF THEN ELSE(SWITCH WATER STORAGE INCREASE=1:AND:Time>FINAL YEAR WATER STORAGE INCREASE,WATER STORAGE INCREASE AREA OBJECTIVE,(WATER STORAGE INCREASE AREA OBJECTIVE*((Time-INITIAL YEAR WATER STORAGE INCREASE)/(FINAL YEAR WATER STORAGE INCREASE -INITIAL YEAR WATER STORAGE INCREASE))))))
    """
    return if_then_else(
        switch_water_storage_increase() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_water_storage_increase() == 1,
                time() < initial_year_water_storage_increase(),
            ),
            lambda: 0,
            lambda: if_then_else(
                np.logical_and(
                    switch_water_storage_increase() == 1,
                    time() > final_year_water_storage_increase(),
                ),
                lambda: 0,
                lambda: water_storage_increase_area_objective()
                * (
                    (time() - initial_year_water_storage_increase())
                    / (
                        final_year_water_storage_increase()
                        - initial_year_water_storage_increase()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="SELECT LOCATION WATER STORAGE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_location_water_storage"},
)
def select_location_water_storage():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Land system', 'SELECT_LOCATION_WATER_STORAGE*')
    """
    return _ext_constant_select_location_water_storage()


_ext_constant_select_location_water_storage = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SELECT_LOCATION_WATER_STORAGE*",
    {},
    _root,
    {},
    "_ext_constant_select_location_water_storage",
)


@component.add(
    name="FINAL FLOW",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"runoff_to_river_flow": 1, "ecological_flow": 1, "storage_inputs": 1},
)
def final_flow():
    return runoff_to_river_flow() * (1 - ecological_flow()) - storage_inputs()


@component.add(
    name="DELAY EDUCATION",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_delay_education"},
)
def delay_education():
    return _ext_constant_delay_education()


_ext_constant_delay_education = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "DELAY_EDUCATION*",
    {},
    _root,
    {},
    "_ext_constant_delay_education",
)


@component.add(
    name="TEMPERATURE SSP585",
    units="ºc",
    comp_type="Data",
    comp_subtype="Normal",
    depends_on={"mean_temperature_ssp585": 1},
)
def temperature_ssp585():
    return (
        sum(mean_temperature_ssp585().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        / 12
    )


@component.add(
    name="INCREASE IN POPULATION ADAPTATION CAPACITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_education_and_training_increase": 2,
        "delay_education": 1,
        "time": 1,
        "educational_level": 1,
        "education_and_training_objective": 1,
        "initial_year_education_and_training_increase": 1,
    },
)
def increase_in_population_adaptation_capacity():
    return if_then_else(
        switch_education_and_training_increase() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_education_and_training_increase() == 1,
                time()
                < initial_year_education_and_training_increase() + delay_education(),
            ),
            lambda: 0,
            lambda: educational_level() * education_and_training_objective(),
        ),
    )


@component.add(
    name="TEMPERATURE SSP245",
    units="ºc",
    comp_type="Data",
    comp_subtype="Normal",
    depends_on={"mean_temperature_ssp245": 1},
)
def temperature_ssp245():
    return (
        sum(mean_temperature_ssp245().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])
        / 12
    )


@component.add(
    name="FOREST MORTALITY COEFFICIENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_forest_mortality_coefficient"},
)
def forest_mortality_coefficient():
    return _ext_constant_forest_mortality_coefficient()


_ext_constant_forest_mortality_coefficient = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "FOREST_MORTALITY_COEFFICIENT",
    {},
    _root,
    {},
    "_ext_constant_forest_mortality_coefficient",
)


@component.add(
    name="FOREST MORTALITY",
    units="tC",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"biomass_stock": 1, "forest_mortality_coefficient": 1},
)
def forest_mortality():
    return biomass_stock() * forest_mortality_coefficient()


@component.add(
    name="WATER STORAGE INCREASE AREA OBJECTIVE",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_water_storage_increase_area_objective"},
)
def water_storage_increase_area_objective():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Land system', 'INITIAL_WATER_STORAGE_INCREASE_AREA_OBJECTIVE*')
    """
    return _ext_constant_water_storage_increase_area_objective()


_ext_constant_water_storage_increase_area_objective = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INITIAL_WATER_STORAGE_INCREASE_AREA_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_water_storage_increase_area_objective",
)


@component.add(
    name="FINAL YEAR WATER STORAGE INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_water_storage_increase"},
)
def final_year_water_storage_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'FINAL_YEAR_WATER_STORAGE_INCREASE*')
    """
    return _ext_constant_final_year_water_storage_increase()


_ext_constant_final_year_water_storage_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_WATER_STORAGE_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_water_storage_increase",
)


@component.add(
    name="UNEMPLOYED",
    units="Inhabitants",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "gamma_unemployed": 3,
        "time": 3,
        "beta_unemployed": 3,
        "gdp_per_capita": 5,
        "alfa_unemployed": 3,
        "limit_unemployed": 2,
    },
)
def unemployed():
    """
    IF THEN ELSE (GAMMA UNEMPLOYED=0,ALFA UNEMPLOYED*GDP PER CAPITA+BETA UNEMPLOYED,IF THEN ELSE(Time<=2030,(ALFA UNEMPLOYED*GDP PER CAPITA^2+BETA UNEMPLOYED*GDP PER CAPITA+GAMMA UNEMPLOYED), (ALFA UNEMPLOYED*GDP PER CAPITA^2+BETA UNEMPLOYED*GDP PER CAPITA+GAMMA UNEMPLOYED)*(1-LIMIT UNEMPLOYED*(Time-2030))))
    """
    return if_then_else(
        gamma_unemployed() == 0,
        lambda: (alfa_unemployed() * gdp_per_capita() + beta_unemployed())
        * (1 - limit_unemployed() * (time() - 2018)),
        lambda: if_then_else(
            time() <= 2030,
            lambda: alfa_unemployed() * gdp_per_capita() ** 2
            + beta_unemployed() * gdp_per_capita()
            + gamma_unemployed(),
            lambda: (
                alfa_unemployed() * gdp_per_capita() ** 2
                + beta_unemployed() * gdp_per_capita()
                + gamma_unemployed()
            )
            * (1 - limit_unemployed() * (time() - 2030)),
        ),
    )


@component.add(
    name="LIMIT ENERGY POVERTY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limit_energy_poverty"},
)
def limit_energy_poverty():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'LIMIT_ENERGY_POVERTY*')
    """
    return _ext_constant_limit_energy_poverty()


_ext_constant_limit_energy_poverty = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "LIMIT_ENERGY_POVERTY*",
    {},
    _root,
    {},
    "_ext_constant_limit_energy_poverty",
)


@component.add(
    name="WATER STORAGE INCREASE CAPACITY OBJECTIVE",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_water_storage_increase_capacity_objective"
    },
)
def water_storage_increase_capacity_objective():
    return _ext_constant_water_storage_increase_capacity_objective()


_ext_constant_water_storage_increase_capacity_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_WATER_STORAGE_INCREASE_CAPACITY_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_water_storage_increase_capacity_objective",
)


@component.add(
    name="INITIAL RESERVOIR CAPACITY",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_reservoir_capacity"},
)
def initial_reservoir_capacity():
    return _ext_constant_initial_reservoir_capacity()


_ext_constant_initial_reservoir_capacity = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "TOTAL_RESERVOIR_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_reservoir_capacity",
)


@component.add(
    name="SWITCH WATER STORAGE INCREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_water_storage_increase"},
)
def switch_water_storage_increase():
    """
    0 = OFF 1 = ON GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SWITCH_WATER_STORAGE_INCREASE*')
    """
    return _ext_constant_switch_water_storage_increase()


_ext_constant_switch_water_storage_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_WATER_STORAGE_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_switch_water_storage_increase",
)


@component.add(
    name="TOTAL RESERVOIR CAPACITY",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_water_storage_increase": 3,
        "initial_reservoir_capacity": 4,
        "final_year_water_storage_increase": 2,
        "time": 3,
        "initial_year_water_storage_increase": 3,
        "water_storage_increase_capacity_objective": 2,
    },
)
def total_reservoir_capacity():
    return if_then_else(
        switch_water_storage_increase() == 0,
        lambda: initial_reservoir_capacity(),
        lambda: if_then_else(
            np.logical_and(
                switch_water_storage_increase() == 1,
                time() < initial_year_water_storage_increase(),
            ),
            lambda: initial_reservoir_capacity(),
            lambda: if_then_else(
                np.logical_and(
                    switch_water_storage_increase() == 1,
                    time() > final_year_water_storage_increase(),
                ),
                lambda: initial_reservoir_capacity()
                + water_storage_increase_capacity_objective(),
                lambda: initial_reservoir_capacity()
                + water_storage_increase_capacity_objective()
                * (
                    (time() - initial_year_water_storage_increase() + 1)
                    / (
                        final_year_water_storage_increase()
                        - initial_year_water_storage_increase()
                        + 1
                    )
                ),
            ),
        ),
    )


@component.add(
    name="INITIAL YEAR WATER STORAGE INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_water_storage_increase"},
)
def initial_year_water_storage_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'INITIAL_YEAR_WATER_STORAGE_INCREASE*')
    """
    return _ext_constant_initial_year_water_storage_increase()


_ext_constant_initial_year_water_storage_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_WATER_STORAGE_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_water_storage_increase",
)


@component.add(
    name="LIMIT UNEMPLOYED",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limit_unemployed"},
)
def limit_unemployed():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'LIMIT_UNEMPLOYED*')
    """
    return _ext_constant_limit_unemployed()


_ext_constant_limit_unemployed = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "LIMIT_UNEMPLOYED*",
    {},
    _root,
    {},
    "_ext_constant_limit_unemployed",
)


@component.add(
    name="VARIATION WATER DEMAND FROM INDUSTRY",
    units="Hm3/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "water_demand_from_industry_per_capita": 1,
        "industry_variation": 2,
        "limiting_factor_industry": 1,
    },
)
def variation_water_demand_from_industry():
    return (water_demand_from_industry_per_capita() * industry_variation()) * (
        1 - industry_variation() / limiting_factor_industry()
    )


@component.add(
    name="VARIATION WATER DEMAND URBAN PER CAPITA",
    units="Hm3/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "water_demand_from_urban_per_capita": 1,
        "urban_variation": 2,
        "limiting_factor_urban": 1,
    },
)
def variation_water_demand_urban_per_capita():
    return (water_demand_from_urban_per_capita() * urban_variation()) * (
        1 - urban_variation() / limiting_factor_urban()
    )


@component.add(
    name="LIMITING FACTOR INDUSTRY",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limiting_factor_industry"},
)
def limiting_factor_industry():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'LIMITING_FACTOR_INDUSTRY*')
    """
    return _ext_constant_limiting_factor_industry()


_ext_constant_limiting_factor_industry = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LIMITING_FACTOR_INDUSTRY*",
    {},
    _root,
    {},
    "_ext_constant_limiting_factor_industry",
)


@component.add(
    name="LIMITING FACTOR URBAN",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limiting_factor_urban"},
)
def limiting_factor_urban():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'LIMITING_FACTOR_URBAN*')
    """
    return _ext_constant_limiting_factor_urban()


_ext_constant_limiting_factor_urban = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LIMITING_FACTOR_URBAN*",
    {},
    _root,
    {},
    "_ext_constant_limiting_factor_urban",
)


@component.add(
    name="LIMITING FACTOR AGRICULTURE",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_limiting_factor_agriculture"},
)
def limiting_factor_agriculture():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'LIMITING_FACTOR_AGRICULTURE*')
    """
    return _ext_constant_limiting_factor_agriculture()


_ext_constant_limiting_factor_agriculture = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LIMITING_FACTOR_AGRICULTURE*",
    {},
    _root,
    {},
    "_ext_constant_limiting_factor_agriculture",
)


@component.add(
    name="INDUSTRY VARIATION",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_industry_variation"},
)
def industry_variation():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'INDUSTRY_VARIATION*')
    """
    return _ext_constant_industry_variation()


_ext_constant_industry_variation = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INDUSTRY_VARIATION*",
    {},
    _root,
    {},
    "_ext_constant_industry_variation",
)


@component.add(
    name="AGRICULTURE VARIATION",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_agriculture_variation"},
)
def agriculture_variation():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'AGRICULTURE_VARIATION*')
    """
    return _ext_constant_agriculture_variation()


_ext_constant_agriculture_variation = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "AGRICULTURE_VARIATION*",
    {},
    _root,
    {},
    "_ext_constant_agriculture_variation",
)


@component.add(
    name="VARIATION WATER DEMAND FROM AGRICULTURE",
    units="Hm3/Year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "water_demand_from_agriculture_per_capita": 1,
        "agriculture_variation": 2,
        "limiting_factor_agriculture": 1,
    },
)
def variation_water_demand_from_agriculture():
    return (water_demand_from_agriculture_per_capita() * agriculture_variation()) * (
        1 - agriculture_variation() / limiting_factor_agriculture()
    )


@component.add(
    name="URBAN VARIATION",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_urban_variation"},
)
def urban_variation():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'URBAN_VARIATION*')
    """
    return _ext_constant_urban_variation()


_ext_constant_urban_variation = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "URBAN_VARIATION*",
    {},
    _root,
    {},
    "_ext_constant_urban_variation",
)


@component.add(
    name="WATER DEMAND FROM AGRICULTURE PER CAPITA",
    units="Hm3/Year",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_water_demand_from_agriculture_per_capita": 1},
    other_deps={
        "_integ_water_demand_from_agriculture_per_capita": {
            "initial": {"initial_water_demand_from_agriculture_per_capita": 1},
            "step": {"variation_water_demand_from_agriculture": 1},
        }
    },
)
def water_demand_from_agriculture_per_capita():
    return _integ_water_demand_from_agriculture_per_capita()


_integ_water_demand_from_agriculture_per_capita = Integ(
    lambda: variation_water_demand_from_agriculture(),
    lambda: initial_water_demand_from_agriculture_per_capita(),
    "_integ_water_demand_from_agriculture_per_capita",
)


@component.add(
    name="WATER DEMAND FROM INDUSTRY PER CAPITA",
    units="Hm3/Year",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_water_demand_from_industry_per_capita": 1},
    other_deps={
        "_integ_water_demand_from_industry_per_capita": {
            "initial": {"initial_water_demand_from_industry_per_capita": 1},
            "step": {"variation_water_demand_from_industry": 1},
        }
    },
)
def water_demand_from_industry_per_capita():
    return _integ_water_demand_from_industry_per_capita()


_integ_water_demand_from_industry_per_capita = Integ(
    lambda: variation_water_demand_from_industry(),
    lambda: initial_water_demand_from_industry_per_capita(),
    "_integ_water_demand_from_industry_per_capita",
)


@component.add(
    name="WATER DEMAND FROM URBAN PER CAPITA",
    units="Hm3/Year",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_water_demand_from_urban_per_capita": 1},
    other_deps={
        "_integ_water_demand_from_urban_per_capita": {
            "initial": {"initial_water_demand_from_urban": 1},
            "step": {"variation_water_demand_urban_per_capita": 1},
        }
    },
)
def water_demand_from_urban_per_capita():
    return _integ_water_demand_from_urban_per_capita()


_integ_water_demand_from_urban_per_capita = Integ(
    lambda: variation_water_demand_urban_per_capita(),
    lambda: initial_water_demand_from_urban(),
    "_integ_water_demand_from_urban_per_capita",
)


@component.add(
    name="GROUNDWATER MANAGEMENT",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="with Lookup",
    depends_on={"groundwater_resources": 1},
)
def groundwater_management():
    """
    ([(0,0)-(10,10)],(0,0),(5000,1),(10000,1),(20000,1),(50000,1),(75000,1) )
    """
    return np.interp(
        groundwater_resources(),
        [0, 50000, 100000, 200000, 500000, 750000],
        [0, 1, 1, 1, 1, 1],
    )


@component.add(
    name="GROUNDWATER CONSUMPTION",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"request_from_groundwater_extraction": 1, "groundwater_management": 1},
)
def groundwater_consumption():
    return request_from_groundwater_extraction() * groundwater_management()


@component.add(
    name="DECAY GDP PER CAPITA GROWTH RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_decay_gdp_per_capita_growth_rate"},
)
def decay_gdp_per_capita_growth_rate():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'DECAY_GDP_PER_CAPITA_GROWTH_RATE*')
    """
    return _ext_constant_decay_gdp_per_capita_growth_rate()


_ext_constant_decay_gdp_per_capita_growth_rate = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "DECAY_GDP_PER_CAPITA_GROWTH_RATE*",
    {},
    _root,
    {},
    "_ext_constant_decay_gdp_per_capita_growth_rate",
)


@component.add(
    name="INITIAL GDP PER CAPITA",
    units="Euros per capita",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_gdp_per_capita"},
)
def initial_gdp_per_capita():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'INITIAL_GDP_PER_CAPITA*')
    """
    return _ext_constant_initial_gdp_per_capita()


_ext_constant_initial_gdp_per_capita = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "INITIAL_GDP_PER_CAPITA*",
    {},
    _root,
    {},
    "_ext_constant_initial_gdp_per_capita",
)


@component.add(
    name="GDP PER CAPITA",
    units="Euros per capita",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_gdp_per_capita": 1},
    other_deps={
        "_integ_gdp_per_capita": {
            "initial": {"initial_gdp_per_capita": 1},
            "step": {"variation_gdp_per_capita": 1},
        }
    },
)
def gdp_per_capita():
    return _integ_gdp_per_capita()


_integ_gdp_per_capita = Integ(
    lambda: variation_gdp_per_capita(),
    lambda: initial_gdp_per_capita(),
    "_integ_gdp_per_capita",
)


@component.add(
    name="POPULATION IN ENERGY POVERTY",
    units="Inhabitants",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 2,
        "gdp_per_capita": 2,
        "alfa_energy_poverty": 2,
        "beta_energy_poverty": 2,
        "limit_energy_poverty": 1,
    },
)
def population_in_energy_poverty():
    return if_then_else(
        time() <= 2030,
        lambda: alfa_energy_poverty() * gdp_per_capita() + beta_energy_poverty(),
        lambda: (alfa_energy_poverty() * gdp_per_capita() + beta_energy_poverty())
        * (1 - limit_energy_poverty() * (time() - 2030)),
    )


@component.add(
    name="VARIATION GDP PER CAPITA",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "gdp_per_capita": 1,
        "time": 1,
        "gdp_per_capita_growth_rate": 1,
        "decay_gdp_per_capita_growth_rate": 1,
        "initial_time": 1,
    },
)
def variation_gdp_per_capita():
    return gdp_per_capita() * (
        gdp_per_capita_growth_rate()
        - decay_gdp_per_capita_growth_rate() * (time() - initial_time())
    )


@component.add(
    name="GDP PER CAPITA GROWTH RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_gdp_per_capita_growth_rate"},
)
def gdp_per_capita_growth_rate():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'GDP_PER_CAPITA_GROWTH_RATE*')
    """
    return _ext_constant_gdp_per_capita_growth_rate()


_ext_constant_gdp_per_capita_growth_rate = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "GDP_PER_CAPITA_GROWTH_RATE*",
    {},
    _root,
    {},
    "_ext_constant_gdp_per_capita_growth_rate",
)


@component.add(
    name="BETA BIRTHS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_births"},
)
def beta_births():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_BIRTHS*')
    """
    return _ext_constant_beta_births()


_ext_constant_beta_births = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_BIRTHS*",
    {},
    _root,
    {},
    "_ext_constant_beta_births",
)


@component.add(
    name="BETA DEATHS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_deaths"},
)
def beta_deaths():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_DEATHS*')
    """
    return _ext_constant_beta_deaths()


_ext_constant_beta_deaths = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_DEATHS*",
    {},
    _root,
    {},
    "_ext_constant_beta_deaths",
)


@component.add(
    name="BETA EDUCATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_education"},
)
def beta_education():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_EDUCATION*')
    """
    return _ext_constant_beta_education()


_ext_constant_beta_education = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_EDUCATION*",
    {},
    _root,
    {},
    "_ext_constant_beta_education",
)


@component.add(
    name="BETA ENERGY POVERTY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_energy_poverty"},
)
def beta_energy_poverty():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_ENERGY_POVERTY*')
    """
    return _ext_constant_beta_energy_poverty()


_ext_constant_beta_energy_poverty = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_ENERGY_POVERTY*",
    {},
    _root,
    {},
    "_ext_constant_beta_energy_poverty",
)


@component.add(
    name="ALFA ROOM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_room"},
)
def alfa_room():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_ROOMS*')
    """
    return _ext_constant_alfa_room()


_ext_constant_alfa_room = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_ROOMS*",
    {},
    _root,
    {},
    "_ext_constant_alfa_room",
)


@component.add(
    name="ROOMS",
    units="Rooms",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_rooms": 1},
    other_deps={
        "_smooth_rooms": {
            "initial": {
                "alfa_room": 3,
                "cs_population": 6,
                "beta_room": 3,
                "gamma_rooms": 3,
            },
            "step": {
                "alfa_room": 3,
                "cs_population": 6,
                "beta_room": 3,
                "gamma_rooms": 3,
            },
        }
    },
)
def rooms():
    return _smooth_rooms()


_smooth_rooms = Smooth(
    lambda: if_then_else(
        np.logical_and(
            alfa_room() * cs_population() + beta_room() < 0.0008 * cs_population(),
            gamma_rooms() == 0,
        ),
        lambda: 0.0008 * cs_population(),
        lambda: if_then_else(
            gamma_rooms() == 0,
            lambda: alfa_room() * cs_population() + beta_room(),
            lambda: alfa_room() * cs_population() ** 2
            + beta_room() * cs_population()
            + gamma_rooms(),
        ),
    ),
    lambda: 2,
    lambda: if_then_else(
        np.logical_and(
            alfa_room() * cs_population() + beta_room() < 0.0008 * cs_population(),
            gamma_rooms() == 0,
        ),
        lambda: 0.0008 * cs_population(),
        lambda: if_then_else(
            gamma_rooms() == 0,
            lambda: alfa_room() * cs_population() + beta_room(),
            lambda: alfa_room() * cs_population() ** 2
            + beta_room() * cs_population()
            + gamma_rooms(),
        ),
    ),
    lambda: 1,
    "_smooth_rooms",
)


@component.add(
    name="BETA ROOM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_room"},
)
def beta_room():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_ROOMS*')
    """
    return _ext_constant_beta_room()


_ext_constant_beta_room = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_ROOMS*",
    {},
    _root,
    {},
    "_ext_constant_beta_room",
)


@component.add(
    name="BETA UNEMPLOYED",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_unemployed"},
)
def beta_unemployed():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_UNEMPLOYED*')
    """
    return _ext_constant_beta_unemployed()


_ext_constant_beta_unemployed = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_UNEMPLOYED*",
    {},
    _root,
    {},
    "_ext_constant_beta_unemployed",
)


@component.add(
    name="ALFA BIRTHS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_births"},
)
def alfa_births():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_BIRTHS*')
    """
    return _ext_constant_alfa_births()


_ext_constant_alfa_births = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_BIRTHS*",
    {},
    _root,
    {},
    "_ext_constant_alfa_births",
)


@component.add(
    name="ALFA DEATHS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_deaths"},
)
def alfa_deaths():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_DEATHS*')
    """
    return _ext_constant_alfa_deaths()


_ext_constant_alfa_deaths = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_DEATHS*",
    {},
    _root,
    {},
    "_ext_constant_alfa_deaths",
)


@component.add(
    name="GAMMA ROOMS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_gamma_rooms"},
)
def gamma_rooms():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'GAMMA_ROOMS*')
    """
    return _ext_constant_gamma_rooms()


_ext_constant_gamma_rooms = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "GAMMA_ROOMS*",
    {},
    _root,
    {},
    "_ext_constant_gamma_rooms",
)


@component.add(
    name="GAMMA UNEMPLOYED",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_gamma_unemployed"},
)
def gamma_unemployed():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'GAMMA_UNEMPLOYED*')
    """
    return _ext_constant_gamma_unemployed()


_ext_constant_gamma_unemployed = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "GAMMA_UNEMPLOYED*",
    {},
    _root,
    {},
    "_ext_constant_gamma_unemployed",
)


@component.add(
    name="INITIAL WATER DEMAND FROM INDUSTRY PER CAPITA",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_water_demand_from_industry_per_capita"
    },
)
def initial_water_demand_from_industry_per_capita():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'INITIAL_WATER_DEMAND_FROM_INDUSTRY_PER_CAPITA*')
    """
    return _ext_constant_initial_water_demand_from_industry_per_capita()


_ext_constant_initial_water_demand_from_industry_per_capita = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_WATER_DEMAND_FROM_INDUSTRY_PER_CAPITA*",
    {},
    _root,
    {},
    "_ext_constant_initial_water_demand_from_industry_per_capita",
)


@component.add(
    name="BIRTHS",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_births": 1},
    other_deps={
        "_smooth_births": {
            "initial": {"alfa_births": 2, "cs_population": 4, "beta_births": 2},
            "step": {"alfa_births": 2, "cs_population": 4, "beta_births": 2},
        }
    },
)
def births():
    return _smooth_births()


_smooth_births = Smooth(
    lambda: if_then_else(
        alfa_births() * cs_population() + beta_births() < 0.001 * cs_population(),
        lambda: 0.001 * cs_population(),
        lambda: alfa_births() * cs_population() + beta_births(),
    ),
    lambda: 2,
    lambda: if_then_else(
        alfa_births() * cs_population() + beta_births() < 0.001 * cs_population(),
        lambda: 0.001 * cs_population(),
        lambda: alfa_births() * cs_population() + beta_births(),
    ),
    lambda: 1,
    "_smooth_births",
)


@component.add(
    name="INITIAL WATER DEMAND FROM AGRICULTURE PER CAPITA",
    units="Hm3/Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_water_demand_from_agriculture_per_capita"
    },
)
def initial_water_demand_from_agriculture_per_capita():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'INITIAL_WATER_DEMAND_FROM_AGRICULTURE_PER_CAPITA*')
    """
    return _ext_constant_initial_water_demand_from_agriculture_per_capita()


_ext_constant_initial_water_demand_from_agriculture_per_capita = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_WATER_DEMAND_FROM_AGRICULTURE_PER_CAPITA*",
    {},
    _root,
    {},
    "_ext_constant_initial_water_demand_from_agriculture_per_capita",
)


@component.add(
    name="DEATHS",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_deaths": 1},
    other_deps={
        "_smooth_deaths": {
            "initial": {"alfa_deaths": 2, "cs_population": 4, "beta_deaths": 2},
            "step": {"alfa_deaths": 2, "cs_population": 4, "beta_deaths": 2},
        }
    },
)
def deaths():
    return _smooth_deaths()


_smooth_deaths = Smooth(
    lambda: if_then_else(
        alfa_deaths() * cs_population() + beta_deaths() < 0.002 * cs_population(),
        lambda: 0.002 * cs_population(),
        lambda: alfa_deaths() * cs_population() + beta_deaths(),
    ),
    lambda: 2,
    lambda: if_then_else(
        alfa_deaths() * cs_population() + beta_deaths() < 0.002 * cs_population(),
        lambda: 0.002 * cs_population(),
        lambda: alfa_deaths() * cs_population() + beta_deaths(),
    ),
    lambda: 1,
    "_smooth_deaths",
)


@component.add(
    name="ALFA EDUCATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_education"},
)
def alfa_education():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_EDUCATION*')
    """
    return _ext_constant_alfa_education()


_ext_constant_alfa_education = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_EDUCATION*",
    {},
    _root,
    {},
    "_ext_constant_alfa_education",
)


@component.add(
    name="ALFA ENERGY POVERTY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_energy_poverty"},
)
def alfa_energy_poverty():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_ENERGY_POVERTY*')
    """
    return _ext_constant_alfa_energy_poverty()


_ext_constant_alfa_energy_poverty = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_ENERGY_POVERTY*",
    {},
    _root,
    {},
    "_ext_constant_alfa_energy_poverty",
)


@component.add(
    name="ALFA UNEMPLOYED",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_unemployed"},
)
def alfa_unemployed():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_UNEMPLOYED*')
    """
    return _ext_constant_alfa_unemployed()


_ext_constant_alfa_unemployed = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_UNEMPLOYED*",
    {},
    _root,
    {},
    "_ext_constant_alfa_unemployed",
)


@component.add(
    name="EDUCATED POPULATION",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_educated_population": 1},
    other_deps={
        "_smooth_educated_population": {
            "initial": {"alfa_education": 2, "cs_population": 4, "beta_education": 2},
            "step": {"alfa_education": 2, "cs_population": 4, "beta_education": 2},
        }
    },
)
def educated_population():
    return _smooth_educated_population()


_smooth_educated_population = Smooth(
    lambda: if_then_else(
        alfa_education() * cs_population() + beta_education() < 0.05 * cs_population(),
        lambda: 0.05 * cs_population(),
        lambda: alfa_education() * cs_population() + beta_education(),
    ),
    lambda: 2,
    lambda: if_then_else(
        alfa_education() * cs_population() + beta_education() < 0.05 * cs_population(),
        lambda: 0.05 * cs_population(),
        lambda: alfa_education() * cs_population() + beta_education(),
    ),
    lambda: 1,
    "_smooth_educated_population",
)


@component.add(
    name="ALFA POP LOWER 5",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_pop_lower_5"},
)
def alfa_pop_lower_5():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_POP_LOWER_5*')
    """
    return _ext_constant_alfa_pop_lower_5()


_ext_constant_alfa_pop_lower_5 = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_POP_LOWER_5*",
    {},
    _root,
    {},
    "_ext_constant_alfa_pop_lower_5",
)


@component.add(
    name="POPULATION GROWTH SSP245",
    units="Inhabitants",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_population_growth_ssp245",
        "__data__": "_ext_data_population_growth_ssp245",
        "time": 1,
    },
)
def population_growth_ssp245():
    return _ext_data_population_growth_ssp245(time())


_ext_data_population_growth_ssp245 = ExtData(
    r"Historical.xlsx",
    "Energy",
    "POPULATION_GROWTH_TIME",
    "POPULATION_GROWTH_SSP245",
    None,
    {},
    _root,
    {},
    "_ext_data_population_growth_ssp245",
)


@component.add(
    name="POPULATION GROWTH SSP585",
    units="Inhabitants",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_population_growth_ssp585",
        "__data__": "_ext_data_population_growth_ssp585",
        "time": 1,
    },
)
def population_growth_ssp585():
    return _ext_data_population_growth_ssp585(time())


_ext_data_population_growth_ssp585 = ExtData(
    r"Historical.xlsx",
    "Energy",
    "POPULATION_GROWTH_TIME",
    "POPULATION_GROWTH_SSP585",
    None,
    {},
    _root,
    {},
    "_ext_data_population_growth_ssp585",
)


@component.add(
    name="POPULATION GROWTH",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_population_growth_scenario": 2,
        "population_growth_ssp245": 1,
        "population_growth_ssp585": 1,
    },
)
def population_growth():
    return if_then_else(
        select_population_growth_scenario() == 0,
        lambda: population_growth_ssp245(),
        lambda: if_then_else(
            select_population_growth_scenario() == 1,
            lambda: population_growth_ssp585(),
            lambda: 1,
        ),
    )


@component.add(
    name="POPULATION HIGHER 65",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_population_higher_65": 1},
    other_deps={
        "_smooth_population_higher_65": {
            "initial": {
                "alfa_pop_higher_65": 1,
                "cs_population": 1,
                "beta_pop_higher_65": 1,
            },
            "step": {
                "alfa_pop_higher_65": 1,
                "cs_population": 1,
                "beta_pop_higher_65": 1,
            },
        }
    },
)
def population_higher_65():
    return _smooth_population_higher_65()


_smooth_population_higher_65 = Smooth(
    lambda: alfa_pop_higher_65() * cs_population() + beta_pop_higher_65(),
    lambda: 2,
    lambda: alfa_pop_higher_65() * cs_population() + beta_pop_higher_65(),
    lambda: 1,
    "_smooth_population_higher_65",
)


@component.add(
    name="ALFA POP HIGHER 65",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_alfa_pop_higher_65"},
)
def alfa_pop_higher_65():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'ALFA_POP_HIGHER_65*')
    """
    return _ext_constant_alfa_pop_higher_65()


_ext_constant_alfa_pop_higher_65 = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ALFA_POP_HIGHER_65*",
    {},
    _root,
    {},
    "_ext_constant_alfa_pop_higher_65",
)


@component.add(
    name="SELECT POPULATION GROWTH SCENARIO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_population_growth_scenario"},
)
def select_population_growth_scenario():
    """
    0=SSP245 1=SSP585
    """
    return _ext_constant_select_population_growth_scenario()


_ext_constant_select_population_growth_scenario = ExtConstant(
    r"Policy.xlsx",
    "Energy",
    "SELECT_POPULATION_GROWTH_SCENARIO*",
    {},
    _root,
    {},
    "_ext_constant_select_population_growth_scenario",
)


@component.add(
    name="POPULATION LOWER 5",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_population_lower_5": 1},
    other_deps={
        "_smooth_population_lower_5": {
            "initial": {
                "alfa_pop_lower_5": 1,
                "cs_population": 1,
                "beta_pop_lower_5": 1,
            },
            "step": {"alfa_pop_lower_5": 1, "cs_population": 1, "beta_pop_lower_5": 1},
        }
    },
)
def population_lower_5():
    return _smooth_population_lower_5()


_smooth_population_lower_5 = Smooth(
    lambda: alfa_pop_lower_5() * cs_population() + beta_pop_lower_5(),
    lambda: 2,
    lambda: alfa_pop_lower_5() * cs_population() + beta_pop_lower_5(),
    lambda: 1,
    "_smooth_population_lower_5",
)


@component.add(
    name="BETA POP HIGHER 65",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_pop_higher_65"},
)
def beta_pop_higher_65():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_POP_HIGHER_65*')
    """
    return _ext_constant_beta_pop_higher_65()


_ext_constant_beta_pop_higher_65 = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_POP_HIGHER_65*",
    {},
    _root,
    {},
    "_ext_constant_beta_pop_higher_65",
)


@component.add(
    name="BETA POP LOWER 5",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_beta_pop_lower_5"},
)
def beta_pop_lower_5():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Population system', 'BETA_POP_LOWER_5*')
    """
    return _ext_constant_beta_pop_lower_5()


_ext_constant_beta_pop_lower_5 = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "BETA_POP_LOWER_5*",
    {},
    _root,
    {},
    "_ext_constant_beta_pop_lower_5",
)


@component.add(
    name="STORAGE INPUTS",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "storage_water": 2,
        "total_reservoir_capacity": 2,
        "ecological_flow": 1,
        "runoff_to_river_flow": 1,
    },
)
def storage_inputs():
    return if_then_else(
        storage_water() < total_reservoir_capacity(),
        lambda: float(
            np.minimum(
                runoff_to_river_flow() * (1 - ecological_flow()),
                total_reservoir_capacity() - storage_water(),
            )
        ),
        lambda: 0,
    )


@component.add(
    name="STORAGE OUTPUTS",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_demand": 2, "available_water": 2},
)
def storage_outputs():
    return if_then_else(
        water_demand() > available_water(),
        lambda: available_water(),
        lambda: water_demand(),
    )


@component.add(
    name="AVAILABLE WATER",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"storage_water": 2, "remaining_storage_water": 2},
)
def available_water():
    return if_then_else(
        storage_water() < remaining_storage_water(),
        lambda: 0,
        lambda: storage_water() - remaining_storage_water(),
    )


@component.add(
    name="WATER DEMAND FROM INDUSTRY",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_efficient_water_demand_from_industry": 3,
        "cs_population": 4,
        "water_demand_from_industry_per_capita": 4,
        "final_year_efficient_water_demand_from_industry": 2,
        "time": 3,
        "initial_year_efficient_water_demand_from_industry": 3,
        "efficient_water_demand_from_industry_objective": 2,
    },
)
def water_demand_from_industry():
    return if_then_else(
        switch_efficient_water_demand_from_industry() == 0,
        lambda: water_demand_from_industry_per_capita() * cs_population(),
        lambda: if_then_else(
            np.logical_and(
                switch_efficient_water_demand_from_industry() == 1,
                time() < initial_year_efficient_water_demand_from_industry(),
            ),
            lambda: water_demand_from_industry_per_capita() * cs_population(),
            lambda: if_then_else(
                np.logical_and(
                    switch_efficient_water_demand_from_industry() == 1,
                    time() > final_year_efficient_water_demand_from_industry(),
                ),
                lambda: (cs_population() * water_demand_from_industry_per_capita())
                * (1 - efficient_water_demand_from_industry_objective()),
                lambda: (cs_population() * water_demand_from_industry_per_capita())
                * (
                    1
                    - efficient_water_demand_from_industry_objective()
                    * (
                        (time() - initial_year_efficient_water_demand_from_industry())
                        / (
                            final_year_efficient_water_demand_from_industry()
                            - initial_year_efficient_water_demand_from_industry()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="GROUNDWATER INPUTS",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"to_aquifer_yearly": 1, "infiltration_from_agriculture": 1},
)
def groundwater_inputs():
    return to_aquifer_yearly() + infiltration_from_agriculture()


@component.add(
    name="WATER LOSSES",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"evaporation_from_storage": 1, "storage_water": 1},
)
def water_losses():
    return evaporation_from_storage() * storage_water()


@component.add(
    name="STORAGE WATER",
    units="Hm3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_storage_water": 1},
    other_deps={
        "_integ_storage_water": {
            "initial": {"initial_storage": 1},
            "step": {"storage_inputs": 1, "storage_outputs": 1, "water_losses": 1},
        }
    },
)
def storage_water():
    return _integ_storage_water()


_integ_storage_water = Integ(
    lambda: storage_inputs() - storage_outputs() - water_losses(),
    lambda: initial_storage(),
    "_integ_storage_water",
)


@component.add(
    name="TO AQUIFER YEARLY",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"to_aquifer": 1},
)
def to_aquifer_yearly():
    return sum(to_aquifer().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])


@component.add(
    name="INITIAL TRANSBOUNDARY OBJECTIVE OTHER CS MONTHLY",
    units="Hm3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"transboundary_objective_other_cs": 1},
)
def initial_transboundary_objective_other_cs_monthly():
    return if_then_else(
        np.logical_or(
            xr.DataArray(
                np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                {"MONTHS": _subscript_dict["MONTHS"]},
                ["MONTHS"],
            )
            == 2,
            np.logical_or(
                xr.DataArray(
                    np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                    {"MONTHS": _subscript_dict["MONTHS"]},
                    ["MONTHS"],
                )
                == 3,
                np.logical_or(
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 4,
                    xr.DataArray(
                        np.arange(1, len(_subscript_dict["MONTHS"]) + 1),
                        {"MONTHS": _subscript_dict["MONTHS"]},
                        ["MONTHS"],
                    )
                    == 5,
                ),
            ),
        ),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        lambda: xr.DataArray(
            transboundary_objective_other_cs(),
            {"MONTHS": _subscript_dict["MONTHS"]},
            ["MONTHS"],
        ),
    )


@component.add(
    name="GROUNDWATER RESOURCES",
    units="Hm3",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_groundwater_resources": 1},
    other_deps={
        "_integ_groundwater_resources": {
            "initial": {"initial_groundwater": 1},
            "step": {"groundwater_inputs": 1, "groundwater_consumption": 1},
        }
    },
)
def groundwater_resources():
    return _integ_groundwater_resources()


_integ_groundwater_resources = Integ(
    lambda: groundwater_inputs() - groundwater_consumption(),
    lambda: initial_groundwater(),
    "_integ_groundwater_resources",
)


@component.add(
    name="INFILTRATION FROM AGRICULTURE",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_demand_from_agriculture": 1, "infiltration_rate": 1},
)
def infiltration_from_agriculture():
    return water_demand_from_agriculture() * infiltration_rate()


@component.add(
    name="WATER SECURITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"water_supply": 1, "water_demand": 1},
)
def water_security():
    return if_then_else(water_supply() / water_demand() >= 1, lambda: 1, lambda: 0)


@component.add(
    name="HIGH EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_efficient_water_demand_from_urban():
    return 0.75


@component.add(
    name="SELECT INCENTIVES FOR EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_select_incentives_for_efficient_water_demand_from_urban"
    },
)
def select_incentives_for_efficient_water_demand_from_urban():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_incentives_for_efficient_water_demand_from_urban()


_ext_constant_select_incentives_for_efficient_water_demand_from_urban = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SELECT_INCENTIVES_FOR_EFFICIENT_WATER_DEMAND_FROM_URBAN*",
    {},
    _root,
    {},
    "_ext_constant_select_incentives_for_efficient_water_demand_from_urban",
)


@component.add(
    name="EFFICIENT WATER DEMAND FROM AGRICULTURE OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_efficient_water_demand_from_agriculture_objective"
    },
)
def efficient_water_demand_from_agriculture_objective():
    """
    60% decrease of water demand GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'INITIAL_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE_OBJECTIVE*')
    """
    return _ext_constant_efficient_water_demand_from_agriculture_objective()


_ext_constant_efficient_water_demand_from_agriculture_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_efficient_water_demand_from_agriculture_objective",
)


@component.add(
    name="EFFICIENT WATER DEMAND FROM INDUSTRY OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_efficient_water_demand_from_industry_objective"
    },
)
def efficient_water_demand_from_industry_objective():
    return _ext_constant_efficient_water_demand_from_industry_objective()


_ext_constant_efficient_water_demand_from_industry_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_EFFICIENT_WATER_DEMAND_FROM_INDUSTRY_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_efficient_water_demand_from_industry_objective",
)


@component.add(
    name="DESIRED EFFICIENT WATER DEMAND FROM URBAN OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_desired_efficient_water_demand_from_urban_objective"
    },
)
def desired_efficient_water_demand_from_urban_objective():
    return _ext_constant_desired_efficient_water_demand_from_urban_objective()


_ext_constant_desired_efficient_water_demand_from_urban_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_EFFICIENT_WATER_DEMAND_FROM_URBAN_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_desired_efficient_water_demand_from_urban_objective",
)


@component.add(
    name="INITIAL YEAR EFFICIENT WATER DEMAND FROM INDUSTRY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_efficient_water_demand_from_industry"
    },
)
def initial_year_efficient_water_demand_from_industry():
    return _ext_constant_initial_year_efficient_water_demand_from_industry()


_ext_constant_initial_year_efficient_water_demand_from_industry = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_INDUSTRY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_efficient_water_demand_from_industry",
)


@component.add(
    name="INITIAL YEAR EFFICIENT WATER DEMAND FROM URBAN",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_efficient_water_demand_from_urban"
    },
)
def initial_year_efficient_water_demand_from_urban():
    return _ext_constant_initial_year_efficient_water_demand_from_urban()


_ext_constant_initial_year_efficient_water_demand_from_urban = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_URBAN*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_efficient_water_demand_from_urban",
)


@component.add(
    name="VERY HIGH EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_efficient_water_demand_from_urban():
    return 1


@component.add(
    name="INITIAL WATER DEMAND FROM URBAN",
    units="Hm3 per capita",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_water_demand_from_urban"},
)
def initial_water_demand_from_urban():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'INITIAL_WATER_DEMAND_FROM_URBAN_PER_CAPITA*')
    """
    return _ext_constant_initial_water_demand_from_urban()


_ext_constant_initial_water_demand_from_urban = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_WATER_DEMAND_FROM_URBAN_PER_CAPITA*",
    {},
    _root,
    {},
    "_ext_constant_initial_water_demand_from_urban",
)


@component.add(
    name="MEDIUM EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_efficient_water_demand_from_urban():
    return 0.5


@component.add(
    name="FINAL YEAR EFFICIENT WATER DEMAND FROM URBAN",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_efficient_water_demand_from_urban"
    },
)
def final_year_efficient_water_demand_from_urban():
    return _ext_constant_final_year_efficient_water_demand_from_urban()


_ext_constant_final_year_efficient_water_demand_from_urban = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_URBAN*",
    {},
    _root,
    {},
    "_ext_constant_final_year_efficient_water_demand_from_urban",
)


@component.add(
    name="LOW EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_efficient_water_demand_from_urban():
    return 0.25


@component.add(
    name="SWITCH EFFICIENT WATER DEMAND FROM AGRICULTURE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_efficient_water_demand_from_agriculture"
    },
)
def switch_efficient_water_demand_from_agriculture():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SWITCH_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*')
    """
    return _ext_constant_switch_efficient_water_demand_from_agriculture()


_ext_constant_switch_efficient_water_demand_from_agriculture = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*",
    {},
    _root,
    {},
    "_ext_constant_switch_efficient_water_demand_from_agriculture",
)


@component.add(
    name="FINAL YEAR EFFICIENT WATER DEMAND FROM AGRICULTURE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_efficient_water_demand_from_agriculture"
    },
)
def final_year_efficient_water_demand_from_agriculture():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'FINAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*')
    """
    return _ext_constant_final_year_efficient_water_demand_from_agriculture()


_ext_constant_final_year_efficient_water_demand_from_agriculture = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_efficient_water_demand_from_agriculture",
)


@component.add(
    name="SWITCH EFFICIENT WATER DEMAND FROM INDUSTRY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_efficient_water_demand_from_industry"
    },
)
def switch_efficient_water_demand_from_industry():
    return _ext_constant_switch_efficient_water_demand_from_industry()


_ext_constant_switch_efficient_water_demand_from_industry = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_EFFICIENT_WATER_DEMAND_FROM_INDUSTRY*",
    {},
    _root,
    {},
    "_ext_constant_switch_efficient_water_demand_from_industry",
)


@component.add(
    name="FINAL YEAR EFFICIENT WATER DEMAND FROM INDUSTRY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_efficient_water_demand_from_industry"
    },
)
def final_year_efficient_water_demand_from_industry():
    return _ext_constant_final_year_efficient_water_demand_from_industry()


_ext_constant_final_year_efficient_water_demand_from_industry = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_INDUSTRY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_efficient_water_demand_from_industry",
)


@component.add(
    name="SWITCH EFFICIENT WATER DEMAND FROM URBAN",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_efficient_water_demand_from_urban"
    },
)
def switch_efficient_water_demand_from_urban():
    return _ext_constant_switch_efficient_water_demand_from_urban()


_ext_constant_switch_efficient_water_demand_from_urban = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_EFFICIENT_WATER_DEMAND_FROM_URBAN*",
    {},
    _root,
    {},
    "_ext_constant_switch_efficient_water_demand_from_urban",
)


@component.add(
    name="INITIAL YEAR EFFICIENT WATER DEMAND FROM AGRICULTURE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_efficient_water_demand_from_agriculture"
    },
)
def initial_year_efficient_water_demand_from_agriculture():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'INITIAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*')
    """
    return _ext_constant_initial_year_efficient_water_demand_from_agriculture()


_ext_constant_initial_year_efficient_water_demand_from_agriculture = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_EFFICIENT_WATER_DEMAND_FROM_AGRICULTURE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_efficient_water_demand_from_agriculture",
)


@component.add(
    name="EFFICIENT WATER DEMAND FROM URBAN OBJECTIVE",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_incentives_for_efficient_water_demand_from_urban": 4,
        "low_efficient_water_demand_from_urban": 2,
        "desired_efficient_water_demand_from_urban_objective": 5,
        "high_efficient_water_demand_from_urban": 1,
        "medium_efficient_water_demand_from_urban": 1,
        "very_high_efficient_water_demand_from_urban": 1,
    },
)
def efficient_water_demand_from_urban_objective():
    return if_then_else(
        select_incentives_for_efficient_water_demand_from_urban() == 0,
        lambda: low_efficient_water_demand_from_urban()
        * desired_efficient_water_demand_from_urban_objective(),
        lambda: if_then_else(
            select_incentives_for_efficient_water_demand_from_urban() == 1,
            lambda: medium_efficient_water_demand_from_urban()
            * desired_efficient_water_demand_from_urban_objective(),
            lambda: if_then_else(
                select_incentives_for_efficient_water_demand_from_urban() == 2,
                lambda: high_efficient_water_demand_from_urban()
                * desired_efficient_water_demand_from_urban_objective(),
                lambda: if_then_else(
                    select_incentives_for_efficient_water_demand_from_urban() == 3,
                    lambda: very_high_efficient_water_demand_from_urban()
                    * desired_efficient_water_demand_from_urban_objective(),
                    lambda: low_efficient_water_demand_from_urban()
                    * desired_efficient_water_demand_from_urban_objective(),
                ),
            ),
        ),
    )


@component.add(
    name="FINAL YEAR DESALINATION INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_desalination_increase"},
)
def final_year_desalination_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'FINAL_YEAR_DESALINATION_INCREASE*')
    """
    return _ext_constant_final_year_desalination_increase()


_ext_constant_final_year_desalination_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_DESALINATION_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_desalination_increase",
)


@component.add(
    name="INITIAL YEAR DESALINATION INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_desalination_increase"},
)
def initial_year_desalination_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'INITIAL_YEAR_DESALINATION_INCREASE*')
    """
    return _ext_constant_initial_year_desalination_increase()


_ext_constant_initial_year_desalination_increase = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_DESALINATION_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_desalination_increase",
)


@component.add(
    name="DESALINATION OBJECTIVE",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desalination_objective"},
)
def desalination_objective():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'INITIAL_DESALINATION_OBJECTIVE*')
    """
    return _ext_constant_desalination_objective()


_ext_constant_desalination_objective = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_DESALINATION_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_desalination_objective",
)


@component.add(
    name="SWITCH CS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_cs"},
)
def switch_cs():
    """
    1 = CS1 2 = CS2 3 = CS3 4 = CS4 5 = CS5 GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SWITCH_CS*')
    """
    return _ext_constant_switch_cs()


_ext_constant_switch_cs = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_CS*",
    {},
    _root,
    {},
    "_ext_constant_switch_cs",
)


@component.add(
    name="SWITCH DESALINATION INCREASE POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_desalination_increase_policy"},
)
def switch_desalination_increase_policy():
    """
    0 = OFF 1 = ON GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SWITCH_DESALINATION_INCREASE_POLICY*')
    """
    return _ext_constant_switch_desalination_increase_policy()


_ext_constant_switch_desalination_increase_policy = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SWITCH_DESALINATION_INCREASE_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_switch_desalination_increase_policy",
)


@component.add(
    name="TRANSBOUNDARY OBJECTIVE MURCIA",
    units="Hm3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_murcia_conditions": 4,
        "level_4_murcia": 1,
        "level_3_murcia": 1,
        "level_1_murcia": 1,
        "level_2_murcia": 1,
    },
)
def transboundary_objective_murcia():
    return xr.DataArray(
        if_then_else(
            select_murcia_conditions() == 3,
            lambda: level_4_murcia(),
            lambda: if_then_else(
                select_murcia_conditions() == 2,
                lambda: level_3_murcia(),
                lambda: if_then_else(
                    select_murcia_conditions() == 1,
                    lambda: level_2_murcia(),
                    lambda: if_then_else(
                        select_murcia_conditions() == 0,
                        lambda: level_1_murcia(),
                        lambda: 0,
                    ),
                ),
            ),
        ),
        {"MONTHS": _subscript_dict["MONTHS"]},
        ["MONTHS"],
    )


@component.add(
    name="PRECIPITATION",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="Normal",
    depends_on={
        "select_precipitation_scenario": 2,
        "precipitation_ssp245": 1,
        "precipitation_ssp585": 1,
    },
)
def precipitation():
    return if_then_else(
        select_precipitation_scenario() == 0,
        lambda: precipitation_ssp245(),
        lambda: if_then_else(
            select_precipitation_scenario() == 1,
            lambda: precipitation_ssp585(),
            lambda: xr.DataArray(1, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
        ),
    )


@component.add(
    name="MEAN TEMPERATURE SSP245",
    units="ºc",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_mean_temperature_ssp245",
        "__data__": "_ext_data_mean_temperature_ssp245",
        "time": 1,
    },
)
def mean_temperature_ssp245():
    return _ext_data_mean_temperature_ssp245(time())


_ext_data_mean_temperature_ssp245 = ExtData(
    r"Historical.xlsx",
    "Water system",
    "MEAN_TEMPERATURE_SSP245_TIME",
    "MEAN_TEMPERATURE_SSP245",
    "interpolate",
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_data_mean_temperature_ssp245",
)


@component.add(
    name="MEAN TEMPERATURE SSP585",
    units="ºc",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_mean_temperature_ssp585",
        "__data__": "_ext_data_mean_temperature_ssp585",
        "time": 1,
    },
)
def mean_temperature_ssp585():
    return _ext_data_mean_temperature_ssp585(time())


_ext_data_mean_temperature_ssp585 = ExtData(
    r"Historical.xlsx",
    "Water system",
    "MEAN_TEMPERATURE_SSP585_TIME",
    "MEAN_TEMPERATURE_SSP585",
    "interpolate",
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_data_mean_temperature_ssp585",
)


@component.add(
    name="PRECIPITATION SSP585",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_precipitation_ssp585",
        "__data__": "_ext_data_precipitation_ssp585",
        "time": 1,
    },
)
def precipitation_ssp585():
    return _ext_data_precipitation_ssp585(time())


_ext_data_precipitation_ssp585 = ExtData(
    r"Historical.xlsx",
    "Water system",
    "PRECIPITATION_SSP585_TIME",
    "PRECIPITATION_SSP585",
    None,
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_data_precipitation_ssp585",
)


@component.add(
    name="LEVEL 1 MURCIA",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_level_1_murcia"},
)
def level_1_murcia():
    """
    Nivel 1. Se dará cuando las existencias conjuntas en Entrepeñas y Buendía sean iguales o mayores que 1300 hm³, o cuando las aportaciones conjuntas entrantes a estos embalses en los últimos doce meses sean iguales o mayores que 1200 hm³. En este caso el órgano competente autorizará un trasvase mensual de 60 hm³, hasta el máximo anual antes referido.
    """
    return _ext_constant_level_1_murcia()


_ext_constant_level_1_murcia = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LEVEL_1_MURCIA*",
    {},
    _root,
    {},
    "_ext_constant_level_1_murcia",
)


@component.add(
    name="TRANSBOUNDARY OBJECTIVE OTHER CS",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_transboundary_objective_other_cs"},
)
def transboundary_objective_other_cs():
    return _ext_constant_transboundary_objective_other_cs()


_ext_constant_transboundary_objective_other_cs = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_TRANSBOUNDARY_OBJECTIVE_OTHER_CS*",
    {},
    _root,
    {},
    "_ext_constant_transboundary_objective_other_cs",
)


@component.add(
    name="FINAL YEAR TRANSBOUNDARY POLICY OTHER CS",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_transboundary_policy_other_cs"
    },
)
def final_year_transboundary_policy_other_cs():
    return _ext_constant_final_year_transboundary_policy_other_cs()


_ext_constant_final_year_transboundary_policy_other_cs = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "FINAL_YEAR_TRANSBOUNDARY_POLICY_OTHER_CS*",
    {},
    _root,
    {},
    "_ext_constant_final_year_transboundary_policy_other_cs",
)


@component.add(
    name="INITIAL YEAR TRANSBOUNDARY POLICY OTHER CS",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_transboundary_policy_other_cs"
    },
)
def initial_year_transboundary_policy_other_cs():
    return _ext_constant_initial_year_transboundary_policy_other_cs()


_ext_constant_initial_year_transboundary_policy_other_cs = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "INITIAL_YEAR_TRANSBOUNDARY_POLICY_OTHER_CS*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_transboundary_policy_other_cs",
)


@component.add(
    name="INITIAL DESALINATION",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_desalination"},
)
def initial_desalination():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'Water system', 'INITIAL_DESALINATION*')
    """
    return _ext_constant_initial_desalination()


_ext_constant_initial_desalination = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_DESALINATION*",
    {},
    _root,
    {},
    "_ext_constant_initial_desalination",
)


@component.add(
    name="LEVEL 4 MURCIA",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_level_4_murcia"},
)
def level_4_murcia():
    """
    Nivel 4. Se dará esta situación cuando las existencias conjuntas en Entrepeñas y Buendía sean inferiores a 400 hm³, en cuyo caso no cabe aprobar trasvase alguno.
    """
    return _ext_constant_level_4_murcia()


_ext_constant_level_4_murcia = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LEVEL_4_MURCIA*",
    {},
    _root,
    {},
    "_ext_constant_level_4_murcia",
)


@component.add(
    name="SELECT MURCIA CONDITIONS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_murcia_conditions"},
)
def select_murcia_conditions():
    """
    0 = Level 1 1 = Level 2 2 = Level 3 3 = Level 4 GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SELECT_MURCIA_CONDITIONS*')
    """
    return _ext_constant_select_murcia_conditions()


_ext_constant_select_murcia_conditions = ExtConstant(
    r"Policy.xlsx",
    "Water system",
    "SELECT_MURCIA_CONDITIONS*",
    {},
    _root,
    {},
    "_ext_constant_select_murcia_conditions",
)


@component.add(
    name="LEVEL 3 MURCIA",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_level_3_murcia"},
)
def level_3_murcia():
    """
    Nivel 3. Se dará cuando las existencias conjuntas en Entrepeñas y Buendía no superen, a comienzos de cada mes, los valores mostrados en la tabla (valores en hm³): Oct. Nov. Dic. Ene. Feb. Marz. Abr. May. Jun. Jul. Ago. Sep. 613 609 605 602 597 591 586 645 673 688 661 631 En este nivel, denominado como de situación hidrológica excepcional, el órgano competente podrá autorizar discrecionalmente y de forma motivada un trasvase de hasta 20 hm³/mes.
    """
    return _ext_constant_level_3_murcia()


_ext_constant_level_3_murcia = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LEVEL_3_MURCIA*",
    {},
    _root,
    {},
    "_ext_constant_level_3_murcia",
)


@component.add(
    name="PRECIPITATION SSP245",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_precipitation_ssp245",
        "__data__": "_ext_data_precipitation_ssp245",
        "time": 1,
    },
)
def precipitation_ssp245():
    return _ext_data_precipitation_ssp245(time())


_ext_data_precipitation_ssp245 = ExtData(
    r"Historical.xlsx",
    "Water system",
    "PRECIPITATION_SSP245_TIME",
    "PRECIPITATION_SSP245",
    None,
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_data_precipitation_ssp245",
)


@component.add(
    name="SELECT PRECIPITATION SCENARIO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def select_precipitation_scenario():
    """
    0 = SSP245 1 = SSP585 GET DIRECT CONSTANTS ('Policy.xlsx', 'Water system', 'SELECT_PRECIPITATION_SCENARIO*')
    """
    return 1


@component.add(
    name="LEVEL 2 MURCIA",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_level_2_murcia"},
)
def level_2_murcia():
    """
    Nivel 2. Se dará cuando las existencias conjuntas de Entrepeñas y Buendía sean inferiores a 1300 hm³, sin llegar a los volúmenes previstos en el Nivel 3, y las aportaciones conjuntas registradas en los últimos doce meses sean inferiores a 1200 hm³. En este caso el órgano competente autorizará un trasvase mensual de 27 hm³, hasta el máximo anual antes referido.
    """
    return _ext_constant_level_2_murcia()


_ext_constant_level_2_murcia = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "LEVEL_2_MURCIA*",
    {},
    _root,
    {},
    "_ext_constant_level_2_murcia",
)


@component.add(
    name="DELAY FOREST",
    units="km2",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_forest": 1},
    other_deps={"_delayfixed_delay_forest": {"initial": {}, "step": {"forest": 1}}},
)
def delay_forest():
    return _delayfixed_delay_forest()


_delayfixed_delay_forest = DelayFixed(
    lambda: forest(), lambda: 1, lambda: 0, time_step, "_delayfixed_delay_forest"
)


@component.add(
    name="TBE",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_tbe": 1},
    other_deps={
        "_integ_tbe": {
            "initial": {"initial_ratio_tbe": 1, "initial_forest_area": 1},
            "step": {"variation_tbe": 3, "initial_ratio_tbe": 1},
        }
    },
)
def tbe():
    return _integ_tbe()


_integ_tbe = Integ(
    lambda: if_then_else(
        variation_tbe() < 0,
        lambda: variation_tbe() * initial_ratio_tbe(),
        lambda: variation_tbe(),
    ),
    lambda: initial_ratio_tbe() * initial_forest_area(),
    "_integ_tbe",
)


@component.add(
    name="TBS",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_tbs": 1},
    other_deps={
        "_integ_tbs": {
            "initial": {"initial_ratio_tbs": 1, "initial_forest_area": 1},
            "step": {"variation_tbs": 3, "initial_ratio_tbs": 1},
        }
    },
)
def tbs():
    return _integ_tbs()


_integ_tbs = Integ(
    lambda: if_then_else(
        variation_tbs() < 0,
        lambda: variation_tbs() * initial_ratio_tbs(),
        lambda: variation_tbs(),
    ),
    lambda: initial_ratio_tbs() * initial_forest_area(),
    "_integ_tbs",
)


@component.add(
    name="RATIO TBE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_tbe"},
)
def ratio_tbe():
    return _ext_constant_ratio_tbe()


_ext_constant_ratio_tbe = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "RATIO_TBE*",
    {},
    _root,
    {},
    "_ext_constant_ratio_tbe",
)


@component.add(
    name="RATIO TBS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_tbs"},
)
def ratio_tbs():
    return _ext_constant_ratio_tbs()


_ext_constant_ratio_tbs = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "RATIO_TBS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_tbs",
)


@component.add(
    name="RATIO TNE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_tne"},
)
def ratio_tne():
    return _ext_constant_ratio_tne()


_ext_constant_ratio_tne = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "RATIO_TNE*",
    {},
    _root,
    {},
    "_ext_constant_ratio_tne",
)


@component.add(
    name="VARIATION TNE",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "initial_time": 1, "ratio_tne": 1, "forest_variation": 3},
)
def variation_tne():
    return if_then_else(
        time() == initial_time(),
        lambda: 0,
        lambda: if_then_else(
            forest_variation() < 0,
            lambda: forest_variation(),
            lambda: forest_variation() * ratio_tne(),
        ),
    )


@component.add(
    name="FOREST VARIATION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"forest": 1, "delay_forest": 1},
)
def forest_variation():
    return forest() - delay_forest()


@component.add(
    name="TNE",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_tne": 1},
    other_deps={
        "_integ_tne": {
            "initial": {"initial_ratio_tne": 1, "initial_forest_area": 1},
            "step": {"variation_tne": 3, "initial_ratio_tne": 1},
        }
    },
)
def tne():
    return _integ_tne()


_integ_tne = Integ(
    lambda: if_then_else(
        variation_tne() < 0,
        lambda: variation_tne() * initial_ratio_tne(),
        lambda: variation_tne(),
    ),
    lambda: initial_ratio_tne() * initial_forest_area(),
    "_integ_tne",
)


@component.add(
    name="BNE",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_bne": 1},
    other_deps={
        "_integ_bne": {
            "initial": {"initial_ratio_bne": 1, "initial_forest_area": 1},
            "step": {"variation_bne": 3, "initial_ratio_bne": 1},
        }
    },
)
def bne():
    return _integ_bne()


_integ_bne = Integ(
    lambda: if_then_else(
        variation_bne() < 0,
        lambda: variation_bne() * initial_ratio_bne(),
        lambda: variation_bne(),
    ),
    lambda: initial_ratio_bne() * initial_forest_area(),
    "_integ_bne",
)


@component.add(
    name="BBS",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_bbs": 1},
    other_deps={
        "_integ_bbs": {
            "initial": {"initial_ratio_bbs": 1, "initial_forest_area": 1},
            "step": {"variation_bbs": 3, "initial_ratio_bbs": 1},
        }
    },
)
def bbs():
    return _integ_bbs()


_integ_bbs = Integ(
    lambda: if_then_else(
        variation_bbs() < 0,
        lambda: variation_bbs() * initial_ratio_bbs(),
        lambda: variation_bbs(),
    ),
    lambda: initial_ratio_bbs() * initial_forest_area(),
    "_integ_bbs",
)


@component.add(
    name="VARIATION BBS",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "initial_time": 1, "ratio_bbs": 1, "forest_variation": 3},
)
def variation_bbs():
    return if_then_else(
        time() == initial_time(),
        lambda: 0,
        lambda: if_then_else(
            forest_variation() < 0,
            lambda: forest_variation(),
            lambda: forest_variation() * ratio_bbs(),
        ),
    )


@component.add(
    name="VARIATION TBE",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "initial_time": 1, "forest_variation": 3, "ratio_tbe": 1},
)
def variation_tbe():
    return if_then_else(
        time() == initial_time(),
        lambda: 0,
        lambda: if_then_else(
            forest_variation() < 0,
            lambda: forest_variation(),
            lambda: forest_variation() * ratio_tbe(),
        ),
    )


@component.add(
    name="RATIO BBS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_bbs"},
)
def ratio_bbs():
    return _ext_constant_ratio_bbs()


_ext_constant_ratio_bbs = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "RATIO_BBS*",
    {},
    _root,
    {},
    "_ext_constant_ratio_bbs",
)


@component.add(
    name="VARIATION BNE",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "initial_time": 1, "ratio_bne": 1, "forest_variation": 3},
)
def variation_bne():
    return if_then_else(
        time() == initial_time(),
        lambda: 0,
        lambda: if_then_else(
            forest_variation() < 0,
            lambda: forest_variation(),
            lambda: forest_variation() * ratio_bne(),
        ),
    )


@component.add(
    name="VARIATION TBS",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1, "initial_time": 1, "forest_variation": 3, "ratio_tbs": 1},
)
def variation_tbs():
    return if_then_else(
        time() == initial_time(),
        lambda: 0,
        lambda: if_then_else(
            forest_variation() < 0,
            lambda: forest_variation(),
            lambda: forest_variation() * ratio_tbs(),
        ),
    )


@component.add(
    name="RATIO BNE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ratio_bne"},
)
def ratio_bne():
    return _ext_constant_ratio_bne()


_ext_constant_ratio_bne = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "RATIO_BNE*",
    {},
    _root,
    {},
    "_ext_constant_ratio_bne",
)


@component.add(
    name="INITIAL RATIO IRRIGATED CROP AREA",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_irrigated_crop_area"},
)
def initial_ratio_irrigated_crop_area():
    return _ext_constant_initial_ratio_irrigated_crop_area()


_ext_constant_initial_ratio_irrigated_crop_area = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "RATIO_IRRIGATED_CROP_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_irrigated_crop_area",
)


@component.add(
    name="GROWTH RATE",
    units="Inhabitants",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_growth": 1},
)
def growth_rate():
    return population_growth()


@component.add(
    name="BIOMASS STOCK",
    units="tC",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_biomass_stock": 1},
    other_deps={
        "_integ_biomass_stock": {
            "initial": {"initial_biomass_stock": 1},
            "step": {
                "boreal_broadleaved_summergreen": 1,
                "bbs": 1,
                "bne": 1,
                "boreal_needleleaved_evergreen": 1,
                "temperate_broadleaved_evergreen": 1,
                "tbe": 1,
                "temperate_broadleaved_summergreen": 1,
                "tbs": 1,
                "temperate_needleleaved_evergreen": 1,
                "tne": 1,
                "factor_ha_to_km2": 1,
                "kg_to_tonnes": 1,
                "decay_rate_for_biomass_stock_influended_by_forest_management": 1,
                "forest_mortality": 1,
            },
        }
    },
)
def biomass_stock():
    """
    (INITIAL BIOMASS STOCK+(((BOREAL BROADLEAVED SUMMERGREEN*BBS)+(BOREAL NEEDLELEAVED EVERGREEN*BBE)+(TEMPERATE BROADLEAVED EVERGREEN *TBE)+(TEMPERATE BROADLEAVED SUMMERGREEN*TBS) +(TEMPERATE NEEDLELEAVED EVERGREEN*TNE)-DECAY RATE FOR BIOMASS STOCK)*FACTOR HA TO KM2))*KG TO TONNES
    """
    return _integ_biomass_stock()


_integ_biomass_stock = Integ(
    lambda: (
        (
            boreal_broadleaved_summergreen() * bbs()
            + boreal_needleleaved_evergreen() * bne()
            + temperate_broadleaved_evergreen() * tbe()
            + temperate_broadleaved_summergreen() * tbs()
            + temperate_needleleaved_evergreen() * tne()
        )
        * factor_ha_to_km2()
    )
    * kg_to_tonnes()
    - decay_rate_for_biomass_stock_influended_by_forest_management()
    - forest_mortality(),
    lambda: initial_biomass_stock(),
    "_integ_biomass_stock",
)


@component.add(
    name="DECAY RATE FOR BIOMASS STOCK INFLUENDED BY FOREST MANAGEMENT",
    units="tC",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_time": 1,
        "initial_biomass_stock": 1,
        "decay_coefficient_forest_management": 1,
        "biomass_stock": 1,
        "decay_rate_forest_management": 1,
    },
)
def decay_rate_for_biomass_stock_influended_by_forest_management():
    """
    BIOMASS STOCK*DECAY RATE
    """
    return if_then_else(
        time() == initial_time(),
        lambda: initial_biomass_stock() * decay_coefficient_forest_management(),
        lambda: biomass_stock() * decay_rate_forest_management(),
    )


@component.add(
    name="DECAY RATE FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_forest_management": 3,
        "decay_coefficient_forest_management": 4,
        "time": 3,
        "final_year_forest_management": 2,
        "forest_management_objective": 2,
        "initial_year_forest_management": 3,
    },
)
def decay_rate_forest_management():
    """
    0.01
    """
    return if_then_else(
        switch_forest_management() == 0,
        lambda: decay_coefficient_forest_management(),
        lambda: if_then_else(
            np.logical_and(
                switch_forest_management() == 1,
                time() < initial_year_forest_management(),
            ),
            lambda: decay_coefficient_forest_management(),
            lambda: if_then_else(
                np.logical_and(
                    switch_forest_management() == 1,
                    time() > final_year_forest_management(),
                ),
                lambda: decay_coefficient_forest_management()
                * (1 - forest_management_objective()),
                lambda: decay_coefficient_forest_management()
                * (
                    1
                    - forest_management_objective()
                    * (
                        (time() - initial_year_forest_management())
                        / (
                            final_year_forest_management()
                            - initial_year_forest_management()
                        )
                    )
                ),
            ),
        ),
    )


@component.add(
    name="DECAY COEFFICIENT FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_decay_coefficient_forest_management"},
)
def decay_coefficient_forest_management():
    return _ext_constant_decay_coefficient_forest_management()


_ext_constant_decay_coefficient_forest_management = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "DECAY_COEFFICIENT_FOREST_MANAGEMENT",
    {},
    _root,
    {},
    "_ext_constant_decay_coefficient_forest_management",
)


@component.add(
    name="FINAL YEAR FOREST MANAGEMENT",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_forest_management"},
)
def final_year_forest_management():
    return _ext_constant_final_year_forest_management()


_ext_constant_final_year_forest_management = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "FINAL_YEAR_FOREST_MANAGEMENT*",
    {},
    _root,
    {},
    "_ext_constant_final_year_forest_management",
)


@component.add(
    name="INITIAL FOREST MANAGEMENT OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_forest_management_objective"},
)
def initial_forest_management_objective():
    """
    Reduction of 30% of decay
    """
    return _ext_constant_initial_forest_management_objective()


_ext_constant_initial_forest_management_objective = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "INITIAL_FOREST_MANAGEMENT_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_initial_forest_management_objective",
)


@component.add(
    name="LOW FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_forest_management():
    return 0.25


@component.add(
    name="HIGH FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_forest_management():
    return 0.75


@component.add(
    name="INITIAL YEAR FOREST MANAGEMENT",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_forest_management"},
)
def initial_year_forest_management():
    return _ext_constant_initial_year_forest_management()


_ext_constant_initial_year_forest_management = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "INITIAL_YEAR_FOREST_MANAGEMENT*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_forest_management",
)


@component.add(
    name="MEDIUM FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_forest_management():
    return 0.5


@component.add(
    name="SELECT FOREST MANAGEMENT INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_forest_management_incentives"},
)
def select_forest_management_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_forest_management_incentives()


_ext_constant_select_forest_management_incentives = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "SELECT_FOREST_MANAGEMENT_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_forest_management_incentives",
)


@component.add(
    name="VERY HIGH FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_forest_management():
    return 1


@component.add(
    name="FOREST MANAGEMENT OBJECTIVE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_forest_management_incentives": 4,
        "low_forest_management": 2,
        "initial_forest_management_objective": 5,
        "high_forest_management": 1,
        "very_high_forest_management": 1,
        "medium_forest_management": 1,
    },
)
def forest_management_objective():
    return if_then_else(
        select_forest_management_incentives() == 0,
        lambda: low_forest_management() * initial_forest_management_objective(),
        lambda: if_then_else(
            select_forest_management_incentives() == 1,
            lambda: medium_forest_management() * initial_forest_management_objective(),
            lambda: if_then_else(
                select_forest_management_incentives() == 2,
                lambda: high_forest_management()
                * initial_forest_management_objective(),
                lambda: if_then_else(
                    select_forest_management_incentives() == 3,
                    lambda: very_high_forest_management()
                    * initial_forest_management_objective(),
                    lambda: low_forest_management()
                    * initial_forest_management_objective(),
                ),
            ),
        ),
    )


@component.add(
    name="SWITCH FOREST MANAGEMENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_forest_management"},
)
def switch_forest_management():
    return _ext_constant_switch_forest_management()


_ext_constant_switch_forest_management = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "SWITCH_FOREST_MANAGEMENT*",
    {},
    _root,
    {},
    "_ext_constant_switch_forest_management",
)


@component.add(
    name="INITIAL RATIO BBS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_bbs"},
)
def initial_ratio_bbs():
    return _ext_constant_initial_ratio_bbs()


_ext_constant_initial_ratio_bbs = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_RATIO_BBS",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_bbs",
)


@component.add(
    name="KG TO TONNES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_kg_to_tonnes"},
)
def kg_to_tonnes():
    return _ext_constant_kg_to_tonnes()


_ext_constant_kg_to_tonnes = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "KG_TO_TONNES",
    {},
    _root,
    {},
    "_ext_constant_kg_to_tonnes",
)


@component.add(
    name="TOTAL CARBON STOCK",
    units="tCO2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "biomass_stock": 1,
        "carbon_transformation_constant": 1,
        "factor_ha_to_km2": 1,
        "kg_to_tonnes": 1,
    },
)
def total_carbon_stock():
    return (
        biomass_stock() * carbon_transformation_constant() * factor_ha_to_km2()
    ) * kg_to_tonnes()


@component.add(
    name="INITIAL RATIO TBE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_tbe"},
)
def initial_ratio_tbe():
    return _ext_constant_initial_ratio_tbe()


_ext_constant_initial_ratio_tbe = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_RATIO_TBE",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_tbe",
)


@component.add(
    name="INITIAL RATIO TBS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_tbs"},
)
def initial_ratio_tbs():
    return _ext_constant_initial_ratio_tbs()


_ext_constant_initial_ratio_tbs = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_RATIO_TBS",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_tbs",
)


@component.add(
    name="INITIAL RATIO TNE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_tne"},
)
def initial_ratio_tne():
    return _ext_constant_initial_ratio_tne()


_ext_constant_initial_ratio_tne = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_RATIO_TNE",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_tne",
)


@component.add(
    name="FACTOR HA TO KM2",
    units="ha",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_factor_ha_to_km2"},
)
def factor_ha_to_km2():
    return _ext_constant_factor_ha_to_km2()


_ext_constant_factor_ha_to_km2 = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "FACTOR_HA_TO_KM2",
    {},
    _root,
    {},
    "_ext_constant_factor_ha_to_km2",
)


@component.add(
    name="REFERENCE TEMPERATURE 2017",
    units="ºc",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_reference_temperature_2017"},
)
def reference_temperature_2017():
    return _ext_constant_reference_temperature_2017()


_ext_constant_reference_temperature_2017 = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "REFERENCE_TEMPERATURE_2017",
    {},
    _root,
    {},
    "_ext_constant_reference_temperature_2017",
)


@component.add(
    name="TEMPERATE BROADLEAVED EVERGREEN",
    units="kgC/ha·year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature_increase": 2, "co2_concentration": 2},
)
def temperate_broadleaved_evergreen():
    return (
        0.28639
        - 0.03491 * temperature_increase()
        - 0.0007 * float(np.power(temperature_increase(), 2))
    ) + 0.07744 * (co2_concentration() / (co2_concentration() + 98.501))


@component.add(
    name="TEMPERATE BROADLEAVED SUMMERGREEN",
    units="kgC/ha·year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature_increase": 2, "co2_concentration": 2},
)
def temperate_broadleaved_summergreen():
    return (
        0.08529
        + 0.01341 * temperature_increase()
        - 0.00161 * float(np.power(temperature_increase(), 2))
    ) + 0.24635 * (co2_concentration() / (co2_concentration() + 768.02))


@component.add(
    name="TEMPERATE NEEDLELEAVED EVERGREEN",
    units="kgC/ha·year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature_increase": 2, "co2_concentration": 2},
)
def temperate_needleleaved_evergreen():
    return (
        0.22092
        - 0.01822 * temperature_increase()
        - 0.00094 * float(np.power(temperature_increase(), 2))
    ) + 0.35027 * (co2_concentration() / (co2_concentration() + 1053.22))


@component.add(
    name="TEMPERATURE INCREASE",
    units="ºc",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "future_temperature": 2,
        "reference_temperature_2017": 1,
        "future_temperature_delay": 1,
    },
)
def temperature_increase():
    return if_then_else(
        time() == 2018,
        lambda: future_temperature() - reference_temperature_2017(),
        lambda: future_temperature() - future_temperature_delay(),
    )


@component.add(
    name="CO2 FUTURE DATA",
    units="ppm",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_co2_future_data",
        "__data__": "_ext_data_co2_future_data",
        "time": 1,
    },
)
def co2_future_data():
    return _ext_data_co2_future_data(time())


_ext_data_co2_future_data = ExtData(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "CO2_TIME",
    "CO2_FUTURE_DATA",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_co2_future_data",
)


@component.add(
    name="REFERENCE CO2 2017",
    units="ppm",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_reference_co2_2017"},
)
def reference_co2_2017():
    return _ext_constant_reference_co2_2017()


_ext_constant_reference_co2_2017 = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "REFERENCE_CO2_2017",
    {},
    _root,
    {},
    "_ext_constant_reference_co2_2017",
)


@component.add(
    name="SELECT TEMPERATURE SCENARIO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_temperature_scenario"},
)
def select_temperature_scenario():
    """
    SSP245=0 SSP585=1
    """
    return _ext_constant_select_temperature_scenario()


_ext_constant_select_temperature_scenario = ExtConstant(
    r"Policy.xlsx",
    "Carbon stock biomass",
    "SELECT_TEMPERATURE_SCENARIO*",
    {},
    _root,
    {},
    "_ext_constant_select_temperature_scenario",
)


@component.add(
    name="BOREAL BROADLEAVED SUMMERGREEN",
    units="kgC/ha·year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature_increase": 2, "co2_concentration": 2},
)
def boreal_broadleaved_summergreen():
    return (
        0.24572
        - 0.06422 * temperature_increase()
        - 0.00477 * float(np.power(temperature_increase(), 2))
    ) + 0.0235 * (co2_concentration() / (co2_concentration() + 70.6306))


@component.add(
    name="BOREAL NEEDLELEAVED EVERGREEN",
    units="kgC/ha·year",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature_increase": 2, "co2_concentration": 2},
)
def boreal_needleleaved_evergreen():
    return (
        0.34654
        - 0.08512 * temperature_increase()
        - 0.00558 * float(np.power(temperature_increase(), 2))
    ) + 0.02119 * (co2_concentration() / (co2_concentration() + 16.6054))


@component.add(
    name="INITIAL RATIO BNE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_ratio_bne"},
)
def initial_ratio_bne():
    return _ext_constant_initial_ratio_bne()


_ext_constant_initial_ratio_bne = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_RATIO_BNE",
    {},
    _root,
    {},
    "_ext_constant_initial_ratio_bne",
)


@component.add(
    name="INITIAL BIOMASS STOCK",
    units="tC",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_biomass_stock"},
)
def initial_biomass_stock():
    return _ext_constant_initial_biomass_stock()


_ext_constant_initial_biomass_stock = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "INITIAL_BIOMASS_STOCK",
    {},
    _root,
    {},
    "_ext_constant_initial_biomass_stock",
)


@component.add(
    name="FUTURE TEMPERATURE",
    units="ºc",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_temperature_scenario": 2,
        "temperature_ssp245": 1,
        "temperature_ssp585": 1,
    },
)
def future_temperature():
    return if_then_else(
        select_temperature_scenario() == 0,
        lambda: temperature_ssp245(),
        lambda: if_then_else(
            select_temperature_scenario() == 1, lambda: temperature_ssp585(), lambda: 0
        ),
    )


@component.add(
    name="CARBON TRANSFORMATION CONSTANT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_carbon_transformation_constant"},
)
def carbon_transformation_constant():
    return _ext_constant_carbon_transformation_constant()


_ext_constant_carbon_transformation_constant = ExtConstant(
    r"Historical.xlsx",
    "Carbon stock biomass",
    "CARBON_TRANSFORMATION_CONSTANT",
    {},
    _root,
    {},
    "_ext_constant_carbon_transformation_constant",
)


@component.add(
    name="FUTURE CO2 DELAY",
    units="ppm",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_future_co2_delay": 1},
    other_deps={
        "_delayfixed_future_co2_delay": {"initial": {}, "step": {"co2_future_data": 1}}
    },
)
def future_co2_delay():
    return _delayfixed_future_co2_delay()


_delayfixed_future_co2_delay = DelayFixed(
    lambda: co2_future_data(),
    lambda: 1,
    lambda: 1,
    time_step,
    "_delayfixed_future_co2_delay",
)


@component.add(
    name="FUTURE TEMPERATURE DELAY",
    units="ºc",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_future_temperature_delay": 1},
    other_deps={
        "_delayfixed_future_temperature_delay": {
            "initial": {},
            "step": {"future_temperature": 1},
        }
    },
)
def future_temperature_delay():
    return _delayfixed_future_temperature_delay()


_delayfixed_future_temperature_delay = DelayFixed(
    lambda: future_temperature(),
    lambda: 1,
    lambda: 1,
    time_step,
    "_delayfixed_future_temperature_delay",
)


@component.add(
    name="CO2 CONCENTRATION",
    units="ppm",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "co2_future_data": 2,
        "reference_co2_2017": 1,
        "future_co2_delay": 1,
    },
)
def co2_concentration():
    return if_then_else(
        time() == 2018,
        lambda: co2_future_data() - reference_co2_2017(),
        lambda: co2_future_data() - future_co2_delay(),
    )


@component.add(
    name="SELECT EDUCATION AND TRAINING INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_select_education_and_training_incentives"
    },
)
def select_education_and_training_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_education_and_training_incentives()


_ext_constant_select_education_and_training_incentives = ExtConstant(
    r"Policy.xlsx",
    "Population system",
    "SELECT_EDUCATION_AND_TRAINING_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_education_and_training_incentives",
)


@component.add(
    name="FINAL YEAR INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_investment_in_tourism_adaptation_and_resilience"
    },
)
def final_year_investment_in_tourism_adaptation_and_resilience():
    return _ext_constant_final_year_investment_in_tourism_adaptation_and_resilience()


_ext_constant_final_year_investment_in_tourism_adaptation_and_resilience = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "FINAL_YEAR_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_investment_in_tourism_adaptation_and_resilience",
)


@component.add(
    name="SELECT INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_select_investment_in_tourism_adaptation_and_resilience"
    },
)
def select_investment_in_tourism_adaptation_and_resilience():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High GET DIRECT CONSTANTS ('Policy.xlsx', 'Tourism system', 'SELECT_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*')
    """
    return _ext_constant_select_investment_in_tourism_adaptation_and_resilience()


_ext_constant_select_investment_in_tourism_adaptation_and_resilience = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "SELECT_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*",
    {},
    _root,
    {},
    "_ext_constant_select_investment_in_tourism_adaptation_and_resilience",
)


@component.add(
    name="EDUCATION AND TRAINING OBJECTIVE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_education_and_training_incentives": 4,
        "desired_education_and_training_objective": 5,
        "low_education_and_training": 2,
        "very_high_education_and_training": 1,
        "medium_education_and_training": 1,
        "high_education_and_training": 1,
    },
)
def education_and_training_objective():
    return if_then_else(
        select_education_and_training_incentives() == 0,
        lambda: low_education_and_training()
        * desired_education_and_training_objective(),
        lambda: if_then_else(
            select_education_and_training_incentives() == 1,
            lambda: medium_education_and_training()
            * desired_education_and_training_objective(),
            lambda: if_then_else(
                select_education_and_training_incentives() == 2,
                lambda: high_education_and_training()
                * desired_education_and_training_objective(),
                lambda: if_then_else(
                    select_education_and_training_incentives() == 3,
                    lambda: very_high_education_and_training()
                    * desired_education_and_training_objective(),
                    lambda: low_education_and_training()
                    * desired_education_and_training_objective(),
                ),
            ),
        ),
    )


@component.add(
    name="DESIRED EDUCATION AND TRAINING OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_desired_education_and_training_objective"
    },
)
def desired_education_and_training_objective():
    return _ext_constant_desired_education_and_training_objective()


_ext_constant_desired_education_and_training_objective = ExtConstant(
    r"Policy.xlsx",
    "Population system",
    "INITIAL_EDUCATION_AND_TRAINING_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_desired_education_and_training_objective",
)


@component.add(
    name="LOW EDUCATION AND TRAINING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_education_and_training():
    return 0.25


@component.add(
    name="INITIAL YEAR INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_investment_in_tourism_adaptation_and_resilience"
    },
)
def initial_year_investment_in_tourism_adaptation_and_resilience():
    return _ext_constant_initial_year_investment_in_tourism_adaptation_and_resilience()


_ext_constant_initial_year_investment_in_tourism_adaptation_and_resilience = (
    ExtConstant(
        r"Policy.xlsx",
        "Tourism system",
        "INITIAL_YEAR_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*",
        {},
        _root,
        {},
        "_ext_constant_initial_year_investment_in_tourism_adaptation_and_resilience",
    )
)


@component.add(
    name="LOW INVESTMENT IN TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_investment_in_tourism():
    return 0.25


@component.add(
    name="HIGH INVESTMENT IN TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_investment_in_tourism():
    return 0.75


@component.add(
    name="INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_investment_in_tourism_adaptation_and_resilience": 3,
        "final_year_investment_in_tourism_adaptation_and_resilience": 2,
        "time": 3,
        "investment_in_tourism_adaptation_and_resilience_objective": 2,
        "initial_year_investment_in_tourism_adaptation_and_resilience": 3,
    },
)
def investment_in_tourism_adaptation_and_resilience():
    return if_then_else(
        switch_investment_in_tourism_adaptation_and_resilience() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_investment_in_tourism_adaptation_and_resilience() == 1,
                time() < initial_year_investment_in_tourism_adaptation_and_resilience(),
            ),
            lambda: 0,
            lambda: if_then_else(
                np.logical_and(
                    switch_investment_in_tourism_adaptation_and_resilience() == 1,
                    time()
                    > final_year_investment_in_tourism_adaptation_and_resilience(),
                ),
                lambda: investment_in_tourism_adaptation_and_resilience_objective(),
                lambda: investment_in_tourism_adaptation_and_resilience_objective()
                * (
                    (
                        time()
                        - initial_year_investment_in_tourism_adaptation_and_resilience()
                    )
                    / (
                        final_year_investment_in_tourism_adaptation_and_resilience()
                        - initial_year_investment_in_tourism_adaptation_and_resilience()
                    )
                ),
            ),
        ),
    )


@component.add(
    name="SWITCH INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_investment_in_tourism_adaptation_and_resilience"
    },
)
def switch_investment_in_tourism_adaptation_and_resilience():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Tourism system', 'SWITCH_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*')
    """
    return _ext_constant_switch_investment_in_tourism_adaptation_and_resilience()


_ext_constant_switch_investment_in_tourism_adaptation_and_resilience = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "SWITCH_INVESTMENT_IN_TOURISM_ADAPTATION_AND_RESILIENCE*",
    {},
    _root,
    {},
    "_ext_constant_switch_investment_in_tourism_adaptation_and_resilience",
)


@component.add(
    name="INITIAL YEAR EDUCATION AND TRAINING INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_education_and_training_increase"
    },
)
def initial_year_education_and_training_increase():
    return _ext_constant_initial_year_education_and_training_increase()


_ext_constant_initial_year_education_and_training_increase = ExtConstant(
    r"Policy.xlsx",
    "Population system",
    "INITIAL_YEAR_EDUCATION_AND_TRAINING_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_education_and_training_increase",
)


@component.add(
    name="MEDIUM EDUCATION AND TRAINING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_education_and_training():
    return 0.5


@component.add(
    name="MEDIUM INVESTMENT IN TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_investment_in_tourism():
    return 0.5


@component.add(
    name="HIGH EDUCATION AND TRAINING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_education_and_training():
    return 0.75


@component.add(
    name="VERY HIGH INVESTMENT IN TOURISM",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_investment_in_tourism():
    return 1


@component.add(
    name="VERY HIGH EDUCATION AND TRAINING",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_education_and_training():
    return 1


@component.add(
    name="SWITCH EDUCATION AND TRAINING INCREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_education_and_training_increase"},
)
def switch_education_and_training_increase():
    return _ext_constant_switch_education_and_training_increase()


_ext_constant_switch_education_and_training_increase = ExtConstant(
    r"Policy.xlsx",
    "Population system",
    "SWITCH_EDUCATION_AND_TRAINING_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_switch_education_and_training_increase",
)


@component.add(
    name="INVESTMENT IN TOURISM ADAPTATION AND RESILIENCE OBJECTIVE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_investment_in_tourism_adaptation_and_resilience": 4,
        "low_investment_in_tourism": 2,
        "medium_investment_in_tourism": 1,
        "high_investment_in_tourism": 1,
        "very_high_investment_in_tourism": 1,
    },
)
def investment_in_tourism_adaptation_and_resilience_objective():
    return if_then_else(
        select_investment_in_tourism_adaptation_and_resilience() == 0,
        lambda: low_investment_in_tourism(),
        lambda: if_then_else(
            select_investment_in_tourism_adaptation_and_resilience() == 1,
            lambda: medium_investment_in_tourism(),
            lambda: if_then_else(
                select_investment_in_tourism_adaptation_and_resilience() == 2,
                lambda: high_investment_in_tourism(),
                lambda: if_then_else(
                    select_investment_in_tourism_adaptation_and_resilience() == 3,
                    lambda: very_high_investment_in_tourism(),
                    lambda: low_investment_in_tourism(),
                ),
            ),
        ),
    )


@component.add(
    name="VULNERABILITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tourism_sensitivity": 2, "adaptive_capacity": 2},
)
def vulnerability():
    return if_then_else(
        tourism_sensitivity() * (1 - adaptive_capacity()) < 0,
        lambda: 0,
        lambda: tourism_sensitivity() * (1 - adaptive_capacity()),
    )


@component.add(
    name="VARIABILITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_variability"},
)
def variability():
    return _ext_constant_variability()


_ext_constant_variability = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "VARIABILITY*",
    {},
    _root,
    {},
    "_ext_constant_variability",
)


@component.add(
    name="TOURISM EXPOSURE AND PROBABILITIES",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "tci_weight_in_exposure": 1,
        "mean_tci": 1,
        "weight_of_extreme_events_in_exposure": 1,
        "number_of_extreme_events": 1,
    },
)
def tourism_exposure_and_probabilities():
    return tci_weight_in_exposure() * (
        (100 - mean_tci()) / 100
    ) + weight_of_extreme_events_in_exposure() * (number_of_extreme_events() / 365)


@component.add(
    name="TOURISM SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"climate_dependancy": 1, "impact_perception": 1, "variability": 1},
)
def tourism_sensitivity():
    return (climate_dependancy() + impact_perception() + variability()) / 3


@component.add(
    name="CLIMATE DEPENDANCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_climate_dependancy"},
)
def climate_dependancy():
    return _ext_constant_climate_dependancy()


_ext_constant_climate_dependancy = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "CLIMATE_DEPENDANCY*",
    {},
    _root,
    {},
    "_ext_constant_climate_dependancy",
)


@component.add(
    name="IMPACT PERCEPTION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_impact_perception"},
)
def impact_perception():
    return _ext_constant_impact_perception()


_ext_constant_impact_perception = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "IMPACT_PERCEPTION*",
    {},
    _root,
    {},
    "_ext_constant_impact_perception",
)


@component.add(
    name="TOTAL IMPACT",
    units="Million Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"tourism_income": 1, "impact_since_reference_year": 1},
)
def total_impact():
    return tourism_income() - impact_since_reference_year()


@component.add(
    name="IMPACT SINCE REFERENCE YEAR",
    units="Million Euros",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_impact_since_reference_year": 1},
    other_deps={
        "_delayfixed_impact_since_reference_year": {
            "initial": {},
            "step": {"tourism_income": 1},
        }
    },
)
def impact_since_reference_year():
    return _delayfixed_impact_since_reference_year()


_delayfixed_impact_since_reference_year = DelayFixed(
    lambda: tourism_income(),
    lambda: 1,
    lambda: 1,
    time_step,
    "_delayfixed_impact_since_reference_year",
)


@component.add(
    name="NEW ROOMS",
    units="Rooms",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_rooms": 2, "rooms_delay": 2},
)
def new_rooms():
    return if_then_else(
        total_rooms() - rooms_delay() > 0,
        lambda: total_rooms() - rooms_delay(),
        lambda: 0,
    )


@component.add(
    name="NUMBER OF ROOMS",
    units="Rooms",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_number_of_rooms"},
)
def number_of_rooms():
    return _ext_constant_number_of_rooms()


_ext_constant_number_of_rooms = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "NUMBER_OF_ROOMS*",
    {},
    _root,
    {},
    "_ext_constant_number_of_rooms",
)


@component.add(
    name="TCI INDEX SSP585",
    subscripts=["MONTHS"],
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_tci_index_ssp585",
        "__data__": "_ext_data_tci_index_ssp585",
        "time": 1,
    },
)
def tci_index_ssp585():
    return _ext_data_tci_index_ssp585(time())


_ext_data_tci_index_ssp585 = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "TCI_INDEX_SSP585_TIME",
    "TCI_INDEX_SSP585",
    "interpolate",
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_data_tci_index_ssp585",
)


@component.add(
    name="DISUSE",
    units="Infrastructure",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "building_deterioration_rate": 1,
        "infrastructure_capacity": 1,
        "number_of_extreme_events": 1,
        "effects_of_extreme_events": 1,
    },
)
def disuse():
    return (
        building_deterioration_rate() * infrastructure_capacity()
        + effects_of_extreme_events() * number_of_extreme_events()
    )


@component.add(
    name="TCI WEIGHT IN EXPOSURE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_tci_weight_in_exposure"},
)
def tci_weight_in_exposure():
    return _ext_constant_tci_weight_in_exposure()


_ext_constant_tci_weight_in_exposure = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "TCI_WEIGHT_IN_EXPOSURE*",
    {},
    _root,
    {},
    "_ext_constant_tci_weight_in_exposure",
)


@component.add(
    name="BUILDING DETERIORATION RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_building_deterioration_rate"},
)
def building_deterioration_rate():
    return _ext_constant_building_deterioration_rate()


_ext_constant_building_deterioration_rate = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "BUILDING_DETERIORATION_RATE*",
    {},
    _root,
    {},
    "_ext_constant_building_deterioration_rate",
)


@component.add(
    name="INFRASTRUCTURE CAPACITY",
    units="Infrastructure",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_infrastructure_capacity": 1},
    other_deps={
        "_integ_infrastructure_capacity": {
            "initial": {"capacity": 1},
            "step": {"new_infrastructure": 1, "disuse": 1},
        }
    },
)
def infrastructure_capacity():
    return _integ_infrastructure_capacity()


_integ_infrastructure_capacity = Integ(
    lambda: new_infrastructure() - disuse(),
    lambda: capacity(),
    "_integ_infrastructure_capacity",
)


@component.add(
    name="INITIAL ATTRACTIVENESS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_attractiveness"},
)
def initial_attractiveness():
    return _ext_constant_initial_attractiveness()


_ext_constant_initial_attractiveness = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "TCI_WEIGHT_IN_EXPOSURE*",
    {},
    _root,
    {},
    "_ext_constant_initial_attractiveness",
)


@component.add(
    name="NUMBER OF COLD WAVES",
    units="Days",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_number_of_cold_waves",
        "__data__": "_ext_data_number_of_cold_waves",
        "time": 1,
    },
)
def number_of_cold_waves():
    return _ext_data_number_of_cold_waves(time())


_ext_data_number_of_cold_waves = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "SSP585_HAZARDS_TIME",
    "NUMBER_OF_COLD_WAVES",
    None,
    {},
    _root,
    {},
    "_ext_data_number_of_cold_waves",
)


@component.add(
    name="NUMBER OF DAYS WITH EXTREME PRECIPITATION",
    units="Days",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_number_of_days_with_extreme_precipitation",
        "__data__": "_ext_data_number_of_days_with_extreme_precipitation",
        "time": 1,
    },
)
def number_of_days_with_extreme_precipitation():
    return _ext_data_number_of_days_with_extreme_precipitation(time())


_ext_data_number_of_days_with_extreme_precipitation = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "SSP585_HAZARDS_TIME",
    "NUMBER_OF_DAYS_WITH_EXTREME_PRECIPITATION",
    None,
    {},
    _root,
    {},
    "_ext_data_number_of_days_with_extreme_precipitation",
)


@component.add(
    name="EFFECTS OF EXTREME EVENTS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_effects_of_extreme_events"},
)
def effects_of_extreme_events():
    return _ext_constant_effects_of_extreme_events()


_ext_constant_effects_of_extreme_events = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "EFFECTS_OF_EXTREME_EVENTS*",
    {},
    _root,
    {},
    "_ext_constant_effects_of_extreme_events",
)


@component.add(
    name="NUMBER OF EXTREME EVENTS",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "number_of_cold_waves": 1,
        "number_of_days_with_extreme_precipitation": 1,
        "number_of_days_with_high_winds": 1,
        "number_of_heatwaves": 1,
        "number_of_very_hot_days": 1,
    },
)
def number_of_extreme_events():
    return (
        number_of_cold_waves()
        + number_of_days_with_extreme_precipitation()
        + number_of_days_with_high_winds()
        + number_of_heatwaves()
        + number_of_very_hot_days()
    )


@component.add(
    name="NUMBER OF HEATWAVES",
    units="Number",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_number_of_heatwaves",
        "__data__": "_ext_data_number_of_heatwaves",
        "time": 1,
    },
)
def number_of_heatwaves():
    return _ext_data_number_of_heatwaves(time())


_ext_data_number_of_heatwaves = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "SSP585_HAZARDS_TIME",
    "NUMBER_OF_HEATWAVES",
    None,
    {},
    _root,
    {},
    "_ext_data_number_of_heatwaves",
)


@component.add(
    name="NUMBER OF DAYS WITH HIGH WINDS",
    units="Days",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_number_of_days_with_high_winds",
        "__data__": "_ext_data_number_of_days_with_high_winds",
        "time": 1,
    },
)
def number_of_days_with_high_winds():
    return _ext_data_number_of_days_with_high_winds(time())


_ext_data_number_of_days_with_high_winds = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "SSP585_HAZARDS_TIME",
    "NUMBER_OF_DAYS_WITH_HIGH_WINDS",
    None,
    {},
    _root,
    {},
    "_ext_data_number_of_days_with_high_winds",
)


@component.add(
    name="HISTORICAL EXPENDITURE FOR INTERNATIONAL VISITORS",
    units="Euros",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_historical_expenditure_for_international_visitors"
    },
)
def historical_expenditure_for_international_visitors():
    return _ext_constant_historical_expenditure_for_international_visitors()


_ext_constant_historical_expenditure_for_international_visitors = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "HISTORICAL_EXPENDITURE_FOR_INTERNATIONAL_VISITORS*",
    {},
    _root,
    {},
    "_ext_constant_historical_expenditure_for_international_visitors",
)


@component.add(
    name="ATTRACTIVENESS FOR TOURISM",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_attractiveness_for_tourism": 1},
    other_deps={
        "_integ_attractiveness_for_tourism": {
            "initial": {"initial_attractiveness": 1},
            "step": {
                "attractiveness_for_tourism": 1,
                "change_in_tourism_attraction": 1,
            },
        }
    },
)
def attractiveness_for_tourism():
    return _integ_attractiveness_for_tourism()


_integ_attractiveness_for_tourism = Integ(
    lambda: attractiveness_for_tourism() * change_in_tourism_attraction(),
    lambda: initial_attractiveness(),
    "_integ_attractiveness_for_tourism",
)


@component.add(
    name="NATIONAL EXPENDITURE",
    units="Euros",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_national_expenditure": 1},
    other_deps={
        "_integ_national_expenditure": {
            "initial": {"historical_expenditure_for_national_visitors": 1},
            "step": {"national_expenditure_increase": 1},
        }
    },
)
def national_expenditure():
    return _integ_national_expenditure()


_integ_national_expenditure = Integ(
    lambda: national_expenditure_increase(),
    lambda: historical_expenditure_for_national_visitors(),
    "_integ_national_expenditure",
)


@component.add(
    name="AVERAGE EXPENDITURE INCREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_average_expenditure_increase"},
)
def average_expenditure_increase():
    return _ext_constant_average_expenditure_increase()


_ext_constant_average_expenditure_increase = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "AVERAGE_EXPENDITURE_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_average_expenditure_increase",
)


@component.add(
    name="TOURISM FACTOR",
    units="Tourist per room and day",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_tourism_factor"},
)
def tourism_factor():
    return _ext_constant_tourism_factor()


_ext_constant_tourism_factor = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "TOURISM_FACTOR*",
    {},
    _root,
    {},
    "_ext_constant_tourism_factor",
)


@component.add(
    name="WEIGHT OF EXTREME EVENTS IN EXPOSURE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_weight_of_extreme_events_in_exposure"},
)
def weight_of_extreme_events_in_exposure():
    return _ext_constant_weight_of_extreme_events_in_exposure()


_ext_constant_weight_of_extreme_events_in_exposure = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "WEIGHT_OF_EXTREME_EVENTS_IN_EXPOSURE*",
    {},
    _root,
    {},
    "_ext_constant_weight_of_extreme_events_in_exposure",
)


@component.add(
    name="ROOMS DELAY",
    units="Rooms",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_rooms_delay": 1},
    other_deps={
        "_delayfixed_rooms_delay": {
            "initial": {"total_rooms": 1},
            "step": {"total_rooms": 1},
        }
    },
)
def rooms_delay():
    return _delayfixed_rooms_delay()


_delayfixed_rooms_delay = DelayFixed(
    lambda: total_rooms(),
    lambda: 1,
    lambda: total_rooms(),
    time_step,
    "_delayfixed_rooms_delay",
)


@component.add(
    name="CAPACITY",
    units="Tourists",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_rooms": 1, "tourism_factor": 1, "mean_factor_of_use": 1},
)
def capacity():
    return ((number_of_rooms() * tourism_factor()) / mean_factor_of_use()) * 365


@component.add(
    name="NATIONAL EXPENDITURE INCREASE",
    units="Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"national_expenditure": 2, "average_expenditure_increase": 1},
)
def national_expenditure_increase():
    """
    (NATIONAL EXPENDITURE * AVERAGE EXPENDITURE INCREASE) * (1 - (NATIONAL EXPENDITURE / 330)) (NATIONAL EXPENDITURE * AVERAGE EXPENDITURE INCREASE) * (1 - (NATIONAL EXPENDITURE / 680))
    """
    return (national_expenditure() * average_expenditure_increase()) * (
        1 - national_expenditure() / 900
    )


@component.add(
    name="CHANGE IN TOURISM ATTRACTION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"mean_tci": 1, "mean_historical_tci": 1},
)
def change_in_tourism_attraction():
    return 1 - mean_tci() / mean_historical_tci()


@component.add(
    name="HISTORICAL EXPENDITURE FOR NATIONAL VISITORS",
    units="Euros",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_historical_expenditure_for_national_visitors"
    },
)
def historical_expenditure_for_national_visitors():
    return _ext_constant_historical_expenditure_for_national_visitors()


_ext_constant_historical_expenditure_for_national_visitors = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "HISTORICAL_EXPENDITURE_FOR_NATIONAL_VISITORS*",
    {},
    _root,
    {},
    "_ext_constant_historical_expenditure_for_national_visitors",
)


@component.add(
    name="NUMBER OF VERY HOT DAYS",
    units="Days",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_number_of_very_hot_days",
        "__data__": "_ext_data_number_of_very_hot_days",
        "time": 1,
    },
)
def number_of_very_hot_days():
    return _ext_data_number_of_very_hot_days(time())


_ext_data_number_of_very_hot_days = ExtData(
    r"Historical.xlsx",
    "Tourism system",
    "SSP585_HAZARDS_TIME",
    "NUMBER_OF_VERY_HOT_DAYS",
    None,
    {},
    _root,
    {},
    "_ext_data_number_of_very_hot_days",
)


@component.add(
    name="MEAN FACTOR OF USE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_mean_factor_of_use"},
)
def mean_factor_of_use():
    return _ext_constant_mean_factor_of_use()


_ext_constant_mean_factor_of_use = ExtConstant(
    r"Historical.xlsx",
    "Tourism system",
    "MEAN_FACTOR_OF_USE*",
    {},
    _root,
    {},
    "_ext_constant_mean_factor_of_use",
)


@component.add(
    name="INVESTMENT IN TOURISTIC INFRASTRUCTURE OBJECTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_investment_in_touristic_infrastructure_objectives"
    },
)
def investment_in_touristic_infrastructure_objectives():
    return _ext_constant_investment_in_touristic_infrastructure_objectives()


_ext_constant_investment_in_touristic_infrastructure_objectives = ExtConstant(
    r"Policy.xlsx",
    "Tourism system",
    "INVESTMENT_IN_INFRASTRUCTURE_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_investment_in_touristic_infrastructure_objectives",
)


@component.add(
    name="INTERNATIONAL EXPENDITURE",
    units="Euros",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_international_expenditure": 1},
    other_deps={
        "_integ_international_expenditure": {
            "initial": {"historical_expenditure_for_international_visitors": 1},
            "step": {"international_expenditure_increase": 1},
        }
    },
)
def international_expenditure():
    return _integ_international_expenditure()


_integ_international_expenditure = Integ(
    lambda: international_expenditure_increase(),
    lambda: historical_expenditure_for_international_visitors(),
    "_integ_international_expenditure",
)


@component.add(
    name="TOTAL ROOMS",
    units="Rooms",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "infrastructure_capacity": 1,
        "mean_factor_of_use": 1,
        "tourism_factor": 1,
        "number_of_rooms": 1,
    },
)
def total_rooms():
    return integer(
        (infrastructure_capacity() * mean_factor_of_use()) / (tourism_factor() * 365)
        - number_of_rooms()
    )


@component.add(
    name="INTERNATIONAL EXPENDITURE INCREASE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"international_expenditure": 2, "average_expenditure_increase": 1},
)
def international_expenditure_increase():
    return (international_expenditure() * average_expenditure_increase()) * (
        1 - international_expenditure() / 1600
    )


@component.add(
    name="POPULATION DENSITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"cs_population": 1, "total_cs_area": 1},
)
def population_density():
    return cs_population() / total_cs_area()


@component.add(
    name="BIRTH RATE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"births": 1, "cs_population": 1},
)
def birth_rate():
    return births() / cs_population()


@component.add(
    name="RATIO ENERGY POVERTY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_in_energy_poverty": 1, "cs_population": 1},
)
def ratio_energy_poverty():
    return population_in_energy_poverty() / cs_population()


@component.add(
    name='"RATIO OF POPULATION <5"',
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_lower_5": 1, "cs_population": 1},
)
def ratio_of_population_5():
    return population_lower_5() / cs_population()


@component.add(
    name='"RATIO OF POPULATION >65"',
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_higher_65": 1, "cs_population": 1},
)
def ratio_of_population_65():
    return population_higher_65() / cs_population()


@component.add(
    name="SENSITIVITY FROM DENSITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_density": 2},
)
def sensitivity_from_density():
    return if_then_else(
        population_density() > 1000,
        lambda: 1,
        lambda: if_then_else(500 < population_density(), lambda: 0.25, lambda: 0.5),
    )


@component.add(
    name="MORTALITY RATE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"deaths": 1, "cs_population": 1},
)
def mortality_rate():
    return deaths() / cs_population()


@component.add(
    name="EDUCATIONAL LEVEL",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"educated_population": 1, "cs_population": 1},
)
def educational_level():
    return educated_population() / cs_population()


@component.add(
    name="HEALTH SERVICES",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"rooms": 1, "cs_population": 1},
)
def health_services():
    return rooms() / cs_population()


@component.add(
    name="DECREASE POPULATION SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_adaptation_efficiency": 1, "adaptation_for_population": 1},
)
def decrease_population_sensitivity():
    return population_adaptation_efficiency() * adaptation_for_population()


@component.add(
    name="SENSITIVITY FOR POPULATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "population_baseline_sensitivity": 3,
        "decrease_population_sensitivity": 3,
    },
)
def sensitivity_for_population():
    return if_then_else(
        population_baseline_sensitivity() - decrease_population_sensitivity() > 1,
        lambda: 1,
        lambda: if_then_else(
            population_baseline_sensitivity() - decrease_population_sensitivity() < 0,
            lambda: 0,
            lambda: population_baseline_sensitivity()
            - decrease_population_sensitivity(),
        ),
    )


@component.add(
    name="MEDIUM BIOMASS", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_biomass():
    return 0.5


@component.add(
    name="BIOMASS PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_biomass_performance"},
)
def biomass_performance():
    return _ext_constant_biomass_performance()


_ext_constant_biomass_performance = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "BIOMASS_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_biomass_performance",
)


@component.add(
    name="LIFETIME BIOMASS",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_biomass"},
)
def lifetime_biomass():
    return _ext_constant_lifetime_biomass()


_ext_constant_lifetime_biomass = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_BIOMASS*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_biomass",
)


@component.add(
    name="SWITCH BIOMASS CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_biomass_capacity"},
)
def switch_biomass_capacity():
    return _ext_constant_switch_biomass_capacity()


_ext_constant_switch_biomass_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_BIOMASS_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_biomass_capacity",
)


@component.add(
    name="VERY HIGH BIOMASS", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_biomass():
    return 1


@component.add(
    name="LOW BIOMASS", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_biomass():
    return 0.25


@component.add(
    name="MINIMUM DECOMMISSIONING BIOMASS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_biomass"},
)
def minimum_decommissioning_biomass():
    return _ext_constant_minimum_decommissioning_biomass()


_ext_constant_minimum_decommissioning_biomass = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_BIOMASS*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_biomass",
)


@component.add(
    name="DELAY BIOMASS",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_biomass": 1},
    other_deps={
        "_delayfixed_delay_biomass": {
            "initial": {"lifetime_biomass": 1},
            "step": {
                "new_biomass_capacity": 1,
                "minimum_decommissioning_for_new_biomass_capacity": 1,
            },
        }
    },
)
def delay_biomass():
    return _delayfixed_delay_biomass()


_delayfixed_delay_biomass = DelayFixed(
    lambda: new_biomass_capacity() * minimum_decommissioning_for_new_biomass_capacity(),
    lambda: lifetime_biomass(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_biomass",
)


@component.add(
    name="DELAY BIOMASS INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_biomass_initial": 1},
    other_deps={
        "_delayfixed_delay_biomass_initial": {
            "initial": {},
            "step": {"minimum_decommissioning_biomass": 1, "biomass_capacity": 1},
        }
    },
)
def delay_biomass_initial():
    return _delayfixed_delay_biomass_initial()


_delayfixed_delay_biomass_initial = DelayFixed(
    lambda: minimum_decommissioning_biomass() * biomass_capacity(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_biomass_initial",
)


@component.add(
    name="INITIAL BIOMASS CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_biomass_capacity"},
)
def initial_biomass_capacity():
    return _ext_constant_initial_biomass_capacity()


_ext_constant_initial_biomass_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_BIOMASS_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_biomass_capacity",
)


@component.add(
    name="BIOMASS ENERGY CONTENT",
    units="MWh/ton",
    comp_type="Constant",
    comp_subtype="Normal",
)
def biomass_energy_content():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'ENERGY', 'BIOMASS_ENERGY_CONTENT*')
    """
    return 5


@component.add(
    name="FINAL YEAR BIOMASS POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_biomass_policy_capacity"},
)
def final_year_biomass_policy_capacity():
    return _ext_constant_final_year_biomass_policy_capacity()


_ext_constant_final_year_biomass_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_BIOMASS_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_biomass_policy_capacity",
)


@component.add(
    name="BIOMASS CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_biomass_capacity": 1},
    other_deps={
        "_integ_biomass_capacity": {
            "initial": {"initial_biomass_capacity": 1},
            "step": {"new_biomass_capacity": 1, "decomminssioning_biomass": 1},
        }
    },
)
def biomass_capacity():
    return _integ_biomass_capacity()


_integ_biomass_capacity = Integ(
    lambda: new_biomass_capacity() - decomminssioning_biomass(),
    lambda: initial_biomass_capacity(),
    "_integ_biomass_capacity",
)


@component.add(
    name="INITIAL YEAR HYDRO POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_hydro_policy_capacity"},
)
def initial_year_hydro_policy_capacity():
    return _ext_constant_initial_year_hydro_policy_capacity()


_ext_constant_initial_year_hydro_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_HYDRO_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_hydro_policy_capacity",
)


@component.add(
    name="SWITCH HYDRO CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_hydro_capacity"},
)
def switch_hydro_capacity():
    return _ext_constant_switch_hydro_capacity()


_ext_constant_switch_hydro_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_HYDRO_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_hydro_capacity",
)


@component.add(
    name="FINAL YEAR HYDRO POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_hydro_policy_capacity"},
)
def final_year_hydro_policy_capacity():
    return _ext_constant_final_year_hydro_policy_capacity()


_ext_constant_final_year_hydro_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_HYDRO_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_hydro_policy_capacity",
)


@component.add(
    name="ENERGY SHARE OBJECTIVES",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_energy_share": 3,
        "best_progress_scenario": 1,
        "good_progress_scenario": 1,
        "gradual_progress_scenario": 2,
    },
)
def energy_share_objectives():
    return if_then_else(
        select_energy_share() == 2,
        lambda: best_progress_scenario(),
        lambda: if_then_else(
            select_energy_share() == 1,
            lambda: good_progress_scenario(),
            lambda: if_then_else(
                select_energy_share() == 0,
                lambda: gradual_progress_scenario(),
                lambda: gradual_progress_scenario(),
            ),
        ),
    )


@component.add(
    name="BIOMASS CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_biomass_incentives": 4,
        "desired_objective_for_biomass": 5,
        "low_biomass": 2,
        "medium_biomass": 1,
        "high_biomass": 1,
        "very_high_biomass": 1,
    },
)
def biomass_capacity_objective():
    return if_then_else(
        select_biomass_incentives() == 0,
        lambda: low_biomass() * desired_objective_for_biomass(),
        lambda: if_then_else(
            select_biomass_incentives() == 1,
            lambda: medium_biomass() * desired_objective_for_biomass(),
            lambda: if_then_else(
                select_biomass_incentives() == 2,
                lambda: high_biomass() * desired_objective_for_biomass(),
                lambda: if_then_else(
                    select_biomass_incentives() == 3,
                    lambda: very_high_biomass() * desired_objective_for_biomass(),
                    lambda: low_biomass() * desired_objective_for_biomass(),
                ),
            ),
        ),
    )


@component.add(
    name="SELECT BIOMASS INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_biomass_incentives"},
)
def select_biomass_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_biomass_incentives()


_ext_constant_select_biomass_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_BIOMASS_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_biomass_incentives",
)


@component.add(
    name="INITIAL YEAR BIOMASS POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_biomass_policy_capacity"},
)
def initial_year_biomass_policy_capacity():
    return _ext_constant_initial_year_biomass_policy_capacity()


_ext_constant_initial_year_biomass_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_BIOMASS_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_biomass_policy_capacity",
)


@component.add(
    name="HIGH BIOMASS", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_biomass():
    return 0.75


@component.add(
    name="DECOMMINSSIONING BIOMASS",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "lifetime_biomass": 1,
        "initial_time": 1,
        "delay_biomass_initial": 1,
        "delay_biomass": 1,
    },
)
def decomminssioning_biomass():
    return if_then_else(
        time() < initial_time() + lifetime_biomass(),
        lambda: delay_biomass_initial(),
        lambda: delay_biomass(),
    )


@component.add(
    name="DESIRED OBJECTIVE FOR BIOMASS",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_objective_for_biomass"},
)
def desired_objective_for_biomass():
    return _ext_constant_desired_objective_for_biomass()


_ext_constant_desired_objective_for_biomass = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_BIOMASS*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_biomass",
)


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW BIOMASS CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_biomass_capacity"
    },
)
def minimum_decommissioning_for_new_biomass_capacity():
    return _ext_constant_minimum_decommissioning_for_new_biomass_capacity()


_ext_constant_minimum_decommissioning_for_new_biomass_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_BIOMASS_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_biomass_capacity",
)


@component.add(
    name="ENERGY SUPPLY",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"production": 1, "losses": 1},
)
def energy_supply():
    return production() - losses()


@component.add(
    name="LIFETIME HYDRO",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_hydro"},
)
def lifetime_hydro():
    return _ext_constant_lifetime_hydro()


_ext_constant_lifetime_hydro = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_HYDRO*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_hydro",
)


@component.add(
    name="LIFETIME NON RENEWABLE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_non_renewable"},
)
def lifetime_non_renewable():
    return _ext_constant_lifetime_non_renewable()


_ext_constant_lifetime_non_renewable = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_NON_RENEWABLE*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_non_renewable",
)


@component.add(
    name="LIFETIME ROOFTOP",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_rooftop"},
)
def lifetime_rooftop():
    return _ext_constant_lifetime_rooftop()


_ext_constant_lifetime_rooftop = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_ROOFTOP*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_rooftop",
)


@component.add(
    name="LIFETIME SOLAR",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_solar"},
)
def lifetime_solar():
    return _ext_constant_lifetime_solar()


_ext_constant_lifetime_solar = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_solar",
)


@component.add(
    name="LIFETIME WIND",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_lifetime_wind"},
)
def lifetime_wind():
    return _ext_constant_lifetime_wind()


_ext_constant_lifetime_wind = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "LIFETIME_WIND*",
    {},
    _root,
    {},
    "_ext_constant_lifetime_wind",
)


@component.add(
    name="LOSSES",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"production": 1, "losses_factor": 1},
)
def losses():
    return production() * losses_factor()


@component.add(
    name="LOSSES FACTOR",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_losses_factor"},
)
def losses_factor():
    return _ext_constant_losses_factor()


_ext_constant_losses_factor = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "LOSSES_FACTOR*",
    {},
    _root,
    {},
    "_ext_constant_losses_factor",
)


@component.add(
    name="DELAY HYDRO",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_hydro": 1},
    other_deps={
        "_delayfixed_delay_hydro": {
            "initial": {"lifetime_hydro": 1},
            "step": {
                "new_hydro_capacity": 1,
                "minimum_decommissioning_for_new_hydro_capacity": 1,
            },
        }
    },
)
def delay_hydro():
    return _delayfixed_delay_hydro()


_delayfixed_delay_hydro = DelayFixed(
    lambda: new_hydro_capacity() * minimum_decommissioning_for_new_hydro_capacity(),
    lambda: lifetime_hydro(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_hydro",
)


@component.add(
    name="LOW ENERGY EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_energy_efficiency():
    return 0.25


@component.add(
    name="LOW ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_energy_emissions_decrease():
    return 0.2


@component.add(
    name="LOW HYDRO", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_hydro():
    return 0.25


@component.add(
    name="LOW ROOFTOP", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_rooftop():
    return 0.25


@component.add(
    name="BALANCE",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_consumption": 1, "energy_supply": 1},
)
def balance():
    return energy_consumption() - energy_supply()


@component.add(
    name="LOW TAX", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_tax():
    return 0.25


@component.add(
    name="LOW WIND", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_wind():
    return 0.25


@component.add(
    name="BEST PROGRESS SCENARIO",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_best_progress_scenario"},
)
def best_progress_scenario():
    return _ext_constant_best_progress_scenario()


_ext_constant_best_progress_scenario = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "BEST_PROGRESS_SCENARIO*",
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    _root,
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    "_ext_constant_best_progress_scenario",
)


@component.add(
    name="NON RENEWABLE CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_non_renewable_capacity": 1},
    other_deps={
        "_integ_non_renewable_capacity": {
            "initial": {"initial_non_renewable_capacity": 1},
            "step": {
                "new_non_renewable_capacity": 1,
                "decommissionning_non_renewable": 1,
            },
        }
    },
)
def non_renewable_capacity():
    return _integ_non_renewable_capacity()


_integ_non_renewable_capacity = Integ(
    lambda: new_non_renewable_capacity() - decommissionning_non_renewable(),
    lambda: initial_non_renewable_capacity(),
    "_integ_non_renewable_capacity",
)


@component.add(
    name="CARBON EMISSIONS",
    units="tCO2",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"energy_emission_factors": 1, "energy_consumption_share": 1},
)
def carbon_emissions():
    return energy_emission_factors() * energy_consumption_share()


@component.add(
    name="SELECT HYDRO INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_hydro_incentives"},
)
def select_hydro_incentives():
    """
    = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_hydro_incentives()


_ext_constant_select_hydro_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_HYDRO_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_hydro_incentives",
)


@component.add(
    name="MAX AREA ROOFTOP", units="km2", comp_type="Constant", comp_subtype="Normal"
)
def max_area_rooftop():
    return 10


@component.add(
    name="HIGH HYDRO", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_hydro():
    return 0.75


@component.add(
    name="CS POPULATION",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_cs_population": 1},
    other_deps={
        "_smooth_cs_population": {
            "initial": {"growth_rate": 1},
            "step": {"growth_rate": 1},
        },
        "_integ_cs_population": {
            "initial": {"initial_population": 1},
            "step": {"_smooth_cs_population": 1},
        },
    },
)
def cs_population():
    return _integ_cs_population()


_smooth_cs_population = Smooth(
    lambda: growth_rate(),
    lambda: 2,
    lambda: growth_rate(),
    lambda: 1,
    "_smooth_cs_population",
)

_integ_cs_population = Integ(
    lambda: _smooth_cs_population(),
    lambda: initial_population(),
    "_integ_cs_population",
)


@component.add(
    name="MAX POTENTIAL HYDRO",
    units="MWh",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_max_potential_hydro"},
)
def max_potential_hydro():
    return _ext_constant_max_potential_hydro()


_ext_constant_max_potential_hydro = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MAX_POTENTIAL_HYDRO*",
    {},
    _root,
    {},
    "_ext_constant_max_potential_hydro",
)


@component.add(
    name="MAX POTENTIAL SOLAR",
    units="MW/km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_max_potential_solar"},
)
def max_potential_solar():
    return _ext_constant_max_potential_solar()


_ext_constant_max_potential_solar = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MAX_POTENTIAL_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_max_potential_solar",
)


@component.add(
    name="MAX POTENTIAL WIND",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_max_potential_wind"},
)
def max_potential_wind():
    return _ext_constant_max_potential_wind()


_ext_constant_max_potential_wind = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MAX_POTENTIAL_WIND*",
    {},
    _root,
    {},
    "_ext_constant_max_potential_wind",
)


@component.add(
    name="DECOMMISSIONING HYDRO",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "lifetime_hydro": 1,
        "initial_time": 1,
        "delay_hydro_initial": 1,
        "delay_hydro": 1,
    },
)
def decommissioning_hydro():
    return if_then_else(
        time() < initial_time() + lifetime_hydro(),
        lambda: delay_hydro_initial(),
        lambda: delay_hydro(),
    )


@component.add(
    name="DECOMMISSIONING ROOFTOP",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "lifetime_rooftop": 1,
        "initial_time": 1,
        "delay_rooftop_initial": 1,
        "delay_rooftop": 1,
    },
)
def decommissioning_rooftop():
    return if_then_else(
        time() < initial_time() + lifetime_rooftop(),
        lambda: delay_rooftop_initial(),
        lambda: delay_rooftop(),
    )


@component.add(
    name="DECOMMISSIONING WIND",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "lifetime_wind": 1,
        "initial_time": 1,
        "delay_wind_initial": 1,
        "delay_wind": 1,
    },
)
def decommissioning_wind():
    return if_then_else(
        time() < initial_time() + lifetime_wind(),
        lambda: delay_wind_initial(),
        lambda: delay_wind(),
    )


@component.add(
    name="DECOMMISSIONNING NON RENEWABLE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "non_renewable_capacity": 2,
        "delay_non_renewable_initial": 4,
        "time": 5,
        "non_renewable_capacity_objective": 4,
        "initial_year_decrease_non_renewable_policy_capacity": 5,
        "lifetime_non_renewable": 3,
        "delay_non_renewable": 4,
        "switch_decrease_non_renewable_capacity": 2,
        "final_year_decrease_non_renewable_policy_capacity": 5,
        "initial_time": 3,
    },
)
def decommissionning_non_renewable():
    return if_then_else(
        non_renewable_capacity() - non_renewable_capacity() * 0.2 <= 0,
        lambda: 0,
        lambda: if_then_else(
            switch_decrease_non_renewable_capacity() == 0,
            lambda: if_then_else(
                time() < initial_time() + lifetime_non_renewable(),
                lambda: delay_non_renewable_initial(),
                lambda: delay_non_renewable(),
            ),
            lambda: if_then_else(
                np.logical_and(
                    switch_decrease_non_renewable_capacity() == 1,
                    np.logical_or(
                        time() < initial_year_decrease_non_renewable_policy_capacity(),
                        time() > final_year_decrease_non_renewable_policy_capacity(),
                    ),
                ),
                lambda: if_then_else(
                    time() < initial_time() + lifetime_non_renewable(),
                    lambda: delay_non_renewable_initial(),
                    lambda: delay_non_renewable(),
                ),
                lambda: if_then_else(
                    time() < initial_time() + lifetime_non_renewable(),
                    lambda: if_then_else(
                        delay_non_renewable_initial()
                        > non_renewable_capacity_objective()
                        / (
                            final_year_decrease_non_renewable_policy_capacity()
                            - initial_year_decrease_non_renewable_policy_capacity()
                        ),
                        lambda: delay_non_renewable_initial(),
                        lambda: non_renewable_capacity_objective()
                        / (
                            final_year_decrease_non_renewable_policy_capacity()
                            - initial_year_decrease_non_renewable_policy_capacity()
                        ),
                    ),
                    lambda: if_then_else(
                        delay_non_renewable()
                        > non_renewable_capacity_objective()
                        / (
                            final_year_decrease_non_renewable_policy_capacity()
                            - initial_year_decrease_non_renewable_policy_capacity()
                        ),
                        lambda: delay_non_renewable(),
                        lambda: non_renewable_capacity_objective()
                        / (
                            final_year_decrease_non_renewable_policy_capacity()
                            - initial_year_decrease_non_renewable_policy_capacity()
                        ),
                    ),
                ),
            ),
        ),
    )


@component.add(
    name="DELAY WIND",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_wind": 1},
    other_deps={
        "_delayfixed_delay_wind": {
            "initial": {"lifetime_wind": 1},
            "step": {
                "new_wind_capacity": 1,
                "minimum_decommissioning_for_new_wind_capacity": 1,
            },
        }
    },
)
def delay_wind():
    return _delayfixed_delay_wind()


_delayfixed_delay_wind = DelayFixed(
    lambda: new_wind_capacity() * minimum_decommissioning_for_new_wind_capacity(),
    lambda: lifetime_wind(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_wind",
)


@component.add(
    name="MEDIUM ENERGY EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_energy_efficiency():
    return 0.5


@component.add(
    name="MEDIUM ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_energy_emissions_decrease():
    return 0.4


@component.add(
    name="MEDIUM HYDRO", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_hydro():
    return 0.5


@component.add(
    name="MEDIUM ROOFTOP", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_rooftop():
    return 0.5


@component.add(
    name="MEDIUM SOLAR", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_solar():
    return 0.5


@component.add(
    name="MEDIUM TAX", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_tax():
    return 0.5


@component.add(
    name="DEFICIT TO IMPORT",
    units="Million Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"balance": 2},
)
def deficit_to_import():
    return if_then_else(balance() > 0, lambda: balance(), lambda: 0)


@component.add(
    name="VERY HIGH ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_energy_emissions_decrease():
    return 0.8


@component.add(
    name="EXPORT",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"exceedance_to_export": 1},
)
def export():
    return exceedance_to_export()


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW HYDRO CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_hydro_capacity"
    },
)
def minimum_decommissioning_for_new_hydro_capacity():
    return _ext_constant_minimum_decommissioning_for_new_hydro_capacity()


_ext_constant_minimum_decommissioning_for_new_hydro_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_HYDRO_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_hydro_capacity",
)


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW NON RENEWABLE CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_non_renewable_capacity"
    },
)
def minimum_decommissioning_for_new_non_renewable_capacity():
    return _ext_constant_minimum_decommissioning_for_new_non_renewable_capacity()


_ext_constant_minimum_decommissioning_for_new_non_renewable_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_NON_RENEWABLE_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_non_renewable_capacity",
)


@component.add(
    name="SWITCH ROOFTOP CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_rooftop_capacity"},
)
def switch_rooftop_capacity():
    return _ext_constant_switch_rooftop_capacity()


_ext_constant_switch_rooftop_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_ROOFTOP_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_rooftop_capacity",
)


@component.add(
    name="DESIRED ENERGY EFFICIENCY OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_energy_efficiency_objective"},
)
def desired_energy_efficiency_objective():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'INITIAL_EFFICIENCY_OBJECTIVE*')
    """
    return _ext_constant_desired_energy_efficiency_objective()


_ext_constant_desired_energy_efficiency_objective = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_EFFICIENCY_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_desired_energy_efficiency_objective",
)


@component.add(
    name="DELAY HYDRO INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_hydro_initial": 1},
    other_deps={
        "_delayfixed_delay_hydro_initial": {
            "initial": {},
            "step": {"hydropower_capacity": 1, "minimum_decommissioning_hydro": 1},
        }
    },
)
def delay_hydro_initial():
    return _delayfixed_delay_hydro_initial()


_delayfixed_delay_hydro_initial = DelayFixed(
    lambda: hydropower_capacity() * minimum_decommissioning_hydro(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_hydro_initial",
)


@component.add(
    name="DELAY NON RENEWABLE",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_non_renewable": 1},
    other_deps={
        "_delayfixed_delay_non_renewable": {
            "initial": {"lifetime_non_renewable": 1},
            "step": {
                "new_non_renewable_capacity": 1,
                "minimum_decommissioning_for_new_non_renewable_capacity": 1,
            },
        }
    },
)
def delay_non_renewable():
    return _delayfixed_delay_non_renewable()


_delayfixed_delay_non_renewable = DelayFixed(
    lambda: new_non_renewable_capacity()
    * minimum_decommissioning_for_new_non_renewable_capacity(),
    lambda: lifetime_non_renewable(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_non_renewable",
)


@component.add(
    name="DELAY NON RENEWABLE INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_non_renewable_initial": 1},
    other_deps={
        "_delayfixed_delay_non_renewable_initial": {
            "initial": {},
            "step": {
                "non_renewable_capacity": 1,
                "minimum_decommissioning_non_renewable": 1,
            },
        }
    },
)
def delay_non_renewable_initial():
    return _delayfixed_delay_non_renewable_initial()


_delayfixed_delay_non_renewable_initial = DelayFixed(
    lambda: non_renewable_capacity() * minimum_decommissioning_non_renewable(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_non_renewable_initial",
)


@component.add(
    name="DELAY ROOFTOP",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_rooftop": 1},
    other_deps={
        "_delayfixed_delay_rooftop": {
            "initial": {"lifetime_rooftop": 1},
            "step": {
                "new_rooftop_capacity": 1,
                "minimum_decommissioning_for_new_rooftop_capacity": 1,
            },
        }
    },
)
def delay_rooftop():
    return _delayfixed_delay_rooftop()


_delayfixed_delay_rooftop = DelayFixed(
    lambda: new_rooftop_capacity() * minimum_decommissioning_for_new_rooftop_capacity(),
    lambda: lifetime_rooftop(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_rooftop",
)


@component.add(
    name="DELAY ROOFTOP INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_rooftop_initial": 1},
    other_deps={
        "_delayfixed_delay_rooftop_initial": {
            "initial": {},
            "step": {"minimum_decommissioning_rooftop": 1, "rooftop_capacity": 1},
        }
    },
)
def delay_rooftop_initial():
    return _delayfixed_delay_rooftop_initial()


_delayfixed_delay_rooftop_initial = DelayFixed(
    lambda: minimum_decommissioning_rooftop() * rooftop_capacity(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_rooftop_initial",
)


@component.add(
    name="DELAY SOLAR",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_solar": 1},
    other_deps={
        "_delayfixed_delay_solar": {
            "initial": {"lifetime_solar": 1},
            "step": {
                "new_solar_capacity": 1,
                "minimum_decommissioning_for_new_solar_capacity": 1,
            },
        }
    },
)
def delay_solar():
    return _delayfixed_delay_solar()


_delayfixed_delay_solar = DelayFixed(
    lambda: new_solar_capacity() * minimum_decommissioning_for_new_solar_capacity(),
    lambda: lifetime_solar(),
    lambda: 0,
    time_step,
    "_delayfixed_delay_solar",
)


@component.add(
    name="DELAY SOLAR INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_solar_initial": 1},
    other_deps={
        "_delayfixed_delay_solar_initial": {
            "initial": {},
            "step": {"solar_capacity": 1, "minimum_decommissioning_solar": 1},
        }
    },
)
def delay_solar_initial():
    return _delayfixed_delay_solar_initial()


_delayfixed_delay_solar_initial = DelayFixed(
    lambda: solar_capacity() * minimum_decommissioning_solar(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_solar_initial",
)


@component.add(
    name="DELAY WIND INITIAL",
    units="MW",
    comp_type="Stateful",
    comp_subtype="DelayFixed",
    depends_on={"_delayfixed_delay_wind_initial": 1},
    other_deps={
        "_delayfixed_delay_wind_initial": {
            "initial": {},
            "step": {"wind_capacity": 1, "minimum_decommissioning_wind": 1},
        }
    },
)
def delay_wind_initial():
    return _delayfixed_delay_wind_initial()


_delayfixed_delay_wind_initial = DelayFixed(
    lambda: wind_capacity() * minimum_decommissioning_wind(),
    lambda: 5,
    lambda: 0,
    time_step,
    "_delayfixed_delay_wind_initial",
)


@component.add(
    name="SELECT CARBON TAX",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_carbon_tax"},
)
def select_carbon_tax():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_carbon_tax()


_ext_constant_select_carbon_tax = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_CARBON_TAX*",
    {},
    _root,
    {},
    "_ext_constant_select_carbon_tax",
)


@component.add(
    name="NON RENEWABLE CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_carbon_tax": 4,
        "low_tax": 2,
        "desired_objective_for_decrease_non_renewable_capacity": 5,
        "high_tax": 1,
        "very_high_tax": 1,
        "medium_tax": 1,
    },
)
def non_renewable_capacity_objective():
    return if_then_else(
        select_carbon_tax() == 0,
        lambda: low_tax() * desired_objective_for_decrease_non_renewable_capacity(),
        lambda: if_then_else(
            select_carbon_tax() == 1,
            lambda: medium_tax()
            * desired_objective_for_decrease_non_renewable_capacity(),
            lambda: if_then_else(
                select_carbon_tax() == 2,
                lambda: high_tax()
                * desired_objective_for_decrease_non_renewable_capacity(),
                lambda: if_then_else(
                    select_carbon_tax() == 3,
                    lambda: very_high_tax()
                    * desired_objective_for_decrease_non_renewable_capacity(),
                    lambda: low_tax()
                    * desired_objective_for_decrease_non_renewable_capacity(),
                ),
            ),
        ),
    )


@component.add(
    name="NON RENEWABLE PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_non_renewable_performance"},
)
def non_renewable_performance():
    return _ext_constant_non_renewable_performance()


_ext_constant_non_renewable_performance = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "NON_RENEWABLE_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_non_renewable_performance",
)


@component.add(
    name="INCREASE NON RENEWABLE CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_increase_non_renewable_capacity_objective"
    },
)
def increase_non_renewable_capacity_objective():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'ENERGY', 'INCREASE_NON_RENEWABLE_CAPACITY*')
    """
    return _ext_constant_increase_non_renewable_capacity_objective()


_ext_constant_increase_non_renewable_capacity_objective = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INCREASE_NON_RENEWABLE_CAPACITY_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_increase_non_renewable_capacity_objective",
)


@component.add(
    name="VERY HIGH HYDRO", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_hydro():
    return 1


@component.add(
    name="INITIAL WIND CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_wind_capacity"},
)
def initial_wind_capacity():
    return _ext_constant_initial_wind_capacity()


_ext_constant_initial_wind_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_WIND_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_wind_capacity",
)


@component.add(
    name="ENERGY EFFICIENCY OBJECTIVE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_energy_efficiency_incentives": 4,
        "desired_energy_efficiency_objective": 5,
        "low_energy_efficiency": 2,
        "very_high_energy_efficiency": 1,
        "high_energy_efficiency": 1,
        "medium_energy_efficiency": 1,
    },
)
def energy_efficiency_objective():
    return if_then_else(
        select_energy_efficiency_incentives() == 0,
        lambda: low_energy_efficiency() * desired_energy_efficiency_objective(),
        lambda: if_then_else(
            select_energy_efficiency_incentives() == 1,
            lambda: medium_energy_efficiency() * desired_energy_efficiency_objective(),
            lambda: if_then_else(
                select_energy_efficiency_incentives() == 2,
                lambda: high_energy_efficiency()
                * desired_energy_efficiency_objective(),
                lambda: if_then_else(
                    select_energy_efficiency_incentives() == 3,
                    lambda: very_high_energy_efficiency()
                    * desired_energy_efficiency_objective(),
                    lambda: low_energy_efficiency()
                    * desired_energy_efficiency_objective(),
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY EMISSION FACTORS",
    units="tCO2/MWh",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_incentive_for_energy_emissions_decrease": 3,
        "initial_energy_emission_factors": 4,
        "initial_year_incentive_for_energy_emissions_decrease": 3,
        "time": 3,
        "final_year_incentive_for_energy_emissions_decrease": 2,
        "energy_emissions_decrease_incentives": 2,
    },
)
def energy_emission_factors():
    return if_then_else(
        switch_incentive_for_energy_emissions_decrease() == 0,
        lambda: initial_energy_emission_factors(),
        lambda: if_then_else(
            np.logical_and(
                switch_incentive_for_energy_emissions_decrease() == 1,
                time() < initial_year_incentive_for_energy_emissions_decrease(),
            ),
            lambda: initial_energy_emission_factors(),
            lambda: if_then_else(
                np.logical_and(
                    switch_incentive_for_energy_emissions_decrease() == 1,
                    time() > final_year_incentive_for_energy_emissions_decrease(),
                ),
                lambda: initial_energy_emission_factors()
                - energy_emissions_decrease_incentives(),
                lambda: initial_energy_emission_factors()
                - (
                    energy_emissions_decrease_incentives()
                    * (time() - initial_year_incentive_for_energy_emissions_decrease())
                )
                / (
                    final_year_incentive_for_energy_emissions_decrease()
                    - initial_year_incentive_for_energy_emissions_decrease()
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY EMISSIONS DECREASE INCENTIVES",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_incentive_for_energy_emissions_decrease": 4,
        "very_high_energy_emissions_decrease": 1,
        "initial_energy_emission_factors": 5,
        "medium_energy_emissions_decrease": 1,
        "low_energy_emissions_decrease": 2,
        "high_energy_emissions_decrease": 1,
    },
)
def energy_emissions_decrease_incentives():
    return if_then_else(
        select_incentive_for_energy_emissions_decrease() == 3,
        lambda: very_high_energy_emissions_decrease()
        * initial_energy_emission_factors(),
        lambda: if_then_else(
            select_incentive_for_energy_emissions_decrease() == 2,
            lambda: high_energy_emissions_decrease()
            * initial_energy_emission_factors(),
            lambda: if_then_else(
                select_incentive_for_energy_emissions_decrease() == 1,
                lambda: medium_energy_emissions_decrease()
                * initial_energy_emission_factors(),
                lambda: if_then_else(
                    select_incentive_for_energy_emissions_decrease() == 0,
                    lambda: low_energy_emissions_decrease()
                    * initial_energy_emission_factors(),
                    lambda: low_energy_emissions_decrease()
                    * initial_energy_emission_factors(),
                ),
            ),
        ),
    )


@component.add(
    name="ENERGY CONSUMPTION SHARE",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_energy_share_policy": 2,
        "initial_energy_share": 2,
        "energy_consumption": 3,
        "initial_year_energy_share_policy": 1,
        "time": 1,
        "energy_share_objectives": 1,
    },
)
def energy_consumption_share():
    return if_then_else(
        switch_energy_share_policy() == 0,
        lambda: initial_energy_share() * energy_consumption(),
        lambda: if_then_else(
            np.logical_and(
                switch_energy_share_policy() == 1,
                time() < initial_year_energy_share_policy(),
            ),
            lambda: initial_energy_share() * energy_consumption(),
            lambda: energy_share_objectives() * energy_consumption(),
        ),
    )


@component.add(
    name="HIGH TAX", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_tax():
    return 0.75


@component.add(
    name="FINAL YEAR ROOFTOP POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_rooftop_policy_capacity"},
)
def final_year_rooftop_policy_capacity():
    return _ext_constant_final_year_rooftop_policy_capacity()


_ext_constant_final_year_rooftop_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_ROOFTOP_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_rooftop_policy_capacity",
)


@component.add(
    name="FINAL YEAR SOLAR POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_solar_policy_capacity"},
)
def final_year_solar_policy_capacity():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'FINAL_YEAR_SOLAR_POLICY_CAPACITY*')
    """
    return _ext_constant_final_year_solar_policy_capacity()


_ext_constant_final_year_solar_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_SOLAR_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_solar_policy_capacity",
)


@component.add(
    name="IMPORT",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"deficit_to_import": 1},
)
def import_1():
    return deficit_to_import()


@component.add(
    name="HOURS IN A YEAR",
    units="Hours",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_hours_in_a_year"},
)
def hours_in_a_year():
    return _ext_constant_hours_in_a_year()


_ext_constant_hours_in_a_year = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "HOURS_IN_A_YEAR*",
    {},
    _root,
    {},
    "_ext_constant_hours_in_a_year",
)


@component.add(
    name="DESIRED OBJECTIVE FOR WIND",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_objective_for_wind"},
)
def desired_objective_for_wind():
    return _ext_constant_desired_objective_for_wind()


_ext_constant_desired_objective_for_wind = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_WIND*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_wind",
)


@component.add(
    name="SELECT ENERGY EFFICIENCY INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_energy_efficiency_incentives"},
)
def select_energy_efficiency_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SELECT_EFFICIENCY_INCENTIVES*')
    """
    return _ext_constant_select_energy_efficiency_incentives()


_ext_constant_select_energy_efficiency_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_EFFICIENCY_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_energy_efficiency_incentives",
)


@component.add(
    name="GOOD PROGRESS SCENARIO",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_good_progress_scenario"},
)
def good_progress_scenario():
    return _ext_constant_good_progress_scenario()


_ext_constant_good_progress_scenario = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "GOOD_PROGRESS_SCENARIO*",
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    _root,
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    "_ext_constant_good_progress_scenario",
)


@component.add(
    name="EXCEEDANCE TO EXPORT",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"balance": 2},
)
def exceedance_to_export():
    return if_then_else(balance() < 0, lambda: -1 * balance(), lambda: 0)


@component.add(
    name="MEDIUM WIND", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def medium_wind():
    return 0.5


@component.add(
    name="SWITCH ENERGY SHARE POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_energy_share_policy"},
)
def switch_energy_share_policy():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_ENERGY_SHARE_POLICY*')
    """
    return _ext_constant_switch_energy_share_policy()


_ext_constant_switch_energy_share_policy = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_ENERGY_SHARE_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_switch_energy_share_policy",
)


@component.add(
    name="SWITCH INCENTIVE FOR ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_switch_incentive_for_energy_emissions_decrease"
    },
)
def switch_incentive_for_energy_emissions_decrease():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_INCENTIVE_EMISSIONS*')
    """
    return _ext_constant_switch_incentive_for_energy_emissions_decrease()


_ext_constant_switch_incentive_for_energy_emissions_decrease = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_INCENTIVE_EMISSIONS*",
    {},
    _root,
    {},
    "_ext_constant_switch_incentive_for_energy_emissions_decrease",
)


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW ROOFTOP CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_rooftop_capacity"
    },
)
def minimum_decommissioning_for_new_rooftop_capacity():
    return _ext_constant_minimum_decommissioning_for_new_rooftop_capacity()


_ext_constant_minimum_decommissioning_for_new_rooftop_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_ROOFTOP_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_rooftop_capacity",
)


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW SOLAR CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_solar_capacity"
    },
)
def minimum_decommissioning_for_new_solar_capacity():
    return _ext_constant_minimum_decommissioning_for_new_solar_capacity()


_ext_constant_minimum_decommissioning_for_new_solar_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_SOLAR_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_solar_capacity",
)


@component.add(
    name="INITIAL ENERGY EMISSION FACTORS",
    units="tCO2/MWh",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_energy_emission_factors"},
)
def initial_energy_emission_factors():
    return _ext_constant_initial_energy_emission_factors()


_ext_constant_initial_energy_emission_factors = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_EMISSION_FACTORS*",
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    _root,
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    "_ext_constant_initial_energy_emission_factors",
)


@component.add(
    name="FINAL YEAR ENERGY EFFICIENCY INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_energy_efficiency_increase"},
)
def final_year_energy_efficiency_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'FINAL_YEAR_EFFICIENCY_INCREASE*')
    """
    return _ext_constant_final_year_energy_efficiency_increase()


_ext_constant_final_year_energy_efficiency_increase = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_EFFICIENCY_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_final_year_energy_efficiency_increase",
)


@component.add(
    name="MINIMUM DECOMMISSIONING ROOFTOP",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_rooftop"},
)
def minimum_decommissioning_rooftop():
    return _ext_constant_minimum_decommissioning_rooftop()


_ext_constant_minimum_decommissioning_rooftop = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_ROOFTOP*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_rooftop",
)


@component.add(
    name="LOW SOLAR", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def low_solar():
    return 0.25


@component.add(
    name="ROOFTOP CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_rooftop_capacity": 1},
    other_deps={
        "_integ_rooftop_capacity": {
            "initial": {"initial_rooftop_capacity": 1},
            "step": {"new_rooftop_capacity": 1, "decommissioning_rooftop": 1},
        }
    },
)
def rooftop_capacity():
    return _integ_rooftop_capacity()


_integ_rooftop_capacity = Integ(
    lambda: new_rooftop_capacity() - decommissioning_rooftop(),
    lambda: initial_rooftop_capacity(),
    "_integ_rooftop_capacity",
)


@component.add(
    name="ROOFTOP CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_rooftop_incentives": 4,
        "desired_objective_for_rooftop": 5,
        "low_rooftop": 2,
        "high_rooftop": 1,
        "very_high_rooftop": 1,
        "medium_rooftop": 1,
    },
)
def rooftop_capacity_objective():
    return if_then_else(
        select_rooftop_incentives() == 0,
        lambda: low_rooftop() * desired_objective_for_rooftop(),
        lambda: if_then_else(
            select_rooftop_incentives() == 1,
            lambda: medium_rooftop() * desired_objective_for_rooftop(),
            lambda: if_then_else(
                select_rooftop_incentives() == 2,
                lambda: high_rooftop() * desired_objective_for_rooftop(),
                lambda: if_then_else(
                    select_rooftop_incentives() == 3,
                    lambda: very_high_rooftop() * desired_objective_for_rooftop(),
                    lambda: low_rooftop() * desired_objective_for_rooftop(),
                ),
            ),
        ),
    )


@component.add(
    name="FINAL YEAR INCENTIVE FOR ENERGY EMISSIONS DECREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_incentive_for_energy_emissions_decrease"
    },
)
def final_year_incentive_for_energy_emissions_decrease():
    return _ext_constant_final_year_incentive_for_energy_emissions_decrease()


_ext_constant_final_year_incentive_for_energy_emissions_decrease = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_INCENTIVE_EMISSIONS*",
    {},
    _root,
    {},
    "_ext_constant_final_year_incentive_for_energy_emissions_decrease",
)


@component.add(
    name="FINAL YEAR DECREASE NON RENEWABLE POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_final_year_decrease_non_renewable_policy_capacity"
    },
)
def final_year_decrease_non_renewable_policy_capacity():
    return _ext_constant_final_year_decrease_non_renewable_policy_capacity()


_ext_constant_final_year_decrease_non_renewable_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_NON_RENEWABLE_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_decrease_non_renewable_policy_capacity",
)


@component.add(
    name="HYDROPOWER CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_hydropower_capacity": 1},
    other_deps={
        "_integ_hydropower_capacity": {
            "initial": {"initial_hydro_capacity": 1},
            "step": {"new_hydro_capacity": 1, "decommissioning_hydro": 1},
        }
    },
)
def hydropower_capacity():
    return _integ_hydropower_capacity()


_integ_hydropower_capacity = Integ(
    lambda: new_hydro_capacity() - decommissioning_hydro(),
    lambda: initial_hydro_capacity(),
    "_integ_hydropower_capacity",
)


@component.add(
    name="FINAL YEAR WIND POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_wind_policy_capacity"},
)
def final_year_wind_policy_capacity():
    return _ext_constant_final_year_wind_policy_capacity()


_ext_constant_final_year_wind_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "FINAL_YEAR_WIND_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_wind_policy_capacity",
)


@component.add(
    name="DESIRED OBJECTIVE FOR SOLAR",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_objective_for_solar"},
)
def desired_objective_for_solar():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'INITIAL_OBJECTIVE_FOR_SOLAR*')
    """
    return _ext_constant_desired_objective_for_solar()


_ext_constant_desired_objective_for_solar = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_solar",
)


@component.add(
    name="NEW SOLAR CAPACITY",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_solar_capacity": 2,
        "time": 2,
        "solar_capacity": 1,
        "final_year_solar_policy_capacity": 2,
        "max_solar_capacity": 1,
        "initial_year_solar_policy_capacity": 2,
        "solar_capacity_objective": 1,
    },
)
def new_solar_capacity():
    return if_then_else(
        switch_solar_capacity() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_solar_capacity() == 1,
                np.logical_or(
                    time() < initial_year_solar_policy_capacity(),
                    time() > final_year_solar_policy_capacity(),
                ),
            ),
            lambda: 0,
            lambda: float(
                np.minimum(
                    solar_capacity_objective()
                    / (
                        final_year_solar_policy_capacity()
                        - initial_year_solar_policy_capacity()
                    ),
                    max_solar_capacity() - solar_capacity(),
                )
            ),
        ),
    )


@component.add(
    name="INITIAL POPULATION",
    units="Inhabitants",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_population"},
)
def initial_population():
    return _ext_constant_initial_population()


_ext_constant_initial_population = ExtConstant(
    r"Historical.xlsx",
    "Energy",
    "INITIAL_POPULATION*",
    {},
    _root,
    {},
    "_ext_constant_initial_population",
)


@component.add(
    name="GRADUAL PROGRESS SCENARIO",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_gradual_progress_scenario"},
)
def gradual_progress_scenario():
    return _ext_constant_gradual_progress_scenario()


_ext_constant_gradual_progress_scenario = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "GRADUAL_PROGRESS_SCENARIO*",
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    _root,
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    "_ext_constant_gradual_progress_scenario",
)


@component.add(
    name="SWITCH ENERGY EFFICIENCY INCREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_energy_efficiency_increase"},
)
def switch_energy_efficiency_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_EFFICIENCY_INCREASE*')
    """
    return _ext_constant_switch_energy_efficiency_increase()


_ext_constant_switch_energy_efficiency_increase = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_EFFICIENCY_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_switch_energy_efficiency_increase",
)


@component.add(
    name="VERY HIGH ENERGY EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_energy_efficiency():
    return 1


@component.add(
    name="SELECT ENERGY SHARE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_energy_share"},
)
def select_energy_share():
    """
    0 = Gradual progress scenario 1 = Good progress scenario 2 = Best progress scenario GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SELECT_ENERGY_SHARE*')
    """
    return _ext_constant_select_energy_share()


_ext_constant_select_energy_share = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_ENERGY_SHARE*",
    {},
    _root,
    {},
    "_ext_constant_select_energy_share",
)


@component.add(
    name="SELECT ROOFTOP INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_rooftop_incentives"},
)
def select_rooftop_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_rooftop_incentives()


_ext_constant_select_rooftop_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_ROOFTOP_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_rooftop_incentives",
)


@component.add(
    name="SELECT SOLAR INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_solar_incentives"},
)
def select_solar_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SELECT_SOLAR_INCENTIVES*')
    """
    return _ext_constant_select_solar_incentives()


_ext_constant_select_solar_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_SOLAR_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_solar_incentives",
)


@component.add(
    name="HIGH ENERGY EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_energy_efficiency():
    return 0.75


@component.add(
    name="HIGH ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_energy_emissions_decrease():
    return 0.6


@component.add(
    name="SWITCH SOLAR CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_solar_capacity"},
)
def switch_solar_capacity():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_SOLAR_CAPACITY*')
    """
    return _ext_constant_switch_solar_capacity()


_ext_constant_switch_solar_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_SOLAR_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_solar_capacity",
)


@component.add(
    name="HIGH ROOFTOP", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_rooftop():
    return 0.75


@component.add(
    name="HIGH SOLAR", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_solar():
    return 0.75


@component.add(
    name="SOLAR CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_solar_incentives": 4,
        "low_solar": 2,
        "desired_objective_for_solar": 5,
        "high_solar": 1,
        "very_high_solar": 1,
        "medium_solar": 1,
    },
)
def solar_capacity_objective():
    return if_then_else(
        select_solar_incentives() == 0,
        lambda: low_solar() * desired_objective_for_solar(),
        lambda: if_then_else(
            select_solar_incentives() == 1,
            lambda: medium_solar() * desired_objective_for_solar(),
            lambda: if_then_else(
                select_solar_incentives() == 2,
                lambda: high_solar() * desired_objective_for_solar(),
                lambda: if_then_else(
                    select_solar_incentives() == 3,
                    lambda: very_high_solar() * desired_objective_for_solar(),
                    lambda: low_solar() * desired_objective_for_solar(),
                ),
            ),
        ),
    )


@component.add(
    name="HIGH WIND", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def high_wind():
    return 0.75


@component.add(
    name="HYDRO CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_hydro_incentives": 4,
        "desired_objective_for_hydro": 5,
        "low_hydro": 2,
        "high_hydro": 1,
        "medium_hydro": 1,
        "very_high_hydro": 1,
    },
)
def hydro_capacity_objective():
    return if_then_else(
        select_hydro_incentives() == 0,
        lambda: low_hydro() * desired_objective_for_hydro(),
        lambda: if_then_else(
            select_hydro_incentives() == 1,
            lambda: medium_hydro() * desired_objective_for_hydro(),
            lambda: if_then_else(
                select_hydro_incentives() == 2,
                lambda: high_hydro() * desired_objective_for_hydro(),
                lambda: if_then_else(
                    select_hydro_incentives() == 3,
                    lambda: very_high_hydro() * desired_objective_for_hydro(),
                    lambda: low_hydro() * desired_objective_for_hydro(),
                ),
            ),
        ),
    )


@component.add(
    name="HYDRO PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_hydro_performance"},
)
def hydro_performance():
    """
    GET DIRECT CONSTANTS ('Historical.xlsx', 'ENERGY', 'HYDRO_PERFORMANCE*')
    """
    return _ext_constant_hydro_performance()


_ext_constant_hydro_performance = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "HYDRO_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_hydro_performance",
)


@component.add(
    name="DESIRED OBJECTIVE FOR HYDRO",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_objective_for_hydro"},
)
def desired_objective_for_hydro():
    return _ext_constant_desired_objective_for_hydro()


_ext_constant_desired_objective_for_hydro = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_HYDRO*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_hydro",
)


@component.add(
    name="DESIRED OBJECTIVE FOR DECREASE NON RENEWABLE CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_desired_objective_for_decrease_non_renewable_capacity"
    },
)
def desired_objective_for_decrease_non_renewable_capacity():
    return _ext_constant_desired_objective_for_decrease_non_renewable_capacity()


_ext_constant_desired_objective_for_decrease_non_renewable_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_NON_RENEWABLE*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_decrease_non_renewable_capacity",
)


@component.add(
    name="DESIRED OBJECTIVE FOR ROOFTOP",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_desired_objective_for_rooftop"},
)
def desired_objective_for_rooftop():
    return _ext_constant_desired_objective_for_rooftop()


_ext_constant_desired_objective_for_rooftop = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_OBJECTIVE_FOR_ROOFTOP*",
    {},
    _root,
    {},
    "_ext_constant_desired_objective_for_rooftop",
)


@component.add(
    name="MINIMUM DECOMMISSIONING FOR NEW WIND CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_minimum_decommissioning_for_new_wind_capacity"
    },
)
def minimum_decommissioning_for_new_wind_capacity():
    return _ext_constant_minimum_decommissioning_for_new_wind_capacity()


_ext_constant_minimum_decommissioning_for_new_wind_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_FOR_NEW_WIND_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_for_new_wind_capacity",
)


@component.add(
    name="INITIAL ENERGY SHARE",
    units="Dmnl",
    subscripts=["TYPE OF ENERGY"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_energy_share"},
)
def initial_energy_share():
    return _ext_constant_initial_energy_share()


_ext_constant_initial_energy_share = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_ENERGY_SHARE*",
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    _root,
    {"TYPE OF ENERGY": _subscript_dict["TYPE OF ENERGY"]},
    "_ext_constant_initial_energy_share",
)


@component.add(
    name="INITIAL ROOFTOP CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_rooftop_capacity"},
)
def initial_rooftop_capacity():
    return _ext_constant_initial_rooftop_capacity()


_ext_constant_initial_rooftop_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_ROOFTOP_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_rooftop_capacity",
)


@component.add(
    name="INITIAL SOLAR CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_solar_capacity"},
)
def initial_solar_capacity():
    return _ext_constant_initial_solar_capacity()


_ext_constant_initial_solar_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_SOLAR_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_solar_capacity",
)


@component.add(
    name="PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_production": 1},
)
def production():
    return total_production()


@component.add(
    name="INITIAL NON RENEWABLE CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_non_renewable_capacity"},
)
def initial_non_renewable_capacity():
    return _ext_constant_initial_non_renewable_capacity()


_ext_constant_initial_non_renewable_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_NON_RENEWABLE_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_non_renewable_capacity",
)


@component.add(
    name="ROOFTOP PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_rooftop_performance"},
)
def rooftop_performance():
    return _ext_constant_rooftop_performance()


_ext_constant_rooftop_performance = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "ROOFTOP_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_rooftop_performance",
)


@component.add(
    name="VERY HIGH ROOFTOP", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_rooftop():
    return 1


@component.add(
    name="SWITCH DECREASE NON RENEWABLE CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_decrease_non_renewable_capacity"},
)
def switch_decrease_non_renewable_capacity():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SWITCH_NON_RENEWABLE_CAPACITY*')
    """
    return _ext_constant_switch_decrease_non_renewable_capacity()


_ext_constant_switch_decrease_non_renewable_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_NON_RENEWABLE_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_decrease_non_renewable_capacity",
)


@component.add(
    name="INITIAL YEAR ENERGY EFFICIENCY INCREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_energy_efficiency_increase"
    },
)
def initial_year_energy_efficiency_increase():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'INITIAL_YEAR_EFFICIENCY_INCREASE*')
    """
    return _ext_constant_initial_year_energy_efficiency_increase()


_ext_constant_initial_year_energy_efficiency_increase = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_EFFICIENCY_INCREASE*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_energy_efficiency_increase",
)


@component.add(
    name="INITIAL YEAR ENERGY SHARE POLICY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_energy_share_policy"},
)
def initial_year_energy_share_policy():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'INITIAL_YEAR_ENERGY_SHARE_POLICY*')
    """
    return _ext_constant_initial_year_energy_share_policy()


_ext_constant_initial_year_energy_share_policy = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_ENERGY_SHARE_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_energy_share_policy",
)


@component.add(
    name="VERY HIGH WIND", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_wind():
    return 1


@component.add(
    name="SWITCH WIND CAPACITY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_wind_capacity"},
)
def switch_wind_capacity():
    return _ext_constant_switch_wind_capacity()


_ext_constant_switch_wind_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SWITCH_WIND_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_switch_wind_capacity",
)


@component.add(
    name="SOLAR CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_solar_capacity": 1},
    other_deps={
        "_integ_solar_capacity": {
            "initial": {"initial_solar_capacity": 1},
            "step": {"new_solar_capacity": 1, "solar_decommissioning": 1},
        }
    },
)
def solar_capacity():
    return _integ_solar_capacity()


_integ_solar_capacity = Integ(
    lambda: new_solar_capacity() - solar_decommissioning(),
    lambda: initial_solar_capacity(),
    "_integ_solar_capacity",
)


@component.add(
    name="MINIMUM DECOMMISSIONING NON RENEWABLE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_non_renewable"},
)
def minimum_decommissioning_non_renewable():
    return _ext_constant_minimum_decommissioning_non_renewable()


_ext_constant_minimum_decommissioning_non_renewable = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_NON_RENEWABLE*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_non_renewable",
)


@component.add(
    name="INITIAL HYDRO CAPACITY",
    units="MW",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_hydro_capacity"},
)
def initial_hydro_capacity():
    return _ext_constant_initial_hydro_capacity()


_ext_constant_initial_hydro_capacity = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "INITIAL_HYDRO_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_hydro_capacity",
)


@component.add(
    name="INITIAL YEAR ROOFTOP POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_rooftop_policy_capacity"},
)
def initial_year_rooftop_policy_capacity():
    return _ext_constant_initial_year_rooftop_policy_capacity()


_ext_constant_initial_year_rooftop_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_ROOFTOP_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_rooftop_policy_capacity",
)


@component.add(
    name="INITIAL YEAR SOLAR POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_solar_policy_capacity"},
)
def initial_year_solar_policy_capacity():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'INITIAL_YEAR_SOLAR_POLICY_CAPACITY*')
    """
    return _ext_constant_initial_year_solar_policy_capacity()


_ext_constant_initial_year_solar_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_SOLAR_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_solar_policy_capacity",
)


@component.add(
    name="INITIAL YEAR WIND POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_wind_policy_capacity"},
)
def initial_year_wind_policy_capacity():
    return _ext_constant_initial_year_wind_policy_capacity()


_ext_constant_initial_year_wind_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_WIND_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_wind_policy_capacity",
)


@component.add(
    name="WIND PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_wind_performance"},
)
def wind_performance():
    return _ext_constant_wind_performance()


_ext_constant_wind_performance = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "WIND_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_wind_performance",
)


@component.add(
    name="SELECT WIND INCENTIVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_wind_incentives"},
)
def select_wind_incentives():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High
    """
    return _ext_constant_select_wind_incentives()


_ext_constant_select_wind_incentives = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_WIND_INCENTIVES*",
    {},
    _root,
    {},
    "_ext_constant_select_wind_incentives",
)


@component.add(
    name="VERY HIGH TAX", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_tax():
    return 1


@component.add(
    name="SOLAR PRODUCTION",
    units="MWh",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"solar_capacity": 1, "solar_performance": 1, "hours_in_a_year": 1},
)
def solar_production():
    return solar_capacity() * solar_performance() * hours_in_a_year()


@component.add(
    name="MINIMUM DECOMMISSIONING HYDRO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_hydro"},
)
def minimum_decommissioning_hydro():
    return _ext_constant_minimum_decommissioning_hydro()


_ext_constant_minimum_decommissioning_hydro = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_HYDRO*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_hydro",
)


@component.add(
    name="TOTAL CARBON EMISSIONS FROM ENERGY",
    units="tCO2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"carbon_emissions": 1},
)
def total_carbon_emissions_from_energy():
    return sum(
        carbon_emissions().rename({"TYPE OF ENERGY": "TYPE OF ENERGY!"}),
        dim=["TYPE OF ENERGY!"],
    )


@component.add(
    name="MINIMUM DECOMMISSIONING WIND",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_wind"},
)
def minimum_decommissioning_wind():
    return _ext_constant_minimum_decommissioning_wind()


_ext_constant_minimum_decommissioning_wind = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_WIND*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_wind",
)


@component.add(
    name="VERY HIGH SOLAR", units="Dmnl", comp_type="Constant", comp_subtype="Normal"
)
def very_high_solar():
    return 1


@component.add(
    name="INITIAL YEAR INCENTIVE FOR ENERGY EMISSIONS DECREASE",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_incentive_for_energy_emissions_decrease"
    },
)
def initial_year_incentive_for_energy_emissions_decrease():
    return _ext_constant_initial_year_incentive_for_energy_emissions_decrease()


_ext_constant_initial_year_incentive_for_energy_emissions_decrease = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_INCENTIVE_EMISSIONS*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_incentive_for_energy_emissions_decrease",
)


@component.add(
    name="INITIAL YEAR DECREASE NON RENEWABLE POLICY CAPACITY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_decrease_non_renewable_policy_capacity"
    },
)
def initial_year_decrease_non_renewable_policy_capacity():
    return _ext_constant_initial_year_decrease_non_renewable_policy_capacity()


_ext_constant_initial_year_decrease_non_renewable_policy_capacity = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "INITIAL_YEAR_NON_RENEWABLE_POLICY_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_decrease_non_renewable_policy_capacity",
)


@component.add(
    name="SOLAR PERFORMANCE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_solar_performance"},
)
def solar_performance():
    return _ext_constant_solar_performance()


_ext_constant_solar_performance = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "SOLAR_PERFORMANCE*",
    {},
    _root,
    {},
    "_ext_constant_solar_performance",
)


@component.add(
    name="WIND CAPACITY",
    units="MW",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_wind_capacity": 1},
    other_deps={
        "_integ_wind_capacity": {
            "initial": {"initial_wind_capacity": 1},
            "step": {"new_wind_capacity": 1, "decommissioning_wind": 1},
        }
    },
)
def wind_capacity():
    return _integ_wind_capacity()


_integ_wind_capacity = Integ(
    lambda: new_wind_capacity() - decommissioning_wind(),
    lambda: initial_wind_capacity(),
    "_integ_wind_capacity",
)


@component.add(
    name="SELECT INCENTIVE FOR ENERGY EMISSIONS DECREASE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_select_incentive_for_energy_emissions_decrease"
    },
)
def select_incentive_for_energy_emissions_decrease():
    """
    0 = Low 1 = Medium 2 = High 3 = Very High GET DIRECT CONSTANTS ('Policy.xlsx', 'ENERGY', 'SELECT_EMISSIONS_DECREASE*')
    """
    return _ext_constant_select_incentive_for_energy_emissions_decrease()


_ext_constant_select_incentive_for_energy_emissions_decrease = ExtConstant(
    r"Policy.xlsx",
    "ENERGY",
    "SELECT_EMISSIONS_DECREASE*",
    {},
    _root,
    {},
    "_ext_constant_select_incentive_for_energy_emissions_decrease",
)


@component.add(
    name="WIND CAPACITY OBJECTIVE",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_wind_incentives": 4,
        "low_wind": 2,
        "desired_objective_for_wind": 5,
        "medium_wind": 1,
        "very_high_wind": 1,
        "high_wind": 1,
    },
)
def wind_capacity_objective():
    return if_then_else(
        select_wind_incentives() == 0,
        lambda: low_wind() * desired_objective_for_wind(),
        lambda: if_then_else(
            select_wind_incentives() == 1,
            lambda: medium_wind() * desired_objective_for_wind(),
            lambda: if_then_else(
                select_wind_incentives() == 2,
                lambda: high_wind() * desired_objective_for_wind(),
                lambda: if_then_else(
                    select_wind_incentives() == 3,
                    lambda: very_high_wind() * desired_objective_for_wind(),
                    lambda: low_wind() * desired_objective_for_wind(),
                ),
            ),
        ),
    )


@component.add(
    name="MINIMUM DECOMMISSIONING SOLAR",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_minimum_decommissioning_solar"},
)
def minimum_decommissioning_solar():
    return _ext_constant_minimum_decommissioning_solar()


_ext_constant_minimum_decommissioning_solar = ExtConstant(
    r"Historical.xlsx",
    "ENERGY",
    "MINIMUM_DECOMMISSIONING_SOLAR*",
    {},
    _root,
    {},
    "_ext_constant_minimum_decommissioning_solar",
)


@component.add(
    name="POPULATION AT RISK",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "exposed_population": 1,
        "cs_population": 1,
        "heatwaves_probability": 1,
    },
)
def population_at_risk():
    return (exposed_population() / cs_population()) * heatwaves_probability()


@component.add(
    name="SOLAR DECOMMISSIONING",
    units="MW",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "time": 1,
        "initial_time": 1,
        "lifetime_solar": 1,
        "delay_solar_initial": 1,
        "delay_solar": 1,
    },
)
def solar_decommissioning():
    return if_then_else(
        time() < initial_time() + lifetime_solar(),
        lambda: delay_solar_initial(),
        lambda: delay_solar(),
    )


@component.add(
    name="RAINFED RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"rainfed_crop_area": 1, "agriculture": 1},
)
def rainfed_ratio():
    return rainfed_crop_area() / agriculture()


@component.add(
    name="TOTAL CS AREA",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agriculture": 1,
        "forest": 1,
        "other": 1,
        "urban": 1,
        "water": 1,
        "wetlands": 1,
    },
)
def total_cs_area():
    return agriculture() + forest() + other() + urban() + water() + wetlands()


@component.add(
    name="DECREASE IN CROP SENSITIVTY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"adaptation_efficiency": 1, "crops_adaptation": 1},
)
def decrease_in_crop_sensitivty():
    return (1 - adaptation_efficiency()) * crops_adaptation()


@component.add(
    name="INCREASE EXPOSED CROPS",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"exposed_crop_rate": 1},
)
def increase_exposed_crops():
    """
    IF THEN ELSE(EXPOSED CROPS<AGRICULTURE*0.995, MIN(CROP RATE*EXPOSED CROPS, AGRICULTURE*CROP RATE),0)
    """
    return exposed_crop_rate()


@component.add(
    name="IRRIGATED RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"irrigated_crop_area": 1, "agriculture": 1},
)
def irrigated_ratio():
    return irrigated_crop_area() / agriculture()


@component.add(
    name="CROPS SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"sensitivity": 3, "decrease_in_crop_sensitivty": 3},
)
def crops_sensitivity():
    """
    IF THEN ELSE(SENSITIVITY-DECREASE IN CROP SENSITIVTY > 1, 1, IF THEN ELSE(SENSITIVITY-DECREASE IN CROP SENSITIVTY < 0, 0, SENSITIVITY-DECREASE IN CROP SENSITIVTY))
    """
    return if_then_else(
        sensitivity() - decrease_in_crop_sensitivty() > 1,
        lambda: 1,
        lambda: if_then_else(
            sensitivity() - decrease_in_crop_sensitivty() < 0,
            lambda: 0,
            lambda: sensitivity() - decrease_in_crop_sensitivty(),
        ),
    )


@component.add(
    name="NORMALIZED PERCENTAGE",
    units="Dmnl",
    subscripts=["CROPS I"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"crop_distribution": 2},
)
def normalized_percentage():
    return crop_distribution() / sum(
        crop_distribution().rename({"CROPS I": "CROPS I!"}), dim=["CROPS I!"]
    )


@component.add(
    name="INITIAL DIVERSIFICATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"normalized_percentage": 1},
)
def initial_diversification():
    return 1 - sum(
        normalized_percentage().rename({"CROPS I": "CROPS I!"}) ** 2, dim=["CROPS I!"]
    )


@component.add(
    name="CROPS IN RISK",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"exposed_crops": 1, "agriculture": 1, "drought_probability": 1},
)
def crops_in_risk():
    return (exposed_crops() / agriculture()) * drought_probability()


@component.add(
    name="CROP DISTRIBUTION",
    units="Dmnl",
    subscripts=["CROPS I"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_crop_distribution"},
)
def crop_distribution():
    return _ext_constant_crop_distribution()


_ext_constant_crop_distribution = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "CROP_DISTRIBUTION*",
    {"CROPS I": _subscript_dict["CROPS I"]},
    _root,
    {"CROPS I": _subscript_dict["CROPS I"]},
    "_ext_constant_crop_distribution",
)


@component.add(
    name="HIGH AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def high_afforestation_policy():
    return 0.75


@component.add(
    name="AFFORESTATION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agriculture": 1,
        "afforestation_ratio": 1,
        "subsides_for_afforestation": 1,
    },
)
def afforestation():
    return agriculture() * afforestation_ratio() * subsides_for_afforestation()


@component.add(
    name="AFFORESTATION RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_afforestation_policy": 2,
        "initial_year_for_afforestation_policy": 1,
        "time": 2,
        "final_year_for_afforestation_policy": 1,
    },
)
def afforestation_ratio():
    return if_then_else(
        switch_afforestation_policy() == 0,
        lambda: 0.0001,
        lambda: if_then_else(
            np.logical_and(
                switch_afforestation_policy() == 1,
                np.logical_or(
                    time() < initial_year_for_afforestation_policy(),
                    time() > final_year_for_afforestation_policy(),
                ),
            ),
            lambda: 0.0001,
            lambda: 0.001,
        ),
    )


@component.add(
    name="AFFORESTATION SUBSIDES",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_afforestation_policy": 4,
        "low_afforestation_policy": 2,
        "high_afforestation_policy": 1,
        "medium_afforestation_policy": 1,
        "very_high_afforestation_policy": 1,
    },
)
def afforestation_subsides():
    """
    Valor entre 0 y 1
    """
    return if_then_else(
        select_afforestation_policy() == 0,
        lambda: low_afforestation_policy(),
        lambda: if_then_else(
            select_afforestation_policy() == 1,
            lambda: medium_afforestation_policy(),
            lambda: if_then_else(
                select_afforestation_policy() == 2,
                lambda: high_afforestation_policy(),
                lambda: if_then_else(
                    select_afforestation_policy() == 3,
                    lambda: very_high_afforestation_policy(),
                    lambda: low_afforestation_policy(),
                ),
            ),
        ),
    )


@component.add(
    name="AGRICULTURAL DEMAND RATIO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_agricultural_demand_ratio"},
)
def agricultural_demand_ratio():
    return _ext_constant_agricultural_demand_ratio()


_ext_constant_agricultural_demand_ratio = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "AGRICULTURAL_DEMAND_RATIO*",
    {},
    _root,
    {},
    "_ext_constant_agricultural_demand_ratio",
)


@component.add(
    name="MEDIUM AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def medium_afforestation_policy():
    return 0.5


@component.add(
    name="VERY HIGH AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def very_high_afforestation_policy():
    return 1


@component.add(
    name="INCREASE OBJECTIVE FOR PROTECTED FOREST LAND",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_increase_objective_for_protected_forest_land"
    },
)
def increase_objective_for_protected_forest_land():
    return _ext_constant_increase_objective_for_protected_forest_land()


_ext_constant_increase_objective_for_protected_forest_land = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INCREASE_OBJECTIVE_FOR_FOREST_PROTECTED_LAND*",
    {},
    _root,
    {},
    "_ext_constant_increase_objective_for_protected_forest_land",
)


@component.add(
    name="INITIAL FOREST PROTECTED LAND",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_forest_protected_land"},
)
def initial_forest_protected_land():
    return _ext_constant_initial_forest_protected_land()


_ext_constant_initial_forest_protected_land = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_PROTECTED_FOREST_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_forest_protected_land",
)


@component.add(
    name="SUBSIDES FOR AFFORESTATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_afforestation_policy": 2,
        "afforestation_subsides": 1,
        "initial_year_for_afforestation_policy": 1,
        "time": 2,
        "final_year_for_afforestation_policy": 1,
    },
)
def subsides_for_afforestation():
    return if_then_else(
        switch_afforestation_policy() == 0,
        lambda: 0.0001,
        lambda: if_then_else(
            np.logical_and(
                switch_afforestation_policy() == 1,
                np.logical_or(
                    time() < initial_year_for_afforestation_policy(),
                    time() > final_year_for_afforestation_policy(),
                ),
            ),
            lambda: 0.0001,
            lambda: afforestation_subsides(),
        ),
    )


@component.add(
    name="FINAL YEAR FOR AFFORESTATION POLICY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_for_afforestation_policy"},
)
def final_year_for_afforestation_policy():
    return _ext_constant_final_year_for_afforestation_policy()


_ext_constant_final_year_for_afforestation_policy = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "FINAL_YEAR_FOR_AFFORESTATION_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_final_year_for_afforestation_policy",
)


@component.add(
    name="SELECT AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_afforestation_policy"},
)
def select_afforestation_policy():
    return _ext_constant_select_afforestation_policy()


_ext_constant_select_afforestation_policy = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SELECT_AFFORESTATION_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_select_afforestation_policy",
)


@component.add(
    name="LOW AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def low_afforestation_policy():
    return 0.25


@component.add(
    name="INITIAL YEAR FOR AFFORESTATION POLICY",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_for_afforestation_policy"},
)
def initial_year_for_afforestation_policy():
    return _ext_constant_initial_year_for_afforestation_policy()


_ext_constant_initial_year_for_afforestation_policy = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INITIAL_YEAR_FOR_AFFORESTATION_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_for_afforestation_policy",
)


@component.add(
    name="FINAL YEAR FOR PROTECTED FOREST LAND",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_for_protected_forest_land"},
)
def final_year_for_protected_forest_land():
    return _ext_constant_final_year_for_protected_forest_land()


_ext_constant_final_year_for_protected_forest_land = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "FINAL_YEAR_FOR_FOREST_PROTECTED_LAND*",
    {},
    _root,
    {},
    "_ext_constant_final_year_for_protected_forest_land",
)


@component.add(
    name="SWITCH AFFORESTATION POLICY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_afforestation_policy"},
)
def switch_afforestation_policy():
    return _ext_constant_switch_afforestation_policy()


_ext_constant_switch_afforestation_policy = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SWITCH_AFFORESTATION_POLICY*",
    {},
    _root,
    {},
    "_ext_constant_switch_afforestation_policy",
)


@component.add(
    name="SWITCH PROTECTED FOREST LAND",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_protected_forest_land"},
)
def switch_protected_forest_land():
    return _ext_constant_switch_protected_forest_land()


_ext_constant_switch_protected_forest_land = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "SWITCH_FOREST_PROTECTED_LAND*",
    {},
    _root,
    {},
    "_ext_constant_switch_protected_forest_land",
)


@component.add(
    name="INITIAL YEAR FOR PROTECTED FOREST LAND",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_for_protected_forest_land"},
)
def initial_year_for_protected_forest_land():
    return _ext_constant_initial_year_for_protected_forest_land()


_ext_constant_initial_year_for_protected_forest_land = ExtConstant(
    r"Policy.xlsx",
    "Land system",
    "INITIAL_YEAR_FOR_FOREST_PROTECTED_LAND*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_for_protected_forest_land",
)


@component.add(
    name="ACCESIBILITY RESTRICTION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"distance": 1},
)
def accesibility_restriction():
    """

    !km
    !dmnl
    """
    return float(np.exp(-0.5 * distance()))


@component.add(
    name="CONVERSION SPEED WETLANDS TO URBAN",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"wetlands": 4, "protected_wetlands_not_corrected": 4, "urban": 2},
)
def conversion_speed_wetlands_to_urban():
    return if_then_else(
        (wetlands() - protected_wetlands_not_corrected())
        / ((wetlands() - protected_wetlands_not_corrected()) + urban())
        > 0,
        lambda: (wetlands() - protected_wetlands_not_corrected())
        / ((wetlands() - protected_wetlands_not_corrected()) + urban()),
        lambda: 0,
    )


@component.add(
    name="DISTANCE",
    units="km",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_distance"},
)
def distance():
    return _ext_constant_distance()


_ext_constant_distance = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "DISTANCE*",
    {},
    _root,
    {},
    "_ext_constant_distance",
)


@component.add(
    name="PERCENTAGE OF PROTECTED LAND",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"protected_forest_land_not_corrected": 1, "forest": 1},
)
def percentage_of_protected_land():
    return protected_forest_land_not_corrected() / forest()


@component.add(
    name="ENVIRONMENTAL RESTRICTION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"percentage_of_protected_land": 1},
)
def environmental_restriction():
    return float(np.exp(-0.8 * percentage_of_protected_land()))


@component.add(
    name="AGRICULTURAL EXPANSION",
    units="km2",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "agricultural_demand_ratio": 1,
        "environmental_restriction": 1,
        "available_forest_land": 1,
    },
)
def agricultural_expansion():
    return (
        agricultural_demand_ratio()
        * environmental_restriction()
        * available_forest_land()
    )


@component.add(
    name="AGRICULTURAL PRESSURE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"available_land": 2, "agricultural_demand_ratio": 1},
)
def agricultural_pressure():
    return if_then_else(
        available_land() > 0,
        lambda: agricultural_demand_ratio() / available_land(),
        lambda: 0,
    )


@component.add(
    name="CONVERSION SPEED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"other": 4, "urban": 2},
)
def conversion_speed():
    return if_then_else(
        other() / (other() + urban()) > 0,
        lambda: other() / (other() + urban()),
        lambda: 0,
    )


@component.add(
    name="EXPANSION RATIO",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_expansion_ratio"},
)
def expansion_ratio():
    return _ext_constant_expansion_ratio()


_ext_constant_expansion_ratio = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "EXPANSION_RATIO*",
    {},
    _root,
    {},
    "_ext_constant_expansion_ratio",
)


@component.add(
    name="FOREST",
    units="km2",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_forest": 1},
    other_deps={
        "_integ_forest": {
            "initial": {"initial_forest_area": 1},
            "step": {
                "afforestation": 1,
                "agricultural_expansion": 1,
                "forest_to_water": 1,
            },
        }
    },
)
def forest():
    return _integ_forest()


_integ_forest = Integ(
    lambda: afforestation() - agricultural_expansion() - forest_to_water(),
    lambda: initial_forest_area(),
    "_integ_forest",
)


@component.add(
    name="INITIAL AGRICULTURAL AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_agricultural_area"},
)
def initial_agricultural_area():
    return _ext_constant_initial_agricultural_area()


_ext_constant_initial_agricultural_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_AGRICULTURAL_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_agricultural_area",
)


@component.add(
    name="INITIAL FOREST AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_forest_area"},
)
def initial_forest_area():
    return _ext_constant_initial_forest_area()


_ext_constant_initial_forest_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_FOREST_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_forest_area",
)


@component.add(
    name="INITIAL OTHER AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_other_area"},
)
def initial_other_area():
    return _ext_constant_initial_other_area()


_ext_constant_initial_other_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_OTHER_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_other_area",
)


@component.add(
    name="INITIAL URBAN AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_urban_area"},
)
def initial_urban_area():
    return _ext_constant_initial_urban_area()


_ext_constant_initial_urban_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_URBAN_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_urban_area",
)


@component.add(
    name="INITIAL WATER AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_water_area"},
)
def initial_water_area():
    return _ext_constant_initial_water_area()


_ext_constant_initial_water_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_WATER_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_water_area",
)


@component.add(
    name="INITIAL WETLANDS AREA",
    units="km2",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_wetlands_area"},
)
def initial_wetlands_area():
    return _ext_constant_initial_wetlands_area()


_ext_constant_initial_wetlands_area = ExtConstant(
    r"Historical.xlsx",
    "Land system",
    "INITIAL_WETLANDS_AREA*",
    {},
    _root,
    {},
    "_ext_constant_initial_wetlands_area",
)


@component.add(
    name="POPULATION ADAPTATION BASELINE",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gdp_per_capita": 2, "educational_level": 2},
)
def population_adaptation_baseline():
    """
    (EDUCATIONAL LEVEL+PIB PER CAPITA)/2
    """
    return if_then_else(
        gdp_per_capita() > 1,
        lambda: (1 + educational_level()) / 2,
        lambda: (educational_level() + gdp_per_capita()) / 2,
    )


@component.add(
    name="POPULATION ADAPTATION CAPACITY",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_population_adaptation_capacity": 1},
    other_deps={
        "_integ_population_adaptation_capacity": {
            "initial": {"population_adaptation_baseline": 1},
            "step": {
                "increase_in_population_adaptation_capacity": 1,
                "decrease_in_population_adaptation_capacity": 1,
            },
        }
    },
)
def population_adaptation_capacity():
    return _integ_population_adaptation_capacity()


_integ_population_adaptation_capacity = Integ(
    lambda: increase_in_population_adaptation_capacity()
    - decrease_in_population_adaptation_capacity(),
    lambda: population_adaptation_baseline(),
    "_integ_population_adaptation_capacity",
)


@component.add(
    name="ADAPTATION DECAY RATE 1",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_adaptation_decay_rate_1"},
)
def adaptation_decay_rate_1():
    return _ext_constant_adaptation_decay_rate_1()


_ext_constant_adaptation_decay_rate_1 = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "ADAPTATION_DECAY_RATE_1*",
    {},
    _root,
    {},
    "_ext_constant_adaptation_decay_rate_1",
)


@component.add(
    name="ADAPTATION FOR POPULATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_adaptation_capacity": 1, "pop_adjusted_ratio": 1},
)
def adaptation_for_population():
    return population_adaptation_capacity() * pop_adjusted_ratio()


@component.add(
    name="DELADAPTATION INVESTMENT RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="Normal",
)
def deladaptation_investment_rate():
    return 0.01


@component.add(
    name="RISK INDEX POPULATION",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_risk_index_population": 1},
    other_deps={
        "_smooth_risk_index_population": {
            "initial": {
                "sensitivity_for_population": 3,
                "adaptation_for_population": 3,
                "population_at_risk": 2,
            },
            "step": {
                "sensitivity_for_population": 3,
                "adaptation_for_population": 3,
                "population_at_risk": 2,
            },
        }
    },
)
def risk_index_population():
    return _smooth_risk_index_population()


_smooth_risk_index_population = Smooth(
    lambda: if_then_else(
        sensitivity_for_population() - adaptation_for_population() < 0,
        lambda: 0,
        lambda: if_then_else(
            population_at_risk()
            * (sensitivity_for_population() - adaptation_for_population())
            > 1,
            lambda: 1,
            lambda: population_at_risk()
            * (sensitivity_for_population() - adaptation_for_population()),
        ),
    ),
    lambda: 2,
    lambda: if_then_else(
        sensitivity_for_population() - adaptation_for_population() < 0,
        lambda: 0,
        lambda: if_then_else(
            population_at_risk()
            * (sensitivity_for_population() - adaptation_for_population())
            > 1,
            lambda: 1,
            lambda: population_at_risk()
            * (sensitivity_for_population() - adaptation_for_population()),
        ),
    ),
    lambda: 1,
    "_smooth_risk_index_population",
)


@component.add(
    name="POPULATION EXPOSED GROWTH",
    units="Dmnl",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_population_exposed_growth",
        "__data__": "_ext_data_population_exposed_growth",
        "time": 1,
    },
)
def population_exposed_growth():
    return _ext_data_population_exposed_growth(time())


_ext_data_population_exposed_growth = ExtData(
    r"Historical.xlsx",
    "Population system",
    "POPULATION_EXPOSED_GROWTH_TIME",
    "POPULATION_EXPOSED_GROWTH",
    "interpolate",
    {},
    _root,
    {},
    "_ext_data_population_exposed_growth",
)


@component.add(
    name="POP ADJUSTED RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"pop_ratio": 2},
)
def pop_adjusted_ratio():
    return if_then_else(pop_ratio() > 1, lambda: 1, lambda: pop_ratio())


@component.add(
    name="POPULATION ADAPTATION EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_population_adaptation_efficiency"},
)
def population_adaptation_efficiency():
    return _ext_constant_population_adaptation_efficiency()


_ext_constant_population_adaptation_efficiency = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "POPULATION_ADAPTATION_EFFICIENCY*",
    {},
    _root,
    {},
    "_ext_constant_population_adaptation_efficiency",
)


@component.add(
    name="DECREASE IN POPULATION ADAPTATION CAPACITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_adaptation_capacity": 1, "adaptation_decay_rate_1": 1},
)
def decrease_in_population_adaptation_capacity():
    return population_adaptation_capacity() * adaptation_decay_rate_1()


@component.add(
    name="HEATWAVES PROBABILITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_heatwaves_probabilities": 2,
        "ssp245_heatwaves": 1,
        "ssp585_heatwaves": 1,
    },
)
def heatwaves_probability():
    return if_then_else(
        select_heatwaves_probabilities() == 0,
        lambda: ssp245_heatwaves(),
        lambda: if_then_else(
            select_heatwaves_probabilities() == 1, lambda: ssp585_heatwaves(), lambda: 1
        ),
    )


@component.add(
    name="IMPACT IN POPULATION",
    units="Deaths",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"risk_index_population": 2, "vulnerable_population": 3},
)
def impact_in_population():
    return if_then_else(
        risk_index_population() > 0.7,
        lambda: 0.1 * vulnerable_population(),
        lambda: if_then_else(
            risk_index_population() < 0.1,
            lambda: 0.02 * vulnerable_population(),
            lambda: 0.05 * vulnerable_population(),
        ),
    )


@component.add(
    name="SELECT HEATWAVES PROBABILITIES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_heatwaves_probabilities"},
)
def select_heatwaves_probabilities():
    """
    0 = SSP245 1 = SSP585
    """
    return _ext_constant_select_heatwaves_probabilities()


_ext_constant_select_heatwaves_probabilities = ExtConstant(
    r"Policy.xlsx",
    "Population system",
    "SELECT_HEAT_WAVES_PROBABILITIES*",
    {},
    _root,
    {},
    "_ext_constant_select_heatwaves_probabilities",
)


@component.add(
    name="POP RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_adaptation_capacity": 1},
)
def pop_ratio():
    return 1 / population_adaptation_capacity()


@component.add(
    name="SSP245 HEATWAVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ssp245_heatwaves"},
)
def ssp245_heatwaves():
    return _ext_constant_ssp245_heatwaves()


_ext_constant_ssp245_heatwaves = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "SSP245_HEATWAVES*",
    {},
    _root,
    {},
    "_ext_constant_ssp245_heatwaves",
)


@component.add(
    name="SSP585 HEATWAVES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ssp585_heatwaves"},
)
def ssp585_heatwaves():
    return _ext_constant_ssp585_heatwaves()


_ext_constant_ssp585_heatwaves = ExtConstant(
    r"Historical.xlsx",
    "Population system",
    "SSP585_HEATWAVES*",
    {},
    _root,
    {},
    "_ext_constant_ssp585_heatwaves",
)


@component.add(
    name="INCREASE EXPOSED POPULATION",
    units="Inhabitants",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_exposed_growth": 1},
)
def increase_exposed_population():
    return population_exposed_growth()


@component.add(
    name="VULNERABLE POPULATION",
    units="Inhabitants",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"population_lower_5": 1, "population_higher_65": 1},
)
def vulnerable_population():
    return population_lower_5() + population_higher_65()


@component.add(
    name="SSP585 DROUGHTS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ssp585_droughts"},
)
def ssp585_droughts():
    return _ext_constant_ssp585_droughts()


_ext_constant_ssp585_droughts = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "SSP585_DROUGHTS*",
    {},
    _root,
    {},
    "_ext_constant_ssp585_droughts",
)


@component.add(
    name="DROUGHT PROBABILITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "select_droughts_probabilities": 2,
        "ssp245_droughts": 1,
        "ssp585_droughts": 1,
    },
)
def drought_probability():
    return if_then_else(
        select_droughts_probabilities() == 0,
        lambda: ssp245_droughts(),
        lambda: if_then_else(
            select_droughts_probabilities() == 1, lambda: ssp585_droughts(), lambda: 1
        ),
    )


@component.add(
    name="FINAL PRODUCTION",
    units="kg/ha·year",
    subscripts=["CROPS I"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"baseline_production": 1, "impact_in_production": 1},
)
def final_production():
    return baseline_production() - impact_in_production()


@component.add(
    name="SSP245 DROUGHTS",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ssp245_droughts"},
)
def ssp245_droughts():
    return _ext_constant_ssp245_droughts()


_ext_constant_ssp245_droughts = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "SSP245_DROUGHTS*",
    {},
    _root,
    {},
    "_ext_constant_ssp245_droughts",
)


@component.add(
    name="SELECT DROUGHTS PROBABILITIES",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_select_droughts_probabilities"},
)
def select_droughts_probabilities():
    """
    0 = SSP245 1 = SSP585
    """
    return _ext_constant_select_droughts_probabilities()


_ext_constant_select_droughts_probabilities = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "SELECT_DROUGHTS_PROBABILITIES*",
    {},
    _root,
    {},
    "_ext_constant_select_droughts_probabilities",
)


@component.add(
    name="INCREASE IN DIVERSIFICATION LEVEL",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_crop_diversification": 2,
        "diversification_objective": 1,
        "initial_year_for_crop_diversification": 2,
        "time": 2,
        "final_year_for_crop_diversification": 2,
    },
)
def increase_in_diversification_level():
    return if_then_else(
        switch_crop_diversification() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_crop_diversification() == 1,
                np.logical_or(
                    time() < initial_year_for_crop_diversification(),
                    time() > final_year_for_crop_diversification(),
                ),
            ),
            lambda: 0,
            lambda: diversification_objective()
            / (
                final_year_for_crop_diversification()
                - initial_year_for_crop_diversification()
            ),
        ),
    )


@component.add(
    name="FINAL YEAR FOR CROP ADAPTATION BUDGET",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_for_crop_adaptation_budget"},
)
def final_year_for_crop_adaptation_budget():
    return _ext_constant_final_year_for_crop_adaptation_budget()


_ext_constant_final_year_for_crop_adaptation_budget = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "FINAL_YEAR_FOR_CROP_ADAPTATION_BUDGET*",
    {},
    _root,
    {},
    "_ext_constant_final_year_for_crop_adaptation_budget",
)


@component.add(
    name="FINAL YEAR FOR CROP DIVERSIFICATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_final_year_for_crop_diversification"},
)
def final_year_for_crop_diversification():
    return _ext_constant_final_year_for_crop_diversification()


_ext_constant_final_year_for_crop_diversification = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "FINAL_YEAR_FOR_CROP_DIVERSIFICATION*",
    {},
    _root,
    {},
    "_ext_constant_final_year_for_crop_diversification",
)


@component.add(
    name="INITIAL YEAR FOR CROP ADAPTATION BUDGET",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_initial_year_for_crop_adaptation_budget"
    },
)
def initial_year_for_crop_adaptation_budget():
    return _ext_constant_initial_year_for_crop_adaptation_budget()


_ext_constant_initial_year_for_crop_adaptation_budget = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "INITIAL_YEAR_FOR_CROP_ADAPTATION_BUDGET*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_for_crop_adaptation_budget",
)


@component.add(
    name="SWITCH CROP DIVERSIFICATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_crop_diversification"},
)
def switch_crop_diversification():
    """
    1 = ACTIVO 0 = DESACTIVO
    """
    return _ext_constant_switch_crop_diversification()


_ext_constant_switch_crop_diversification = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "SWITCH_CROP_DIVERSIFICATION*",
    {},
    _root,
    {},
    "_ext_constant_switch_crop_diversification",
)


@component.add(
    name="ASIGNED BUDGET",
    units="Million Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "switch_adaptation_budget": 2,
        "initial_year_for_crop_adaptation_budget": 2,
        "time": 2,
        "final_year_for_crop_adaptation_budget": 2,
        "investment_objective_in_crop_adaptation": 1,
    },
)
def asigned_budget():
    return if_then_else(
        switch_adaptation_budget() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                switch_adaptation_budget() == 1,
                np.logical_or(
                    time() < initial_year_for_crop_adaptation_budget(),
                    time() > final_year_for_crop_adaptation_budget(),
                ),
            ),
            lambda: 0,
            lambda: investment_objective_in_crop_adaptation()
            / (
                final_year_for_crop_adaptation_budget()
                - initial_year_for_crop_adaptation_budget()
            ),
        ),
    )


@component.add(
    name="DIVERSIFICATION OBJECTIVE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_diversification_objective"},
)
def diversification_objective():
    """
    Valor entre 0 y 1. Mejor cuanto más cerca de 1
    """
    return _ext_constant_diversification_objective()


_ext_constant_diversification_objective = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "DIVERSIFICATION_OBJECTIVE*",
    {},
    _root,
    {},
    "_ext_constant_diversification_objective",
)


@component.add(
    name="SWITCH ADAPTATION BUDGET",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_switch_adaptation_budget"},
)
def switch_adaptation_budget():
    """
    1 = ACTIVO 0 = DESACTIVO GET DIRECT CONSTANTS ('Policy.xlsx', 'Crop system', 'SWITCH_ADAPTATION_BUDGET*')
    """
    return _ext_constant_switch_adaptation_budget()


_ext_constant_switch_adaptation_budget = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "SWITCH_ADAPTATION_BUDGET*",
    {},
    _root,
    {},
    "_ext_constant_switch_adaptation_budget",
)


@component.add(
    name="INITIAL YEAR FOR CROP DIVERSIFICATION",
    units="Year",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_year_for_crop_diversification"},
)
def initial_year_for_crop_diversification():
    return _ext_constant_initial_year_for_crop_diversification()


_ext_constant_initial_year_for_crop_diversification = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "INITIAL_YEAR_FOR_CROP_DIVERSIFICATION*",
    {},
    _root,
    {},
    "_ext_constant_initial_year_for_crop_diversification",
)


@component.add(
    name="INVESTMENT OBJECTIVE IN CROP ADAPTATION",
    units="Million Euros",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_investment_objective_in_crop_adaptation"
    },
)
def investment_objective_in_crop_adaptation():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Crop system', 'INVESTMENT_OBJECTIVE_IN_CROP_ADAPTATION*')
    """
    return _ext_constant_investment_objective_in_crop_adaptation()


_ext_constant_investment_objective_in_crop_adaptation = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "INVESTMENT_OBJECTIVE_IN_CROP_ADAPTATION*",
    {},
    _root,
    {},
    "_ext_constant_investment_objective_in_crop_adaptation",
)


@component.add(
    name="FINAL DIVERSIFICATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"diversification_level": 3},
)
def final_diversification():
    return if_then_else(
        diversification_level() > 1,
        lambda: 1,
        lambda: if_then_else(
            diversification_level() < 0, lambda: 0, lambda: diversification_level()
        ),
    )


@component.add(
    name="SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "baseline_sensitivity": 2,
        "diversification_efficiency": 1,
        "final_diversification": 1,
    },
)
def sensitivity():
    return (
        baseline_sensitivity()
        * (1 - diversification_efficiency() * final_diversification())
        + baseline_sensitivity()
    )


@component.add(
    name="RISK INDEX CROPS",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_risk_index_crops": 1},
    other_deps={
        "_smooth_risk_index_crops": {
            "initial": {
                "crops_in_risk": 3,
                "crops_adaptation": 3,
                "crops_sensitivity": 3,
            },
            "step": {"crops_in_risk": 3, "crops_adaptation": 3, "crops_sensitivity": 3},
        }
    },
)
def risk_index_crops():
    return _smooth_risk_index_crops()


_smooth_risk_index_crops = Smooth(
    lambda: if_then_else(
        (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity()) / 3 > 1,
        lambda: 1,
        lambda: if_then_else(
            (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity()) / 3 < 0,
            lambda: 0,
            lambda: (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity())
            / 3,
        ),
    ),
    lambda: 2,
    lambda: if_then_else(
        (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity()) / 3 > 1,
        lambda: 1,
        lambda: if_then_else(
            (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity()) / 3 < 0,
            lambda: 0,
            lambda: (crops_in_risk() + -1 * crops_adaptation() + crops_sensitivity())
            / 3,
        ),
    ),
    lambda: 1,
    "_smooth_risk_index_crops",
)


@component.add(
    name="CROP ADAPTATION CAPACITY",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_crop_adaptation_capacity": 1},
    other_deps={
        "_integ_crop_adaptation_capacity": {
            "initial": {},
            "step": {
                "increase_in_adaptation_capacity": 1,
                "decrease_in_adaptation_capacity": 1,
            },
        }
    },
)
def crop_adaptation_capacity():
    return _integ_crop_adaptation_capacity()


_integ_crop_adaptation_capacity = Integ(
    lambda: increase_in_adaptation_capacity() - decrease_in_adaptation_capacity(),
    lambda: 0.5,
    "_integ_crop_adaptation_capacity",
)


@component.add(
    name="IMPACT IN PRODUCTION",
    units="kg/ha·year",
    subscripts=["CROPS I"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"risk_index_crops": 2, "baseline_production": 3},
)
def impact_in_production():
    return if_then_else(
        risk_index_crops() > 0.7,
        lambda: 0.25 * baseline_production(),
        lambda: if_then_else(
            risk_index_crops() < 0.1,
            lambda: 0.05 * baseline_production(),
            lambda: 0.15 * baseline_production(),
        ),
    )


@component.add(
    name="CROPS ADAPTATION",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"crop_adaptation_capacity": 1, "adjusted_ratio": 1},
)
def crops_adaptation():
    return crop_adaptation_capacity() * adjusted_ratio()


@component.add(
    name="ADJUSTED RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ratio": 2},
)
def adjusted_ratio():
    return if_then_else(ratio() > 1, lambda: 1, lambda: ratio())


@component.add(
    name="DECREASE IN ADAPTATION CAPACITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"crop_adaptation_capacity": 1, "decay_rate_1": 1},
)
def decrease_in_adaptation_capacity():
    return crop_adaptation_capacity() * decay_rate_1()


@component.add(
    name="RATIO",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"crop_adaptation_capacity": 1},
)
def ratio():
    return 1 / crop_adaptation_capacity()


@component.add(
    name="DIVERSIFICATION LEVEL",
    units="Dmnl",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_diversification_level": 1},
    other_deps={
        "_integ_diversification_level": {
            "initial": {"initial_diversification": 1},
            "step": {
                "increase_in_diversification_level": 1,
                "decrease_in_diversification_level": 1,
            },
        }
    },
)
def diversification_level():
    return _integ_diversification_level()


_integ_diversification_level = Integ(
    lambda: increase_in_diversification_level() - decrease_in_diversification_level(),
    lambda: initial_diversification(),
    "_integ_diversification_level",
)


@component.add(
    name="BASELINE SENSITIVITY",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "irrigated_ratio": 1,
        "irrigated_vs_rainfed": 1,
        "rainfed_ratio": 1,
        "watercrop_ratio": 1,
    },
)
def baseline_sensitivity():
    return (
        irrigated_ratio() + irrigated_vs_rainfed() + rainfed_ratio() + watercrop_ratio()
    ) / 4


@component.add(
    name="IRRIGATED VS RAINFED",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"rainfed_crop_area": 3, "irrigated_crop_area": 2},
)
def irrigated_vs_rainfed():
    return if_then_else(
        rainfed_crop_area() == 0,
        lambda: 1,
        lambda: if_then_else(
            irrigated_crop_area() / rainfed_crop_area() > 1,
            lambda: 1,
            lambda: irrigated_crop_area() / rainfed_crop_area(),
        ),
    )


@component.add(
    name="ADAPTATION EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_adaptation_efficiency"},
)
def adaptation_efficiency():
    return _ext_constant_adaptation_efficiency()


_ext_constant_adaptation_efficiency = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "ADAPTATION_EFFICIENCY*",
    {},
    _root,
    {},
    "_ext_constant_adaptation_efficiency",
)


@component.add(
    name="ADAPTATION INVESTMENT RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_adaptation_investment_rate"},
)
def adaptation_investment_rate():
    return _ext_constant_adaptation_investment_rate()


_ext_constant_adaptation_investment_rate = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "ADAPTATION_INVESTMENT_RATE*",
    {},
    _root,
    {},
    "_ext_constant_adaptation_investment_rate",
)


@component.add(
    name="BASELINE PRODUCTION",
    units="kg/ha·year",
    subscripts=["CROPS I"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_baseline_production"},
)
def baseline_production():
    return _ext_constant_baseline_production()


_ext_constant_baseline_production = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "BASELINE_PRODUCTION*",
    {"CROPS I": _subscript_dict["CROPS I"]},
    _root,
    {"CROPS I": _subscript_dict["CROPS I"]},
    "_ext_constant_baseline_production",
)


@component.add(
    name="EXPOSED CROP RATE",
    units="Dmnl",
    comp_type="Data",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_data_exposed_crop_rate",
        "__data__": "_ext_data_exposed_crop_rate",
        "time": 1,
    },
)
def exposed_crop_rate():
    return _ext_data_exposed_crop_rate(time())


_ext_data_exposed_crop_rate = ExtData(
    r"Historical.xlsx",
    "Crop system",
    "EXPOSED_CROP_RATE_TIME",
    "EXPOSED_CROP_RATE",
    None,
    {},
    _root,
    {},
    "_ext_data_exposed_crop_rate",
)


@component.add(
    name="DECAY RATE 1",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_decay_rate_1"},
)
def decay_rate_1():
    return _ext_constant_decay_rate_1()


_ext_constant_decay_rate_1 = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "DECAY_RATE_1*",
    {},
    _root,
    {},
    "_ext_constant_decay_rate_1",
)


@component.add(
    name="DECAY RATE 2",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_decay_rate_2"},
)
def decay_rate_2():
    return _ext_constant_decay_rate_2()


_ext_constant_decay_rate_2 = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "DECAY_RATE_2*",
    {},
    _root,
    {},
    "_ext_constant_decay_rate_2",
)


@component.add(
    name="DECREASE IN DIVERSIFICATION LEVEL",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"decay_rate_2": 1, "diversification_level": 1},
)
def decrease_in_diversification_level():
    return decay_rate_2() * diversification_level()


@component.add(
    name="DIVERSIFICATION EFFICIENCY",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_diversification_efficiency"},
)
def diversification_efficiency():
    return _ext_constant_diversification_efficiency()


_ext_constant_diversification_efficiency = ExtConstant(
    r"Historical.xlsx",
    "Crop system",
    "DIVERSIFICATION_EFFICIENCY*",
    {},
    _root,
    {},
    "_ext_constant_diversification_efficiency",
)


@component.add(
    name="INVESTMENT EXECUTION RATE FOR CROPLAND ADAPTATION",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={
        "__external__": "_ext_constant_investment_execution_rate_for_cropland_adaptation"
    },
)
def investment_execution_rate_for_cropland_adaptation():
    """
    GET DIRECT CONSTANTS ('Policy.xlsx', 'Crop system', 'EXECUTION_RATE*')
    """
    return _ext_constant_investment_execution_rate_for_cropland_adaptation()


_ext_constant_investment_execution_rate_for_cropland_adaptation = ExtConstant(
    r"Policy.xlsx",
    "Crop system",
    "EXECUTION_RATE*",
    {},
    _root,
    {},
    "_ext_constant_investment_execution_rate_for_cropland_adaptation",
)


@component.add(
    name="INCREASE IN ADAPTATION CAPACITY",
    units="Million Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"investment_funds_in_adaptation": 7, "adaptation_investment_rate": 4},
)
def increase_in_adaptation_capacity():
    """
    IF THEN ELSE(INVESTMENT FUNDS IN ADAPTATION=0, 0 , ((ADAPTATION INVESTMENT RATE*INVESTMENT FUNDS IN ADAPTATION)/INVESTMENT FUNDS IN ADAPTATION))
    """
    return if_then_else(
        investment_funds_in_adaptation() == 0,
        lambda: 0,
        lambda: if_then_else(
            np.logical_and(
                investment_funds_in_adaptation() > 0,
                investment_funds_in_adaptation() <= 20,
            ),
            lambda: 0.25 * adaptation_investment_rate(),
            lambda: if_then_else(
                np.logical_and(
                    investment_funds_in_adaptation() > 20,
                    investment_funds_in_adaptation() <= 75,
                ),
                lambda: 0.5 * adaptation_investment_rate(),
                lambda: if_then_else(
                    np.logical_and(
                        investment_funds_in_adaptation() > 75,
                        investment_funds_in_adaptation() <= 200,
                    ),
                    lambda: 0.75 * adaptation_investment_rate(),
                    lambda: adaptation_investment_rate(),
                ),
            ),
        ),
    )


@component.add(
    name="INVESTMENT FUNDS IN ADAPTATION",
    units="Million Euros",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_investment_funds_in_adaptation": 1},
    other_deps={
        "_integ_investment_funds_in_adaptation": {
            "initial": {},
            "step": {"asigned_budget": 1, "used_budget": 1},
        }
    },
)
def investment_funds_in_adaptation():
    return _integ_investment_funds_in_adaptation()


_integ_investment_funds_in_adaptation = Integ(
    lambda: asigned_budget() - used_budget(),
    lambda: 0,
    "_integ_investment_funds_in_adaptation",
)


@component.add(
    name="USED BUDGET",
    units="Million Euros",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "investment_execution_rate_for_cropland_adaptation": 1,
        "investment_funds_in_adaptation": 1,
    },
)
def used_budget():
    return (
        investment_execution_rate_for_cropland_adaptation()
        * investment_funds_in_adaptation()
    )


@component.add(
    name="INFILTRATION RATE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_infiltration_rate"},
)
def infiltration_rate():
    return _ext_constant_infiltration_rate()


_ext_constant_infiltration_rate = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INFILTRATION_RATE*",
    {},
    _root,
    {},
    "_ext_constant_infiltration_rate",
)


@component.add(
    name="SECURITY COEFFICIENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_security_coefficient"},
)
def security_coefficient():
    return _ext_constant_security_coefficient()


_ext_constant_security_coefficient = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "SECURITY_COEFFICIENT*",
    {},
    _root,
    {},
    "_ext_constant_security_coefficient",
)


@component.add(
    name="EVAPORATION FROM STORAGE",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_evaporation_from_storage"},
)
def evaporation_from_storage():
    return _ext_constant_evaporation_from_storage()


_ext_constant_evaporation_from_storage = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "EVAPORATION_FROM_STORAGE*",
    {},
    _root,
    {},
    "_ext_constant_evaporation_from_storage",
)


@component.add(
    name="REMAINING STORAGE WATER",
    units="Hm3",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"total_reservoir_capacity": 1, "security_coefficient": 1},
)
def remaining_storage_water():
    return total_reservoir_capacity() * security_coefficient()


@component.add(
    name="INITIAL STORAGE",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_storage"},
)
def initial_storage():
    return _ext_constant_initial_storage()


_ext_constant_initial_storage = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_STORAGE*",
    {},
    _root,
    {},
    "_ext_constant_initial_storage",
)


@component.add(
    name="ECOLOGICAL FLOW",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_ecological_flow"},
)
def ecological_flow():
    """
    2592000 segundo s en un mes y 1hm3 se corresponde con 1000000m3
    """
    return _ext_constant_ecological_flow()


_ext_constant_ecological_flow = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "ECOLOGICAL_FLOW*",
    {},
    _root,
    {},
    "_ext_constant_ecological_flow",
)


@component.add(
    name="INITIAL GROUNDWATER",
    units="Hm3",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_initial_groundwater"},
)
def initial_groundwater():
    return _ext_constant_initial_groundwater()


_ext_constant_initial_groundwater = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "INITIAL_GROUNDWATER*",
    {},
    _root,
    {},
    "_ext_constant_initial_groundwater",
)


@component.add(
    name="M3 TO HM3",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_m3_to_hm3"},
)
def m3_to_hm3():
    return _ext_constant_m3_to_hm3()


_ext_constant_m3_to_hm3 = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "M3_TO_HM3*",
    {},
    _root,
    {},
    "_ext_constant_m3_to_hm3",
)


@component.add(
    name="MM PER KM2 TO M3",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_mm_per_km2_to_m3"},
)
def mm_per_km2_to_m3():
    return _ext_constant_mm_per_km2_to_m3()


_ext_constant_mm_per_km2_to_m3 = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "MM_PER_KM2_TO_M3*",
    {},
    _root,
    {},
    "_ext_constant_mm_per_km2_to_m3",
)


@component.add(
    name="PERCOLATION COEFFICIENT",
    units="Dmnl",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_percolation_coefficient"},
)
def percolation_coefficient():
    return _ext_constant_percolation_coefficient()


_ext_constant_percolation_coefficient = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "PERCOLATION_COEFFICIENT*",
    {},
    _root,
    {},
    "_ext_constant_percolation_coefficient",
)


@component.add(
    name="TO AQUIFER",
    units="Hm3",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"infiltration": 1},
)
def to_aquifer():
    return 0.9 * infiltration()


@component.add(
    name="ANNUAL HOT INDEX",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"monthly_hot_index": 1},
)
def annual_hot_index():
    return sum(monthly_hot_index().rename({"MONTHS": "MONTHS!"}), dim=["MONTHS!"])


@component.add(
    name="COEFICIENT A",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"annual_hot_index": 3},
)
def coeficient_a():
    return (
        6.75e-07 * annual_hot_index() ** 3
        - 7.71e-05 * annual_hot_index() ** 2
        + 0.0179 * annual_hot_index()
        + 0.49239
    )


@component.add(
    name="DEFICIT",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"pet": 1, "aet": 1},
)
def deficit():
    return pet() - aet()


@component.add(
    name="MONTH DAYS",
    units="Days",
    subscripts=["MONTHS"],
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_month_days"},
)
def month_days():
    return _ext_constant_month_days()


_ext_constant_month_days = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "MONTH_DAYS*",
    {"MONTHS": _subscript_dict["MONTHS"]},
    _root,
    {"MONTHS": _subscript_dict["MONTHS"]},
    "_ext_constant_month_days",
)


@component.add(
    name="MONTHLY HOT INDEX",
    units="Dmnl",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature": 2},
)
def monthly_hot_index():
    return if_then_else(
        temperature() > 0,
        lambda: (temperature() / 5) ** 1.514,
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="PET",
    units="mm",
    subscripts=["MONTHS"],
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"temperature": 2, "annual_hot_index": 2, "coeficient_a": 1},
)
def pet():
    return if_then_else(
        np.logical_and(temperature() > 0, annual_hot_index() > 0),
        lambda: 16
        * np.power((10 * temperature()) / annual_hot_index(), coeficient_a()),
        lambda: xr.DataArray(0, {"MONTHS": _subscript_dict["MONTHS"]}, ["MONTHS"]),
    )


@component.add(
    name="SNOWMELTING FACTOR",
    units="mm/ºc/dia",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_snowmelting_factor"},
)
def snowmelting_factor():
    return _ext_constant_snowmelting_factor()


_ext_constant_snowmelting_factor = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "SNOWMELTING_FACTOR*",
    {},
    _root,
    {},
    "_ext_constant_snowmelting_factor",
)


@component.add(
    name="SOIL WATER CAPACITY",
    units="mm",
    comp_type="Constant",
    comp_subtype="External",
    depends_on={"__external__": "_ext_constant_soil_water_capacity"},
)
def soil_water_capacity():
    return _ext_constant_soil_water_capacity()


_ext_constant_soil_water_capacity = ExtConstant(
    r"Historical.xlsx",
    "Water system",
    "SOIL_WATER_CAPACITY*",
    {},
    _root,
    {},
    "_ext_constant_soil_water_capacity",
)
