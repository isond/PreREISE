import inspect
import os

import numpy as np
from scipy.io import loadmat

import prereise
from prereise.gather.demanddata.transportation_electrification import (
    data_helper,
    smart_charging,
    smart_charging_HDV,
)
from prereise.gather.demanddata.transportation_electrification.data_helper import (
    generate_daily_weighting,
)
from prereise.gather.demanddata.transportation_electrification.immediate import (
    immediate_charging,
)


def test_immediate_charging_region1():
    immediate_charging(
        census_region=1,
        model_year=2017,
        veh_range=100,
        kwhmi=0.242,
        power=6.6,
        location_strategy=2,
        veh_type="LDV",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "data",
            "nhts_census_updated_dwell",
        ),
    )


def test_smart_charging():
    run_smart_charging()


def run_smart_charging():
    data_dir = os.path.join(
        os.path.dirname(inspect.getsourcefile(prereise)),
        "gather",
        "demanddata",
        "transportation_electrification",
        "data",
        "CAISO_sample_load_2019.mat",
    )
    load_demand = loadmat(data_dir)["load_demand"].flatten()

    daily_values = generate_daily_weighting(2017)

    result = smart_charging.smart_charging(
        census_region=1,
        model_year=2017,
        veh_range=100,
        kwhmi=0.242,
        power=6.6,
        location_strategy=2,
        veh_type="LDV",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "tests",
            "test_census_data.csv",
        ),
        daily_values=daily_values,
        load_demand=load_demand,
        trip_strategy=1,
    )

    correct_cumsum = np.array(
        [
            0.0,
            9796092.83844097,
            19198735.09458018,
            27636677.75177433,
            36032644.7281563,
            44112809.4024421,
            52256940.31259822,
            61077768.57472202,
        ]
    )

    np.testing.assert_allclose(result.cumsum()[::1095], correct_cumsum)


def test_smart_charging_HDV():
    run_smart_charging_HDV()


def run_smart_charging_HDV():
    data_dir = os.path.join(
        os.path.dirname(inspect.getsourcefile(prereise)),
        "gather",
        "demanddata",
        "transportation_electrification",
        "data",
        "CAISO_sample_load_2019.mat",
    )
    load_demand = loadmat(data_dir)["load_demand"].flatten()

    bev_vmt = data_helper.load_urbanized_scaling_factor(
        model_year=2050,
        veh_type="HDV",
        veh_range=200,
        urbanized_area="Antioch",
        state="CA",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "data",
            "regional_scaling_factors",
            "Regional_scaling_factors_UA_",
        ),
    )
    result = smart_charging_HDV.smart_charging(
        model_year=2050,
        veh_range=200,
        power=80,
        location_strategy=1,
        veh_type="HDV",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "data",
            "fdata_v10st.data",
        ),
        initial_load=load_demand,
        bev_vmt=bev_vmt,
        trip_strategy=1,
    )

    correct_cumsum = np.array(
        [
            1.22854177233283,
            4729.417063,
            9456.028814,
            14087.49171,
            18817.56654,
            23521.75604,
            28175.75066,
            32904.52077,
        ]
    )

    np.testing.assert_allclose(result.cumsum()[::1095], correct_cumsum)


def test_smart_charging_MDV():
    run_smart_charging_MDV()


def run_smart_charging_MDV():
    data_dir = os.path.join(
        os.path.dirname(inspect.getsourcefile(prereise)),
        "gather",
        "demanddata",
        "transportation_electrification",
        "data",
        "CAISO_sample_load_2019.mat",
    )
    load_demand = loadmat(data_dir)["load_demand"].flatten()

    bev_vmt = data_helper.load_urbanized_scaling_factor(
        model_year=2050,
        veh_type="MDV",
        veh_range=200,
        urbanized_area="Antioch",
        state="CA",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "data",
            "regional_scaling_factors",
            "Regional_scaling_factors_UA_",
        ),
    )
    result = smart_charging_HDV.smart_charging(
        model_year=2050,
        veh_range=200,
        power=80,
        location_strategy=1,
        veh_type="MDV",
        filepath=os.path.join(
            os.path.dirname(inspect.getsourcefile(prereise)),
            "gather",
            "demanddata",
            "transportation_electrification",
            "data",
            "fdata_v10st.data",
        ),
        initial_load=load_demand,
        bev_vmt=bev_vmt,
        trip_strategy=1,
    )

    correct_cumsum = np.array(
        [
            0.291225084060603,
            4160.770734,
            8320.031013,
            12391.66618,
            16553.00332,
            20705.85663,
            24784.48369,
            28946.34624,
        ]
    )

    np.testing.assert_allclose(result.cumsum()[::1095], correct_cumsum)


if __name__ == "__main__":
    # run_smart_charging()

    run_smart_charging_HDV()
