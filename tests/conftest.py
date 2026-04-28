from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = REPO_ROOT / "workflow" / "scripts"

for path in (REPO_ROOT, WORKFLOW_SCRIPTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture
def steel_tech_costs() -> pd.Series:
    """Synthetic tech-cost data that mirrors the steel electricity chain."""
    rows = [
        ("Alkaline electrolyzer large size", "electricity-input", 1.38),
        ("hydrogen direct iron reduction furnace", "hydrogen-input", 2.1),
        ("hydrogen direct iron reduction furnace", "electricity-input", 1.03),
        ("electric arc furnace", "electricity-input", 0.6395),
    ]

    index = pd.MultiIndex.from_tuples(
        [(technology, parameter) for technology, parameter, _ in rows],
        names=["technology", "parameter"],
    )
    values = [value for _, _, value in rows]
    return pd.Series(values, index=index, dtype=float)


@pytest.fixture
def steel_minimal_case() -> dict:
    """Hand-checkable supply-stage numbers for a 1 t/h steel test case."""
    demand_mt_per_year = 0.00876
    hourly_demand_tph = demand_mt_per_year * 1e6 / 8760

    renewable_electricity_mwh_per_t = 2.1 * 1.38 + 1.03
    total_electricity_mwh_per_t = renewable_electricity_mwh_per_t + 0.6395

    return {
        "product": "steel",
        "demand_mt_per_year": demand_mt_per_year,
        "hourly_demand_tph": hourly_demand_tph,
        "annual_demand_t": hourly_demand_tph * 8760,
        "expected_steel_renewable_electricity_mwh_per_t": renewable_electricity_mwh_per_t,
        "expected_steel_total_electricity_mwh_per_t": total_electricity_mwh_per_t,
        "expected_annual_renewable_electricity_mwh": hourly_demand_tph
        * 8760
        * renewable_electricity_mwh_per_t,
        "expected_annual_total_electricity_mwh": hourly_demand_tph
        * 8760
        * total_electricity_mwh_per_t,
    }


@pytest.fixture
def round_number_tech_costs() -> pd.Series:
    """Minimal renewable tech costs with easy round numbers for tests.

    Values are chosen so conversion in add_renewable_generators remains simple:
    overnight_cost [EUR/MW] = investment [EUR/kW] * 1000.
    """
    rows = [
        ("solar-utility", "investment", 0.10),
        ("solar-utility", "lifetime", 20.0),
        ("solar-utility", "FOM", 10.0),
        ("onwind", "investment", 0.20),
        ("onwind", "lifetime", 25.0),
        ("onwind", "FOM", 5.0),
    ]

    index = pd.MultiIndex.from_tuples(
        [(technology, parameter) for technology, parameter, _ in rows],
        names=["technology", "parameter"],
    )
    values = [value for _, _, value in rows]
    return pd.Series(values, index=index, dtype=float)


@pytest.fixture
def full_chain_tech_costs() -> pd.Series:
    """Synthetic full-chain tech database for electrolysis->DRI->EAF tests.

    Includes all parameters used by build_x_supply_chain helpers and
    add_renewable_generators renewable cost lookup.
    """
    rows = [
        # Conversion chain
        ("Alkaline electrolyzer large size", "investment", 0.10),
        ("Alkaline electrolyzer large size", "lifetime", 20.0),
        ("Alkaline electrolyzer large size", "FOM", 2.0),
        ("Alkaline electrolyzer large size", "electricity-input", 1.38),
        ("hydrogen direct iron reduction furnace", "investment", 100.0),
        ("hydrogen direct iron reduction furnace", "lifetime", 25.0),
        ("hydrogen direct iron reduction furnace", "FOM", 5.0),
        ("hydrogen direct iron reduction furnace", "ore-input", 1.59),
        ("hydrogen direct iron reduction furnace", "hydrogen-input", 2.1),
        ("hydrogen direct iron reduction furnace", "electricity-input", 1.03),
        ("electric arc furnace", "investment", 50.0),
        ("electric arc furnace", "lifetime", 25.0),
        ("electric arc furnace", "FOM", 5.0),
        ("electric arc furnace", "hbi-input", 1.0),
        ("electric arc furnace", "electricity-input", 0.6395),
        # Storage and battery
        ("hydrogen storage underground", "investment", 0.01),
        ("hydrogen storage underground", "lifetime", 30.0),
        ("hydrogen storage underground", "FOM", 0.0),
        ("battery inverter", "investment", 0.05),
        ("battery inverter", "lifetime", 15.0),
        ("battery inverter", "FOM", 1.0),
        ("battery inverter", "efficiency", 0.96),
        ("battery storage", "investment", 0.05),
        ("battery storage", "lifetime", 20.0),
        # Renewable technologies used by add_renewable_generators
        ("solar-utility", "investment", 0.10),
        ("solar-utility", "lifetime", 20.0),
        ("solar-utility", "FOM", 10.0),
        ("onwind", "investment", 0.20),
        ("onwind", "lifetime", 25.0),
        ("onwind", "FOM", 5.0),
    ]

    index = pd.MultiIndex.from_tuples(
        [(technology, parameter) for technology, parameter, _ in rows],
        names=["technology", "parameter"],
    )
    values = [value for _, _, value in rows]
    return pd.Series(values, index=index, dtype=float)


@pytest.fixture
def renewable_region_timeseries_fixture() -> xr.Dataset:
    """Toy regional renewable profile for optimization tests.

    Contains two valid generator candidates:
    1. Solar candidate with constant CF = 0.5
    2. Wind candidate with CF = 1.0 for first half and 0.0 for second half

    Other bus-technology combinations are NaN and should be ignored.
    """
    n_hours = 24
    buses = ["TST_SOL_01", "TST_WND_01"]
    technologies = ["solar", "onwind"]
    hours = np.arange(n_hours)

    capacity_factor = np.full((len(buses), len(technologies), n_hours), np.nan)
    p_nom_max = np.full((len(buses), len(technologies)), np.nan)
    avg_cf = np.full((len(buses), len(technologies)), np.nan)

    # Candidate 1: constant CF 0.5
    capacity_factor[0, 0, :] = 0.5
    p_nom_max[0, 0] = 1e6
    avg_cf[0, 0] = 0.5

    # Candidate 2: half 1.0, half 0.0
    capacity_factor[1, 1, : n_hours // 2] = 1.0
    capacity_factor[1, 1, n_hours // 2 :] = 0.0
    p_nom_max[1, 1] = 1e6
    avg_cf[1, 1] = 0.5

    return xr.Dataset(
        data_vars={
            "capacity_factor": (("bus", "technology", "hour"), capacity_factor),
            "p_nom_max": (("bus", "technology"), p_nom_max),
            "avg_cf": (("bus", "technology"), avg_cf),
        },
        coords={
            "bus": buses,
            "technology": technologies,
            "hour": hours,
        },
    )
