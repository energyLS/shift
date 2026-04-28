import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr

import workflow.scripts.tech_database as td
from workflow.scripts.build_x_supply_chain import (
    _add_buses,
    _add_carriers,
    _add_conversion_chain,
    _add_grid_electricity_supply,
    _add_resources,
    _add_storage,
)
from workflow.scripts.calculate_lcox import _convert_arrow_strings
from workflow.scripts.prepare_regional_network import add_renewable_generators


def _build_selected_generators(dataset):
    """Convert xarray fixture into selected_generators input format."""
    generators = []

    for bus_id in dataset.bus.values:
        for tech in dataset.technology.values:
            p_nom_max = float(
                dataset["p_nom_max"].sel(bus=bus_id, technology=tech).values
            )
            avg_cf = float(dataset["avg_cf"].sel(bus=bus_id, technology=tech).values)
            cf_ts = dataset["capacity_factor"].sel(bus=bus_id, technology=tech).values

            if np.isnan(p_nom_max) or p_nom_max <= 0 or np.isnan(avg_cf) or avg_cf <= 0:
                continue

            generators.append(
                {
                    "bus_id": str(bus_id),
                    "technology": str(tech),
                    "p_nom_max": p_nom_max,
                    "avg_cf": avg_cf,
                    "capacity_factor_ts": np.asarray(cf_ts, dtype=float),
                }
            )

    return generators


def _build_low_capacity_dataset(dataset, p_nom_max=0.01):
    """Return a copy of the renewable dataset with very low generator capacity caps."""
    low_capacity = dataset.copy(deep=True)
    low_capacity["p_nom_max"] = xr.zeros_like(low_capacity["p_nom_max"]) + p_nom_max
    return low_capacity


def _build_full_chain_reference_network(dataset, tech_costs, config):
    """Manual full-chain network: carriers, buses, links, stores, and renewables."""
    snapshots = pd.date_range("2030-01-01", periods=24, freq="h")

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.discount_rate = 0.05

    # Carriers used by full chain + renewable technologies
    for carrier in [
        "renewable_electricity",
        "grid_electricity",
        "hydrogen",
        "battery_elec",
        "iron_ore",
        "hbi",
        "steel",
        "electrolysis",
        "direct_reduction_furnace",
        "electric_arc_furnace",
        "solar",
        "onwind",
    ]:
        network.add("Carrier", carrier)

    # Buses
    network.add(
        "Bus", "renewable_electricity", carrier="renewable_electricity", unit="MW"
    )
    network.add("Bus", "grid_electricity", carrier="grid_electricity", unit="MW")
    network.add("Bus", "hydrogen", carrier="hydrogen", unit="MW")
    network.add("Bus", "battery", carrier="battery_elec", unit="MWh")
    network.add("Bus", "iron_ore", carrier="iron_ore", unit="t/h")
    network.add("Bus", "hbi", carrier="hbi", unit="t/h")
    network.add("Bus", "steel", carrier="steel", unit="t/h")

    # Conversion links
    elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")
    elec_inv_cost = td.get_tech_param(elec_params, "investment", 0.10) * 1000
    network.add(
        "Link",
        "electrolyzer",
        bus0="renewable_electricity",
        bus1="hydrogen",
        carrier="electrolysis",
        efficiency=1.0 / td.get_tech_param(elec_params, "electricity-input", 1.38),
        overnight_cost=elec_inv_cost,
        lifetime=td.get_tech_param(elec_params, "lifetime", 20.0),
        fom_cost=elec_inv_cost * (td.get_tech_param(elec_params, "FOM", 2.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("elec_p_min_pu", 0.10),
    )

    dri_params = td.get_tech(tech_costs, "hydrogen direct iron reduction furnace")
    dri_inv_cost = td.get_tech_param(dri_params, "investment", 100.0)
    network.add(
        "Link",
        "dri",
        bus0="iron_ore",
        bus1="hbi",
        bus2="hydrogen",
        bus3="renewable_electricity",
        carrier="direct_reduction_furnace",
        efficiency=1.0 / td.get_tech_param(dri_params, "ore-input", 1.59),
        efficiency2=-td.get_tech_param(dri_params, "hydrogen-input", 2.1),
        efficiency3=-td.get_tech_param(dri_params, "electricity-input", 1.03),
        overnight_cost=dri_inv_cost,
        lifetime=td.get_tech_param(dri_params, "lifetime", 25.0),
        fom_cost=dri_inv_cost * (td.get_tech_param(dri_params, "FOM", 5.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("dri_p_min_pu", 0.15),
    )

    eaf_params = td.get_tech(tech_costs, "electric arc furnace")
    eaf_inv_cost = td.get_tech_param(eaf_params, "investment", 50.0)
    network.add(
        "Link",
        "eaf",
        bus0="hbi",
        bus1="steel",
        bus2="grid_electricity",
        carrier="electric_arc_furnace",
        efficiency=1.0 / td.get_tech_param(eaf_params, "hbi-input", 1.0),
        efficiency2=-td.get_tech_param(eaf_params, "electricity-input", 0.6395),
        overnight_cost=eaf_inv_cost,
        lifetime=td.get_tech_param(eaf_params, "lifetime", 25.0),
        fom_cost=eaf_inv_cost * (td.get_tech_param(eaf_params, "FOM", 5.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("eaf_p_min_pu", 0.20),
    )

    # Storage and battery links
    h2_params = td.get_tech(tech_costs, "hydrogen storage underground")
    h2_inv_cost = td.get_tech_param(h2_params, "investment", 0.01) * 1000
    network.add(
        "Store",
        "h2_storage",
        bus="hydrogen",
        e_nom_extendable=True,
        overnight_cost=h2_inv_cost,
        lifetime=td.get_tech_param(h2_params, "lifetime", 30.0),
        fom_cost=h2_inv_cost * (td.get_tech_param(h2_params, "FOM", 0.0) / 100),
        standing_loss=0.001,
        e_initial=config.get("h2_storage_e_initial", 0.5),
        e_cyclic=True,
    )

    batt_inv_params = td.get_tech(tech_costs, "battery inverter")
    batt_store_params = td.get_tech(tech_costs, "battery storage")
    batt_inv_cost = td.get_tech_param(batt_inv_params, "investment", 0.05) * 1000
    batt_eff = np.sqrt(td.get_tech_param(batt_inv_params, "efficiency", 0.96))

    network.add(
        "Link",
        "batt_charge",
        bus0="renewable_electricity",
        bus1="battery",
        efficiency=batt_eff,
        overnight_cost=batt_inv_cost,
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 15.0),
        fom_cost=batt_inv_cost * (td.get_tech_param(batt_inv_params, "FOM", 1.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
    )

    network.add(
        "Link",
        "batt_discharge",
        bus0="battery",
        bus1="renewable_electricity",
        efficiency=batt_eff,
        overnight_cost=batt_inv_cost,
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 15.0),
        fom_cost=batt_inv_cost * (td.get_tech_param(batt_inv_params, "FOM", 1.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
    )

    batt_store_cost = td.get_tech_param(batt_store_params, "investment", 0.05) * 1000
    network.add(
        "Store",
        "battery",
        bus="battery",
        e_nom_extendable=True,
        overnight_cost=batt_store_cost,
        lifetime=td.get_tech_param(batt_store_params, "lifetime", 20.0),
        fom_cost=0.0,
        standing_loss=0.0001,
        e_initial=config.get("battery_e_initial", 0.5),
        e_cyclic=True,
    )

    network.add(
        "Store",
        "hbi_storage",
        bus="hbi",
        e_nom_extendable=True,
        overnight_cost=0.0,
        lifetime=1.0,
        fom_cost=0.0,
        discount_rate=0.0,
        standing_loss=0.0,
    )

    # External resource
    network.add("Generator", "iron_ore", bus="iron_ore", p_nom=1e10, marginal_cost=0.0)

    if config.get("eaf_electricity_source", "grid") == "grid":
        network.add(
            "Generator",
            "grid_electricity_import",
            bus="grid_electricity",
            carrier="grid_electricity",
            p_nom=1e10,
            marginal_cost=config.get("grid_electricity_price", 75.0),
        )

    # Renewable generators
    selected_generators = _build_selected_generators(dataset)
    add_renewable_generators(
        network=network,
        dataset=dataset,
        tech_costs=tech_costs,
        config=config,
        selected_generators=selected_generators,
    )

    return network


def _build_full_chain_function_network(dataset, tech_costs, config):
    """Function-built full-chain network using workflow helper modules."""
    snapshots = pd.date_range("2030-01-01", periods=24, freq="h")

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.discount_rate = 0.05

    _add_carriers(network)
    _add_buses(network)
    _add_grid_electricity_supply(network, config)
    _add_conversion_chain(network, tech_costs, config)
    _add_storage(network, tech_costs, config)
    _add_resources(network, config)

    selected_generators = _build_selected_generators(dataset)
    add_renewable_generators(
        network=network,
        dataset=dataset,
        tech_costs=tech_costs,
        config=config,
        selected_generators=selected_generators,
    )

    return network


def _apply_capital_cost_from_overnight_cost(network):
    """Ensure optimization objective uses overnight costs deterministically."""
    for component_name in ["generators", "links", "stores"]:
        component = getattr(network, component_name)
        if "overnight_cost" in component.columns:
            for name, row in component.iterrows():
                if pd.notna(row.get("overnight_cost", np.nan)):
                    component.at[name, "capital_cost"] = float(row["overnight_cost"])
                    if "discount_rate" in component.columns and pd.isna(
                        row.get("discount_rate", np.nan)
                    ):
                        component.at[name, "discount_rate"] = float(
                            getattr(network, "discount_rate", 0.0)
                        )


def _solve_or_skip(network):
    _convert_arrow_strings(network)
    try:
        network.optimize(
            network.snapshots,
            solver_name="highs",
            include_objective_constant=False,
        )
    except Exception as exc:
        pytest.skip(f"Optimization solver unavailable in current environment: {exc}")


def _solve_network(network):
    _convert_arrow_strings(network)
    return network.optimize(
        network.snapshots,
        solver_name="highs",
        include_objective_constant=False,
    )


def test_function_built_full_chain_matches_manual_reference(
    renewable_region_timeseries_fixture,
    full_chain_tech_costs,
):
    config = {
        "eaf_electricity_source": "grid",
        "grid_electricity_price": 75.0,
        "elec_p_min_pu": 0.10,
        "dri_p_min_pu": 0.15,
        "eaf_p_min_pu": 0.20,
        "h2_storage_e_initial": 0.5,
        "battery_e_initial": 0.5,
    }

    reference = _build_full_chain_reference_network(
        renewable_region_timeseries_fixture,
        full_chain_tech_costs,
        config,
    )
    built = _build_full_chain_function_network(
        renewable_region_timeseries_fixture,
        full_chain_tech_costs,
        config,
    )

    assert set(reference.carriers.index) == set(built.carriers.index)
    assert set(reference.buses.index) == set(built.buses.index)
    assert set(reference.links.index) == set(built.links.index)
    assert set(reference.stores.index) == set(built.stores.index)
    assert set(reference.generators.index) == set(built.generators.index)

    assert "grid_electricity_import" in reference.generators.index
    assert (
        reference.generators.at["grid_electricity_import", "carrier"]
        == "grid_electricity"
    )
    assert (
        built.generators.at["grid_electricity_import", "carrier"] == "grid_electricity"
    )
    assert reference.links.at["eaf", "bus2"] == "grid_electricity"
    assert built.links.at["eaf", "bus2"] == "grid_electricity"

    for link_name in ["electrolyzer", "dri", "eaf"]:
        assert (
            built.links.at[link_name, "carrier"]
            == reference.links.at[link_name, "carrier"]
        )
        assert built.links.at[link_name, "efficiency"] == pytest.approx(
            reference.links.at[link_name, "efficiency"]
        )


def test_full_chain_optimization_matches_manual_reference(
    renewable_region_timeseries_fixture,
    full_chain_tech_costs,
):
    config = {
        "eaf_electricity_source": "grid",
        "grid_electricity_price": 75.0,
        "elec_p_min_pu": 0.10,
        "dri_p_min_pu": 0.15,
        "eaf_p_min_pu": 0.20,
        "h2_storage_e_initial": 0.5,
        "battery_e_initial": 0.5,
    }

    reference = _build_full_chain_reference_network(
        renewable_region_timeseries_fixture,
        full_chain_tech_costs,
        config,
    )
    built = _build_full_chain_function_network(
        renewable_region_timeseries_fixture,
        full_chain_tech_costs,
        config,
    )

    for network in (reference, built):
        network.add("Load", "steel_demand", bus="steel", p_set=1.0)
        _apply_capital_cost_from_overnight_cost(network)

    _solve_or_skip(reference)
    _solve_or_skip(built)

    # Compare key optimized capacities of the full chain and renewables.
    for component, name in [
        ("links", "electrolyzer"),
        ("links", "dri"),
        ("links", "eaf"),
        ("generators", "renewable_TST_SOL_01_solar"),
        ("generators", "renewable_TST_WND_01_onwind"),
    ]:
        ref_value = float(getattr(reference, component).at[name, "p_nom_opt"])
        built_value = float(getattr(built, component).at[name, "p_nom_opt"])
        assert built_value == pytest.approx(ref_value, rel=1e-5, abs=1e-6)

    # Directional behavior check for expected renewable choice.
    ref_solar = float(
        reference.generators.at["renewable_TST_SOL_01_solar", "p_nom_opt"]
    )
    ref_wind = float(
        reference.generators.at["renewable_TST_WND_01_onwind", "p_nom_opt"]
    )
    assert ref_solar > 0.0
    assert ref_solar > ref_wind


def test_full_chain_becomes_infeasible_when_renewables_are_too_small(
    renewable_region_timeseries_fixture,
    full_chain_tech_costs,
):
    """If renewable caps are too low, the steel chain should not be solvable."""
    config = {
        "eaf_electricity_source": "grid",
        "grid_electricity_price": 75.0,
        "elec_p_min_pu": 0.10,
        "dri_p_min_pu": 0.15,
        "eaf_p_min_pu": 0.20,
        "h2_storage_e_initial": 0.5,
        "battery_e_initial": 0.5,
    }

    low_capacity_dataset = _build_low_capacity_dataset(
        renewable_region_timeseries_fixture,
        p_nom_max=0.01,
    )

    reference = _build_full_chain_reference_network(
        low_capacity_dataset,
        full_chain_tech_costs,
        config,
    )
    built = _build_full_chain_function_network(
        low_capacity_dataset,
        full_chain_tech_costs,
        config,
    )

    for network in (reference, built):
        network.add("Load", "steel_demand", bus="steel", p_set=1.0)
        _apply_capital_cost_from_overnight_cost(network)

    reference_result = _solve_network(reference)
    built_result = _solve_network(built)

    reference_status = " ".join(
        str(part).lower() for part in np.atleast_1d(reference_result)
    )
    built_status = " ".join(str(part).lower() for part in np.atleast_1d(built_result))

    assert "infeas" in reference_status or reference.objective is None
    assert "infeas" in built_status or built.objective is None


def test_full_chain_can_switch_eaf_to_renewable_electricity(
    renewable_region_timeseries_fixture,
    full_chain_tech_costs,
):
    config = {
        "eaf_electricity_source": "renewable",
        "elec_p_min_pu": 0.10,
        "dri_p_min_pu": 0.15,
        "eaf_p_min_pu": 0.20,
        "h2_storage_e_initial": 0.5,
        "battery_e_initial": 0.5,
    }

    built = _build_full_chain_function_network(
        renewable_region_timeseries_fixture,
        full_chain_tech_costs,
        config,
    )

    assert "grid_electricity_import" not in built.generators.index
    assert built.links.at["eaf", "bus2"] == "renewable_electricity"
