"""Build PyPSA supply chain skeleton for commodity X using technology database.

Generic conversion pathway structure (currently configured for steel):
    Electricity → Electrolyzer → H2 → DRI → HBI → EAF → Commodity Output

Module provides functions to construct a PyPSA energy system network representing
a decarbonized production supply chain. The network includes:
  - Energy carriers (electricity, hydrogen, commodities)
  - Conversion technologies (electrolyzer, DRI, EAF)
  - Storage systems (H2 storage, batteries)
  - External resource supplies

Usage:
  - Snakemake integration: Automatically invoked with config and costs files
  - Standalone: Direct invocation for testing with sample config

Inputs:
  - tech_costs_path (str): Path to PyPSA technology database CSV
  - config (dict): Configuration dict with keys like 'cost_year', '*_p_min_pu'

Outputs:
  - PyPSA Network object ready for optimization
  - Exported to NetCDF format for storage and further analysis

Reusable pattern for any commodity with similar conversion chains.
Modify TECH_ASSUMPTIONS, bus definitions, and links to adapt to different commodities.
"""

import logging
from typing import Any
import pandas as pd
import numpy as np
import pypsa

import tech_database as td

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

snakemake: Any = globals().get("snakemake")


# Technology parameters with no database source (assumed values)
TECH_ASSUMPTIONS = {
    "h2_standing_loss": 0.001,  # 0.1% per hour for underground cavern (leakage)
    "batt_standing_loss": 0.0001,  # 0.01% per hour for battery (self-discharge)
}


def _add_carriers(network: pypsa.Network) -> None:
    """Add carrier components to network.

    PyPSA requires explicit Carrier components before buses/generators can reference them.
    """
    carriers = {
        "renewable_electricity": "Islanded renewable electricity",
        "hydrogen": "Hydrogen gas",
        "battery_elec": "Battery (electrical energy)",
        "iron_ore": "Iron ore (mass)",
        "hbi": "Hot Briquetted Iron (mass)",
        "steel": "Steel (mass)",
        "electrolysis": "Electrolysis process",
        "direct_reduction_furnace": "Direct reduction furnace",
        "electric_arc_furnace": "Electric arc furnace",
        "grid_electricity": "Grid electricity import",
    }
    for carrier_name, description in carriers.items():
        network.add("Carrier", carrier_name)


def _add_buses(network: pypsa.Network) -> None:
    """Add energy carrier buses."""
    buses = {
        "renewable_electricity": {"carrier": "renewable_electricity", "unit": "MW"},
        "grid_electricity": {"carrier": "grid_electricity", "unit": "MW"},
        "hydrogen": {"carrier": "hydrogen", "unit": "MW"},
        "battery": {"carrier": "battery_elec", "unit": "MWh"},
        "iron_ore": {"carrier": "iron_ore", "unit": "t/h"},
        "hbi": {"carrier": "hbi", "unit": "t/h"},
        "steel": {"carrier": "steel", "unit": "t/h"},
    }
    for name, attrs in buses.items():
        network.add("Bus", name, **attrs)


def _add_grid_electricity_supply(network: pypsa.Network, config: dict) -> None:
    """Add a grid import generator for the EAF when requested.

    The default topology uses grid-connected EAF power. When config sets
    `eaf_electricity_source` to anything other than `grid`, this helper is a no-op
    and the EAF remains connected to the local electricity bus.
    """

    if config.get("eaf_electricity_source", "grid") != "grid":
        return

    network.add(
        "Generator",
        "grid_electricity_import",
        bus="grid_electricity",
        carrier="grid_electricity",
        p_nom=1e10,
        marginal_cost=config.get("grid_electricity_price", 75.0),
    )


def _add_conversion_chain(
    network: pypsa.Network, tech_costs: pd.Series, config: dict
) -> None:
    """Add energy conversion pathway: Electricity → H2 → HBI → Steel.
    Note: Costs are added but discount_rate is NOT set here.
    It is applied regionally in prepare_regional_network.
    """

    # Electrolyzer: Electricity → H2
    elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")

    elec_inv_cost = td.get_tech_param(elec_params, "investment", 544.7764) * 1000
    network.add(
        "Link",
        "electrolyzer",
        bus0="renewable_electricity",
        bus1="hydrogen",
        carrier="electrolysis",
        efficiency=1.0 / td.get_tech_param(elec_params, "electricity-input", 1.38),
        overnight_cost=elec_inv_cost,  # EUR/kW → EUR/MW
        lifetime=td.get_tech_param(elec_params, "lifetime", 40.0),
        fom_cost=elec_inv_cost * (td.get_tech_param(elec_params, "FOM", 2.8) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("elec_p_min_pu", 0.10),
    )

    # DRI Furnace: Iron ore + Hydrogen + Electricity → HBI
    dri_params = td.get_tech(tech_costs, "hydrogen direct iron reduction furnace")

    dri_inv_cost = td.get_tech_param(dri_params, "investment", 5378698.8822)
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
        lifetime=td.get_tech_param(dri_params, "lifetime", 40.0),
        fom_cost=dri_inv_cost * (td.get_tech_param(dri_params, "FOM", 11.3) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("dri_p_min_pu", 0.15),
    )

    # EAF: HBI + Electricity → Steel
    eaf_params = td.get_tech(tech_costs, "electric arc furnace")
    eaf_bus2 = (
        "grid_electricity"
        if config.get("eaf_electricity_source", "grid") == "grid"
        else "renewable_electricity"
    )

    eaf_inv_cost = td.get_tech_param(eaf_params, "investment", 2312992.7323)
    network.add(
        "Link",
        "eaf",
        bus0="hbi",
        bus1="steel",
        bus2=eaf_bus2,
        carrier="electric_arc_furnace",
        efficiency=1.0 / td.get_tech_param(eaf_params, "hbi-input", 1.0),
        efficiency2=-td.get_tech_param(eaf_params, "electricity-input", 0.6395),
        overnight_cost=eaf_inv_cost,
        lifetime=td.get_tech_param(eaf_params, "lifetime", 40.0),
        fom_cost=eaf_inv_cost * (td.get_tech_param(eaf_params, "FOM", 30.0) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        p_min_pu=config.get("eaf_p_min_pu", 0.20),
    )


def _add_storage(network: pypsa.Network, tech_costs: pd.Series, config: dict) -> None:
    """Add H2 and battery storage systems.

    Note: Costs are added but discount_rate is NOT set here.
    It is applied regionally in prepare_regional_network.
    """

    # H2 Storage (underground cavern)
    h2_params = td.get_tech(tech_costs, "hydrogen storage underground")

    h2_inv_cost = td.get_tech_param(h2_params, "investment", 1.6045) * 1000
    network.add(
        "Store",
        "h2_storage",
        bus="hydrogen",
        e_nom_extendable=True,
        overnight_cost=h2_inv_cost,  # EUR/kWh → EUR/MWh
        lifetime=td.get_tech_param(h2_params, "lifetime", 100.0),
        fom_cost=h2_inv_cost * (td.get_tech_param(h2_params, "FOM", 0.0) / 100),
        standing_loss=TECH_ASSUMPTIONS["h2_standing_loss"],
        e_initial=config.get("h2_storage_e_initial", 0.5),  # Start at 50% capacity
        e_cyclic=True,  # End state must equal start state
    )

    # Battery Storage: Power (inverter for charger/discharger) + Energy (store)
    batt_inv_params = td.get_tech(tech_costs, "battery inverter")
    batt_store_params = td.get_tech(tech_costs, "battery storage")

    batt_inv_cost = td.get_tech_param(batt_inv_params, "investment", 80.223) * 1000
    network.add(
        "Link",
        "batt_charge",
        bus0="renewable_electricity",
        bus1="battery",
        efficiency=np.sqrt(td.get_tech_param(batt_inv_params, "efficiency", 0.96)),
        overnight_cost=batt_inv_cost,  # EUR/kW → EUR/MW
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 10.0),
        fom_cost=batt_inv_cost * (td.get_tech_param(batt_inv_params, "FOM", 0.9) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
    )

    network.add(
        "Link",
        "batt_discharge",
        bus0="battery",
        bus1="renewable_electricity",
        efficiency=np.sqrt(td.get_tech_param(batt_inv_params, "efficiency", 0.96)),
        overnight_cost=batt_inv_cost,
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 10.0),
        fom_cost=batt_inv_cost * (td.get_tech_param(batt_inv_params, "FOM", 0.9) / 100),
        p_nom_extendable=True,
        p_nom_max=np.inf,
    )

    batt_store_cost = (
        td.get_tech_param(batt_store_params, "investment", 100.2787) * 1000
    )
    network.add(
        "Store",
        "battery",
        bus="battery",
        e_nom_extendable=True,
        overnight_cost=batt_store_cost,  # EUR/kWh → EUR/MWh
        lifetime=td.get_tech_param(batt_store_params, "lifetime", 30.0),
        fom_cost=batt_store_cost * 0.0,
        standing_loss=TECH_ASSUMPTIONS["batt_standing_loss"],
        e_initial=config.get("battery_e_initial", 0.5),  # Start at 50% capacity
        e_cyclic=True,  # End state must equal start state
    )

    network.add(
        "Store",
        "hbi_storage",
        bus="hbi",
        e_nom_extendable=True,
        overnight_cost=0.0,  # Just a pile - no cost
        lifetime=1.0,
        fom_cost=0.0,  # No maintenance cost
        discount_rate=0.0,  # No cost, discount rate doesn't matter but required by PyPSA
        standing_loss=0.0,  # HBI storage doesn't lose energy
    )


def _add_resources(network: pypsa.Network, config: dict) -> None:
    """Add external resource supplies (iron ore)."""
    network.add(
        "Generator",
        "iron_ore",
        bus="iron_ore",
        p_nom=1e10,
        marginal_cost=0,
    )


def build_network(config: dict, tech_costs_path: str, year: int) -> pypsa.Network:
    """Build PyPSA steel supply chain skeleton (region-agnostic).

    The skeleton contains:
    - Carriers and buses (region-independent)
    - Conversion chain with costs but WITHOUT discount_rate
    - Regional discount_rate is applied later in prepare_regional_network

    This design allows the same skeleton to be used across regions with different discount rates.
    """

    network = pypsa.Network()
    network.set_snapshots(pd.date_range(f"{year}-01-01", periods=8760, freq="h"))
    # NOTE: discount_rate is NOT set here (region-agnostic)
    tech_costs = td.load_tech_costs(tech_costs_path)

    # Add network components (carriers MUST be added before buses that reference them)
    _add_carriers(network)
    _add_buses(network)
    _add_grid_electricity_supply(network, config)
    _add_conversion_chain(network, tech_costs, config)
    _add_storage(network, tech_costs, config)
    _add_resources(network, config)

    logger.info(
        f"Built network: {len(network.buses)} buses, {len(network.links)} links, "
        f"{len(network.stores)} stores, {len(network.generators)} generators"
    )

    return network


if __name__ == "__main__":
    if snakemake is None:
        raise RuntimeError(
            "This script must be run via Snakemake with cost_year wildcard"
        )

    config = snakemake.config  # noqa: F821
    tech_costs_path = snakemake.input.costs  # noqa: F821
    output_path = snakemake.output[0]  # noqa: F821

    cost_year = 0
    if cost_year is None:
        raise ValueError("snakemake.wildcards.cost_year is required")

    year = getattr(snakemake.wildcards, "cost_year", 0)  # noqa: F821
    network = build_network(config, tech_costs_path, year)
    network.export_to_netcdf(output_path)
    logger.info(f"Network exported to {output_path}")
