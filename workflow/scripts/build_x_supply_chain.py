"""
Build PyPSA supply chain skeleton for commodity X using technology database.

Generic conversion pathway structure (currently configured for steel):
  Electricity → Electrolyzer → H2 → DRI → HBI → EAF → Commodity Output

Module provides functions to construct a PyPSA energy system network representing a decarbonized
production supply chain. The network includes:
  - Energy carriers (electricity, hydrogen, commodities)
  - Conversion technologies (electrolyzer, DRI, EAF)
  - Storage systems (H2 storage, batteries)
  - External resource supplies

Usage:
  - Snakemake integration: Automatically invoked with config and costs files
  - Standalone: Direct invocation for testing with sample config

Inputs:
  - tech_costs_path (str): Path to PyPSA technology database CSV with columns [technology, parameter]
  - config (dict): Configuration dict with keys like 'cost_year', '*_p_min_pu'

Outputs:
  - PyPSA Network object ready for optimization
  - Exported to NetCDF format for storage and further analysis

Reusable pattern for any commodity with similar conversion chains. Modify TECH_ASSUMPTIONS,
bus definitions, and conversion links to adapt to different commodities.
"""

import logging
import pandas as pd
import numpy as np
import pypsa
import snakemake

import tech_database as td

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Technology parameters with no database source (assumed values)
TECH_ASSUMPTIONS = {
    "h2_standing_loss": 0.001,  # 0.1% per hour for underground cavern (leakage)
    "batt_standing_loss": 0.0001,  # 0.01% per hour for battery (self-discharge)
}


def _add_buses(network: pypsa.Network) -> None:
    """Add energy carrier buses."""
    buses = {
        "electricity": {"carrier": "AC", "unit": "MW"},
        "hydrogen": {"carrier": "H2", "unit": "MW"},
        "iron_ore": {"carrier": "Iron ore", "unit": "t/h"},
        "hbi": {"carrier": "HBI", "unit": "t/h"},
        "steel": {"carrier": "Steel", "unit": "t/h"},
    }
    for name, attrs in buses.items():
        network.add("Bus", name, **attrs)


def _add_conversion_chain(network: pypsa.Network, tech_costs: pd.Series, config: dict) -> None:
    """Add energy conversion pathway: Electricity → H2 → HBI → Steel."""
    
    # Electrolyzer: Electricity → H2
    elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")
    
    network.add(
        "Link", "electrolyzer",
        bus0="electricity", bus1="hydrogen",
        efficiency=1.0 / td.get_tech_param(elec_params, "electricity-input", 1.38),
        overnight_cost=td.get_tech_param(elec_params, "investment", 544.7764) * 1000,  # EUR/kW → EUR/MW
        lifetime=td.get_tech_param(elec_params, "lifetime", 40.0),
        fom_cost=td.get_tech_param(elec_params, "investment", 544.7764) * 1000 * (td.get_tech_param(elec_params, "FOM", 2.8) / 100),  # % → decimal
        p_nom_extendable=True,
        p_min_pu=config.get("elec_p_min_pu", 0.10),
    )
    
    # DRI Furnace: Iron ore + Hydrogen + Electricity → HBI
    dri_params = td.get_tech(tech_costs, "hydrogen direct iron reduction furnace")
    
    network.add(
        "Link", "dri",
        bus0="iron_ore", bus1="hbi", bus2="hydrogen", bus3="electricity",
        efficiency=1.0 / td.get_tech_param(dri_params, "ore-input", 1.59),  # t_ore/t_hbi → efficiency (t_hbi/t_ore)
        efficiency2=-td.get_tech_param(dri_params, "hydrogen-input", 2.1),  # negative = input (MWh_H2/t_hbi)
        efficiency3=-td.get_tech_param(dri_params, "electricity-input", 1.03),  # negative = input (MWh_el/t_hbi auxiliary)
        overnight_cost=td.get_tech_param(dri_params, "investment", 5378698.8822),  # EUR/t_HBI/h from database
        lifetime=td.get_tech_param(dri_params, "lifetime", 40.0),
        fom_cost=td.get_tech_param(dri_params, "investment", 5378698.8822) * (td.get_tech_param(dri_params, "FOM", 11.3) / 100),  # % → decimal
        p_nom_extendable=True,
        p_min_pu=config.get("dri_p_min_pu", 0.15),
    )
    
    # EAF: HBI + Electricity → Steel
    eaf_params = td.get_tech(tech_costs, "electric arc furnace")
    
    network.add(
        "Link", "eaf",
        bus0="hbi", bus1="steel", bus2="electricity",
        efficiency=1.0 / td.get_tech_param(eaf_params, "hbi-input", 1.0),  # t_hbi/t_steel → efficiency (t_steel/t_hbi)
        efficiency2=-td.get_tech_param(eaf_params, "electricity-input", 0.6395),  # negative = input (MWh_el/t_steel)
        overnight_cost=td.get_tech_param(eaf_params, "investment", 2312992.7323),  # EUR/t_steel/h from database
        lifetime=td.get_tech_param(eaf_params, "lifetime", 40.0),
        fom_cost=td.get_tech_param(eaf_params, "investment", 2312992.7323) * (td.get_tech_param(eaf_params, "FOM", 30.0) / 100),  # % → decimal
        p_nom_extendable=True,
        p_min_pu=config.get("eaf_p_min_pu", 0.20),
    )


def _add_storage(network: pypsa.Network, tech_costs: pd.Series, config: dict) -> None:
    """Add H2 and battery storage systems."""
    
    # H2 Storage (underground cavern)
    h2_params = td.get_tech(tech_costs, "hydrogen storage underground")
    
    network.add(
        "Store", "h2_storage",
        bus="hydrogen",
        e_nom_extendable=True,
        overnight_cost=td.get_tech_param(h2_params, "investment", 1.6045) * 1000,  # EUR/kWh → EUR/MWh
        lifetime=td.get_tech_param(h2_params, "lifetime", 100.0),
        fom_cost=td.get_tech_param(h2_params, "investment", 1.6045) * 1000 * (td.get_tech_param(h2_params, "FOM", 0.0) / 100),  # % → decimal
        standing_loss=TECH_ASSUMPTIONS["h2_standing_loss"],
    )
    
    # Battery Storage: Power (inverter for charger/discharger) + Energy (store)
    batt_inv_params = td.get_tech(tech_costs, "battery inverter")
    batt_store_params = td.get_tech(tech_costs, "battery storage")
    
    network.add(
        "Link", "batt_charge",
        bus0="electricity", bus1="battery",
        efficiency=np.sqrt(td.get_tech_param(batt_inv_params, "efficiency", 0.96)),  # Round-trip → per-direction efficiency
        overnight_cost=td.get_tech_param(batt_inv_params, "investment", 80.223) * 1000,  # EUR/kW → EUR/MW
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 10.0),
        fom_cost=td.get_tech_param(batt_inv_params, "investment", 80.223) * 1000 * (td.get_tech_param(batt_inv_params, "FOM", 0.9) / 100),  # % → decimal
        p_nom_extendable=True,
    )
    
    network.add(
        "Link", "batt_discharge",
        bus0="battery", bus1="electricity",
        efficiency=np.sqrt(td.get_tech_param(batt_inv_params, "efficiency", 0.96)),  # Round-trip → per-direction efficiency
        overnight_cost=td.get_tech_param(batt_inv_params, "investment", 80.223) * 1000,  # EUR/kW → EUR/MW
        lifetime=td.get_tech_param(batt_inv_params, "lifetime", 10.0),
        fom_cost=td.get_tech_param(batt_inv_params, "investment", 80.223) * 1000 * (td.get_tech_param(batt_inv_params, "FOM", 0.9) / 100),  # % → decimal
        p_nom_extendable=True,
    )
    
    network.add(
        "Store", "battery",
        bus="battery",
        e_nom_extendable=True,
        overnight_cost=td.get_tech_param(batt_store_params, "investment", 100.2787) * 1000,  # EUR/kWh → EUR/MWh
        lifetime=td.get_tech_param(batt_store_params, "lifetime", 30.0),
        fom_cost=td.get_tech_param(batt_store_params, "investment", 100.2787) * 1000 * 0.0,  # Battery storage has no explicit FOM in database
        standing_loss=TECH_ASSUMPTIONS["batt_standing_loss"],
    )


def _add_resources(network: pypsa.Network, config: dict) -> None:
    """Add external resource supplies (iron ore)."""
    network.add(
        "Generator", "iron_ore",
        bus="iron_ore",
        p_nom=1e10,
        marginal_cost=0, # Assuming zero marginal cost in supply chain model, will be adjusted in trade model
    )


def build_network(config: dict, tech_costs_path: str) -> pypsa.Network:
    """Build PyPSA steel supply chain skeleton."""
    
    # Setup
    year = config.get("cost_year", 2030)
    network = pypsa.Network()
    network.set_snapshots(pd.date_range(f"{year}-01-01", periods=8760, freq="h"))
    tech_costs = td.load_tech_costs(tech_costs_path)
    
    # Add network components
    _add_buses(network)
    _add_conversion_chain(network, tech_costs, config)
    _add_storage(network, tech_costs, config)
    _add_resources(network, config)
    
    logger.info(f"Built network: {len(network.buses)} buses, {len(network.links)} links, "
                f"{len(network.stores)} stores, {len(network.generators)} generators")
    
    return network


if __name__ == "__main__":
    
    # Handle Snakemake or direct invocation
    if "snakemake" in globals():
        config = snakemake.config
        tech_costs_path = snakemake.input.costs
        output_path = snakemake.output[0]
    else:
        # Fallback for testing
        config = {"cost_year": 2030}
        tech_costs_path = "../resources/technology_data/costs_2030.csv"
        output_path = "test_steel_network.nc"
    
    network = build_network(config, tech_costs_path)
    network.export_to_netcdf(output_path)
    logger.info(f"Network exported to {output_path}")
