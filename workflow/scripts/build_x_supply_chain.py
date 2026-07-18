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
Modify the techno-economic parameters, bus definitions, and links to adapt to different commodities.
"""

from typing import Any
from pathlib import Path
import pandas as pd
import numpy as np
import pypsa

import tech_database as td
from _helpers import setup_logging

from trade_chain_utils import (
    get_ordered_stages,
    get_trade_chain,
    split_stage_inputs,
    get_stage_groups,
    _components_for_process_label,
)

snakemake: Any = globals().get("snakemake")
logger = setup_logging(
    __name__, snakemake=snakemake, log_filename="build_x_supply_chain.log"
)


def _techno_economic_parameters(config: dict) -> dict:
    return config.get("techno-economic parameters", {})


def _additional_parameters(config: dict) -> dict:
    return _techno_economic_parameters(config).get("additional_parameters", {})


def _additional_parameter(config: dict, name: str, default: float) -> float:
    parameters = _additional_parameters(config)
    if name not in parameters:
        logger.warning(
            f"Missing techno-economic parameter '{name}'; using default value {default}."
        )
    return float(parameters.get(name, default))


def _part_load(config: dict, technology: str, default: float) -> float:
    """Read a part-load minimum from config.part_load."""
    return float(config.get("part_load", {}).get(technology, default))


def _add_carriers(network: pypsa.Network) -> None:
    """Add carrier components to network.

    PyPSA requires explicit Carrier components before buses/generators can reference them.
    """
    carriers = {
        "renewable_electricity": "Islanded renewable electricity",
        # Technology-specific renewable carriers (used by Generators)
        "renewable_pv": "Photovoltaic (utility)",
        "renewable_wind_onshore": "Onshore wind",
        "renewable_wind_offshore": "Offshore wind",
        "hydrogen": "Hydrogen gas",
        "battery_elec": "Battery (electrical energy)",
        "iron_ore": "Iron ore (mass)",
        "hbi": "Hot Briquetted Iron (mass)",
        "steel": "Steel (mass)",
        "grid_electricity": "Grid electricity import",
    }
    for carrier_name, description in carriers.items():
        network.add("Carrier", carrier_name, description=description)


def _add_buses(network: pypsa.Network, stages: list | None = None) -> None:
    """Add energy carrier buses.

    The `grid_electricity` bus is created only when one of the configured
    stages explicitly requires grid electricity as an energy input. This
    keeps the skeleton free of an unused grid bus when stages are fully
    renewable.
    """
    # determine whether any stage requires grid_electricity
    uses_grid = False
    if stages is not None:
        for stage in stages:
            _, energy_inputs = split_stage_inputs(stage)
            if "grid_electricity" in energy_inputs:
                uses_grid = True
                break

    buses = {
        "renewable_electricity": {"carrier": "renewable_electricity", "unit": "MW"},
        "hydrogen": {"carrier": "hydrogen", "unit": "MW"},
        "battery": {"carrier": "battery_elec", "unit": "MWh"},
        "iron_ore": {"carrier": "iron_ore", "unit": "t/h"},
        "hbi": {"carrier": "hbi", "unit": "t/h"},
        "steel": {"carrier": "steel", "unit": "t/h"},
    }

    # add grid bus only if needed
    if uses_grid:
        buses["grid_electricity"] = {"carrier": "grid_electricity", "unit": "MW"}

    for name, attrs in buses.items():
        if name not in network.buses.index:
            network.add("Bus", name, **attrs)


def _add_grid_electricity_supply(
    network: pypsa.Network, config: dict, stages: list | None = None
) -> None:
    """Add a grid import generator for the EAF when requested.

    The default topology uses grid-connected EAF power. When config sets
    `eaf_electricity_source` to anything other than `grid`, this helper is a no-op
    and the EAF remains connected to the local electricity bus.
    """

    # If specific stages provided, inspect those; otherwise inspect full chain
    if stages is None:
        chain = get_trade_chain(config)
        stages = get_ordered_stages(chain)
    uses_grid = any(
        "grid_electricity" in split_stage_inputs(stage)[1] for stage in stages
    )

    if not uses_grid:
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
    network: pypsa.Network,
    tech_costs: pd.Series,
    config: dict,
    stages: list | None = None,
) -> None:
    """Add the configured stage conversion pathway.
    Note: Costs are added but discount_rate is NOT set here.
    It is applied regionally in prepare_regional_network.
    """

    # Allow building from an explicit list of stages (stage-group) or full chain
    if stages is None:
        chain = get_trade_chain(config)
        stages = get_ordered_stages(chain)
    if not stages:
        raise ValueError("No stages configured in trade_chains")

    for stage in stages:
        process_label = str(stage.get("process_label", "")).strip()
        materials, energy_inputs = split_stage_inputs(stage)

        # Validate stage IO against canonical mapping (sanity check only)
        try:
            # Use strict behavior if config requests it
            strict_validation = bool(config.get("strict_trade_chain_validation", False))
            from trade_chain_utils import validate_stage_io

            validate_stage_io(stage, raise_on_mismatch=strict_validation)
        except Exception as exc:
            # If strict_validation True, validate_stage_io will raise; propagate
            if config.get("strict_trade_chain_validation", False):
                raise
            logger.warning(f"Trade-chain validation issue: {exc}")

        # Resolve concrete components from mapping
        comp = _components_for_process_label(process_label)
        if comp is None:
            logger.warning(
                f"No component mapping for process_label '{process_label}'; skipping stage"
            )
            continue

        # Normalize candidate buses
        # Prefer canonical buses from the mapping; fall back to stage-declared inputs
        def pick_bus(preferred: str, fallback_list: list[str], default: str) -> str:
            # preferred may be like 'renewable_electricity' or 'hydrogen'
            if preferred in comp.get("buses", ()):  # type: ignore[arg-type]
                return preferred
            for f in fallback_list:
                if f:
                    return f
            return default

        # ELECTROLYZER
        if "electrolyzer" in comp.get("links", ()):  # type: ignore[arg-type]
            elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")
            elec_inv_cost = (
                td.get_tech_param(elec_params, "investment", 544.7764) * 1000
            )
            # choose energy bus: prefer 'renewable_electricity' unless grid explicitly listed
            energy_bus = (
                "grid_electricity"
                if "grid_electricity" in energy_inputs
                else "renewable_electricity"
            )
            network.add(
                "Link",
                process_label,
                bus0=energy_bus,
                bus1="hydrogen",
                carrier="electrolysis",
                efficiency=1.0
                / td.get_tech_param(elec_params, "electricity-input", 1.38),
                overnight_cost=elec_inv_cost,
                lifetime=td.get_tech_param(elec_params, "lifetime", 40.0),
                fom_cost=elec_inv_cost
                * (td.get_tech_param(elec_params, "FOM", 2.8) / 100),
                p_nom_extendable=True,
                p_nom_max=np.inf,
                p_min_pu=_part_load(config, "electrolysis", 0.10),
            )

        # DRI (direct reduction) — use canonical buses: iron_ore -> hbi, hydrogen input, electricity
        if "dri" in comp.get("links", ()):  # type: ignore[arg-type]
            dri_params = td.get_tech(
                tech_costs, "hydrogen direct iron reduction furnace"
            )
            dri_inv_cost = td.get_tech_param(dri_params, "investment", 5378698.8822)
            # canonical buses
            bus0 = "iron_ore"
            bus1 = "hbi"
            bus2 = "hydrogen"
            # bus3: electricity; prefer grid if explicitly listed in stage, else renewable
            bus3 = (
                "grid_electricity"
                if "grid_electricity" in energy_inputs
                else "renewable_electricity"
            )
            network.add(
                "Link",
                process_label,
                bus0=bus0,
                bus1=bus1,
                bus2=bus2,
                bus3=bus3,
                carrier="direct_reduction_furnace",
                efficiency=1.0 / td.get_tech_param(dri_params, "ore-input", 1.59),
                efficiency2=-td.get_tech_param(dri_params, "hydrogen-input", 2.1),
                efficiency3=-td.get_tech_param(dri_params, "electricity-input", 1.03),
                overnight_cost=dri_inv_cost,
                lifetime=td.get_tech_param(dri_params, "lifetime", 40.0),
                fom_cost=dri_inv_cost
                * (td.get_tech_param(dri_params, "FOM", 11.3) / 100),
                p_nom_extendable=True,
                p_nom_max=np.inf,
                p_min_pu=_part_load(config, "direct reduction furnace", 0.15),
            )

        # EAF (electric arc furnace) — canonical buses: hbi -> steel, electricity from grid by default
        if "eaf" in comp.get("links", ()):  # type: ignore[arg-type]
            eaf_params = td.get_tech(tech_costs, "electric arc furnace")
            eaf_inv_cost = td.get_tech_param(eaf_params, "investment", 2312992.7323)
            # canonical buses
            bus0 = "hbi"
            bus1 = "steel"
            # choose energy bus: prefer grid by default; if stage explicitly lists renewable, use renewable
            bus2 = (
                "grid_electricity"
                if "grid_electricity" in energy_inputs
                else "renewable_electricity"
            )
            network.add(
                "Link",
                process_label,
                bus0=bus0,
                bus1=bus1,
                bus2=bus2,
                carrier="electric_arc_furnace",
                efficiency=1.0 / td.get_tech_param(eaf_params, "hbi-input", 1.0),
                efficiency2=-td.get_tech_param(eaf_params, "electricity-input", 0.6395),
                overnight_cost=eaf_inv_cost,
                lifetime=td.get_tech_param(eaf_params, "lifetime", 40.0),
                fom_cost=eaf_inv_cost
                * (td.get_tech_param(eaf_params, "FOM", 30.0) / 100),
                p_nom_extendable=True,
                p_nom_max=np.inf,
                p_min_pu=_part_load(config, "electric arc furnace", 0.20),
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
        standing_loss=_additional_parameter(config, "h2_standing_loss", 0.0),
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
        standing_loss=_additional_parameter(config, "batt_standing_loss", 0.0),
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
        e_cyclic=True,
    )

    # Steel Storage: flexible intermediate inventory between EAF and demand
    network.add(
        "Store",
        "steel_storage",
        bus="steel",
        e_nom_extendable=True,
        overnight_cost=0.0,  # Just a pile - no cost
        lifetime=1.0,
        fom_cost=0.0,  # No maintenance cost
        discount_rate=0.0,  # No cost, discount rate doesn't matter but required by PyPSA
        standing_loss=0.0,  # Steel storage doesn't lose energy
        e_cyclic=True,  # End state must equal start state
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
    # determine full-stage ordering for bus creation decisions
    chain = get_trade_chain(config)
    stages = get_ordered_stages(chain)
    _add_buses(network, stages=stages)
    _add_grid_electricity_supply(network, config)
    _add_conversion_chain(network, tech_costs, config)
    _add_storage(network, tech_costs, config)

    logger.info(
        f"Built network: {len(network.buses)} buses, {len(network.links)} links, "
        f"{len(network.stores)} stores, {len(network.generators)} generators"
    )

    return network


def _set_meta(network: pypsa.Network, group: dict | None) -> None:
    """Attach metadata about stage group to the network for downstream tools."""
    meta = {}
    if group is not None:
        meta["stage_group_label"] = group.get("label")
        meta["stages"] = [int(s.get("order", -1)) for s in group.get("stages", [])]
        # Derive whether this group uses renewables
        uses_renewables = any(
            "renewable_electricity" in split_stage_inputs(s)[1]
            for s in group.get("stages", [])
        )
        meta["uses_renewables"] = bool(uses_renewables)
    network.meta = meta


if __name__ == "__main__":
    if snakemake is None:
        raise RuntimeError(
            "This script must be run via Snakemake with cost_year wildcard"
        )

    config = snakemake.config  # noqa: F821
    tech_costs_path = snakemake.input.costs  # noqa: F821
    output_path = snakemake.output[0]  # noqa: F821

    cost_year = getattr(snakemake.wildcards, "cost_year", None)
    if cost_year is None:
        raise ValueError("snakemake.wildcards.cost_year is required")

    year = int(cost_year)

    # Build and export the full skeleton (backwards compatible)
    full_network = build_network(config, tech_costs_path, year)
    _set_meta(full_network, None)
    # Ensure target directory exists and write to new generic_model path
    out_dir = str(Path(output_path).resolve().parent)
    generic_dir = Path(out_dir) / ".." / "generic_production_model"
    generic_dir = generic_dir.resolve()
    full_out = generic_dir / f"generic_model_{year}.nc"
    full_network.export_to_netcdf(str(full_out))
    logger.info(f"Full generic model exported to {full_out}")

    # Additionally export one skeleton per detected stage-group
    chain = get_trade_chain(config)
    groups = get_stage_groups(chain)
    out_dir = str(Path(output_path).resolve().parent)
    generic_dir = Path(out_dir) / ".." / "generic_production_model"
    generic_dir = generic_dir.resolve()
    for group in groups:
        label = group.get("label") or "group"
        # Build a fresh network containing only components for this stage-group
        group_network = pypsa.Network()
        group_network.set_snapshots(
            pd.date_range(f"{year}-01-01", periods=8760, freq="h")
        )
        tech_costs = td.load_tech_costs(tech_costs_path)
        _add_carriers(group_network)
        _add_buses(group_network, stages=group.get("stages"))
        _add_grid_electricity_supply(group_network, config, stages=group.get("stages"))
        _add_conversion_chain(
            group_network, tech_costs, config, stages=group.get("stages")
        )
        _add_storage(group_network, tech_costs, config)
        _set_meta(group_network, group)
        out_path = generic_dir / f"generic_model_{year}_{label}.nc"
        group_network.export_to_netcdf(str(out_path))
        logger.info(f"Exported stage-group generic model: {out_path}")
