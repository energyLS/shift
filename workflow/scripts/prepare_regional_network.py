"""
Prepare regional network: load consolidated renewables and configure supply chain.

This is the simplified Step 1 workflow that:
1. Loads the consolidated renewable profiles directly (region, technology, class, time)
2. Creates PyPSA generators for each technology and class
3. Applies local demand reservation if configured
4. Adds supply chain (electrolyzer, DRI, optional EAF)
5. Applies product-specific cutoff

Usage (Snakemake rule):
    rule prepare_regional_network:
        input:
            skeleton = "resources/networks/skeleton.nc",
            renewables = "data/new_renewables_consolidated.nc",
            tech_costs = "resources/tech_database.csv",
        params:
            region = "{region}",
            product = "{product}",
        output:
            network = "resources/networks/base_{cost_year}_{region}_{product}.nc",
        script:
            "scripts/prepare_regional_network.py"
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import pypsa

import tech_database as td

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

snakemake: Any = globals().get("snakemake")

if snakemake is not None and getattr(snakemake, "log", None):
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_region_renewables_consolidated(
    consolidated_path: str,
    region: str,
) -> Tuple[Dict[str, np.ndarray], xr.DataArray, Dict]:
    """
    Load renewable data for a region from consolidated NetCDF.

    Parameters
    ----------
    consolidated_path : str
        Path to data/new_renewables_consolidated.nc
    region : str
        Region name (e.g., 'Europe', 'North_America')

    Returns
    -------
    technologies_dict : dict
        Mapping {tech_name: capacity_array} where each is (n_class,)

    cf_ts : xr.DataArray
        Capacity factor time series with dims (time, technology, class)

    metadata : dict
        Summary info (n_classes, n_time, technologies, etc.)
    """
    ds = xr.open_dataset(consolidated_path)

    if region not in ds.region.values:
        available = ", ".join(ds.region.values)
        raise ValueError(f"Region '{region}' not found. Available: {available}")

    logger.info(f"Loading consolidated renewables for {region}")

    # Select region (dims: technology, class)
    region_cap = ds["capacity"].sel(region=region)  # (tech, class)
    region_cf = ds["capacity_factor"].sel(region=region)  # (tech, class, time)

    # Extract technology names and data
    techs = list(region_cap.technology.values)
    technologies_dict = {}

    for tech in techs:
        cap = region_cap.sel(technology=tech).values  # (class,)
        technologies_dict[tech] = cap
        logger.info(f"  {tech}: {len(cap)} sites, {cap.sum():.0f} MW total")

    # Capacity factor time series (keep full structure for now)
    cf_ts = region_cf  # (tech, class, time)

    metadata = {
        "region": region,
        "n_classes": region_cap.sizes["class"],
        "n_time": region_cf.sizes["time"],
        "n_technologies": len(techs),
        "technologies": techs,
        "time_start": pd.Timestamp(ds["time"].values[0]),
        "time_end": pd.Timestamp(ds["time"].values[-1]),
        "total_capacity_mw": float(region_cap.sum().values),
    }

    logger.info(
        f"  Total capacity: {metadata['total_capacity_mw']:.0f} MW, "
        f"{metadata['n_time']} timesteps, {metadata['n_technologies']} technologies"
    )

    ds.close()
    return technologies_dict, cf_ts, metadata


def load_local_electricity_demand_mw(
    local_demand_path: str,
    region: str,
) -> float:
    """Load regional electricity demand and convert it to average MW."""

    try:
        local_df = pd.read_csv(local_demand_path)
        region_mask = local_df["region"].str.lower() == region.lower()
        if not region_mask.any():
            logger.warning(f"Region '{region}' not found in local demand data")
            return 0.0

        total_energy_mwh = float(local_df[region_mask]["demand"].values[0])
        el_share = float(local_df[region_mask]["el_share"].values[0]) / 100.0
        local_el_demand_mwh = total_energy_mwh * el_share
        return local_el_demand_mwh / 8760.0
    except Exception as exc:
        logger.warning(f"Could not load local demand for {region}: {exc}")
        return 0.0


def reserve_top_sites_by_highest_cf(
    technologies_dict: Dict[str, np.ndarray],
    cf_ts: xr.DataArray,
    reserve_capacity_mw: float,
    scenario: str = "reserved",
) -> Dict[str, np.ndarray]:
    """
    Reserve top sites (highest average capacity factor) for local demand.

    Parameters
    ----------
    technologies_dict : dict
        {tech_name: capacity_array}
    cf_ts : xr.DataArray
        Capacity factor time series with dims (technology, class, time)
    reserve_capacity_mw : float
        Target MW capacity to reserve
    scenario : str
        "reserved" (default): apply reservation logic
        "unconstrained": skip reservation, return None

    Returns
    -------
    dict or None
        Dict of same structure as technologies_dict, with NaN for non-reserved sites.
        If scenario="unconstrained", returns None (no reservation).
    """
    if scenario == "unconstrained":
        logger.info(
            "Scenario=unconstrained: skipping site reservation (all generators available)"
        )
        return None

    if reserve_capacity_mw <= 0:
        logger.info(
            "No reservation applied: reserve_capacity_mw <= 0 for reserved scenario"
        )
        return None

    # Flatten all sites with (tech, site, capacity, avg_cf) index
    reserved = {}
    total_reserved_mw = 0

    all_sites = []
    for tech, caps in technologies_dict.items():
        for site_idx, cap in enumerate(caps):
            # Calculate average capacity factor for this site
            if tech in cf_ts.coords.get("technology", []):
                cf_data = cf_ts.sel(technology=tech).isel({"class": site_idx})
                avg_cf = float(cf_data.mean().values)
            else:
                avg_cf = 0
            all_sites.append((tech, site_idx, cap, avg_cf))

    # Sort by average capacity factor (descending) — highest CF first
    all_sites.sort(key=lambda x: x[3], reverse=True)

    # Reserve until target capacity (allow partial reservation on the last site)
    reserved_set = set()
    reserved_amounts = {}
    for tech, site_idx, cap, avg_cf in all_sites:
        if total_reserved_mw >= reserve_capacity_mw:
            break
        remaining_mw = max(reserve_capacity_mw - total_reserved_mw, 0)
        reserve_mw = min(cap, remaining_mw)
        if reserve_mw <= 0:
            continue
        reserved_set.add((tech, site_idx))
        reserved_amounts[(tech, site_idx)] = reserve_mw
        total_reserved_mw += reserve_mw

    # Create reserved arrays (copy dict structure, mask non-reserved with NaN)
    for tech, caps in technologies_dict.items():
        reserved_array = np.full_like(caps, np.nan, dtype=np.float32)
        for site_idx, cap in enumerate(caps):
            if (tech, site_idx) in reserved_set:
                reserved_array[site_idx] = reserved_amounts[(tech, site_idx)]
        reserved[tech] = reserved_array

    logger.info(
        f"Reserved {len(reserved_set)} sites, {total_reserved_mw:.0f} MW for local demand"
    )
    return reserved


def add_renewable_generators(
    network: pypsa.Network,
    region: str,
    technologies_dict: Dict[str, np.ndarray],
    cf_ts: xr.DataArray,
    tech_costs: pd.Series,
    config: dict,
    reserved_techs: Dict[str, np.ndarray] = None,
) -> Dict:
    """
    Add renewable generators to PyPSA network.

    For each technology and class (site), creates an extendable generator on the
    electricity bus with capacity ceiling from technologies_dict and time series from cf_ts.

    If reserved_techs provided, marks reserved sites with local_priority=True tag.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network to add generators to
    region : str
        Region name (for bus and generator naming)
    technologies_dict : dict
        {tech_name: capacity_array} where capacity_array is (n_classes,)
    cf_ts : xr.DataArray
        Capacity factor time series with dims (technology, class, time)
    tech_costs : pd.Series
        Technology cost database
    config : dict
        Configuration dict
    reserved_techs : dict, optional
        {tech_name: reserved_capacity_array} (NaN for non-reserved)

    Returns
    -------
    audit_dict : dict
        Statistics: n_generators_added, total_capacity_mw, etc.
    """
    n_added = 0
    total_p_nom_max = 0
    n_reserved = 0

    # Ensure electricity bus exists
    elec_bus = "renewable_electricity"
    if elec_bus not in network.buses.index:
        network.add("Bus", elec_bus, carrier="AC", v_nom=1)

    # Get discount rate
    discount_rate = network.discount_rate if hasattr(network, "discount_rate") else 0.07

    # Map consolidated file tech names to database keys
    tech_db_map = {
        "windonshore": "onwind",
        "windoffshore": "offwind",
        "pvplant": "solar-utility",
    }

    for tech, capacities in technologies_dict.items():
        cf_data = cf_ts.sel(technology=tech).values  # (class, time)

        # Get technology cost parameters from database
        db_tech_name = tech_db_map.get(tech, tech)
        try:
            tech_params = td.get_tech(tech_costs, db_tech_name)
            # investment is EUR/kW, convert to EUR/MW by multiplying by 1000
            overnight_cost = td.get_tech_param(tech_params, "investment", 0) * 1000
            fom_pct = td.get_tech_param(tech_params, "FOM", 0)
            fom_cost = overnight_cost * (fom_pct / 100) if overnight_cost > 0 else 0
            lifetime = td.get_tech_param(tech_params, "lifetime", 20)
        except Exception as e:
            logger.warning(
                f"Could not load costs for {db_tech_name}: {e}, using defaults"
            )
            overnight_cost, fom_cost, lifetime = 0, 0, 20

        # Add generator for each site (class)
        for site_idx, p_nom_max in enumerate(capacities):
            if np.isnan(p_nom_max) or p_nom_max <= 0:
                continue

            gen_name = f"renewable_{region}_{tech}_{site_idx}"
            reserved_cap = 0.0
            is_reserved = False

            # Check if this site is reserved for local demand
            if reserved_techs is not None and tech in reserved_techs:
                reserved_cap = reserved_techs[tech][site_idx]
                if not np.isnan(reserved_cap) and reserved_cap > 0:
                    is_reserved = True
                    n_reserved += 1

            # If reserved, remove reserved capacity from export supply
            if is_reserved:
                if reserved_cap >= p_nom_max:
                    continue
                p_nom_max = p_nom_max - reserved_cap

            # Get time series for this site
            p_max_pu = cf_data[site_idx, :]  # (time,)

            # Add generator (extendable with ceiling)
            network.add(
                "Generator",
                gen_name,
                bus=elec_bus,
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0,  # Start with no capacity; optimization will decide
                p_nom_max=p_nom_max,  # Upper ceiling from dataset (MW)
                p_max_pu=p_max_pu,  # Hourly capacity factor (0-1)
                overnight_cost=overnight_cost,
                discount_rate=discount_rate,
                lifetime=lifetime,
                fom_cost=fom_cost,
                tags={
                    "technology": tech,
                    "region": region,
                    "local_priority": is_reserved,
                    "site_id": site_idx,
                },
            )

            n_added += 1
            total_p_nom_max += p_nom_max

        n_sites_added = sum(~np.isnan(capacities))
        logger.info(f"Added {n_sites_added} {tech} generators for {region}")

    logger.info(
        f"Total generators added: {n_added}, ceiling capacity: {total_p_nom_max:.0f} MW"
    )
    if n_reserved > 0:
        logger.info(f"  Reserved sites (local_priority): {n_reserved}")

    return {
        "n_generators_added": n_added,
        "total_capacity_mw": total_p_nom_max,
        "n_reserved": n_reserved,
    }


def back_propagate_electricity_need(
    tech_costs: pd.Series,
    product: str,
    config: dict,
) -> float:
    """
    Calculate renewable electricity requirement (MWh) per tonne of product.

    Paths:
    - steel (t): Electrolyzer(elec) + DRI(elec + H2) + optional EAF(elec)
    - hbi (t): Electrolyzer(elec) + DRI(elec + H2)
    - h2 (t): Electrolyzer(elec) only
    """
    # Check for config override
    override_key = f"electricity_per_tonne_{product}_mwh"
    if override_key in config:
        value = config[override_key]
        logger.info(f"Using config override: {product} requires {value} MWh/t")
        return value

    # Electrolyzer: Electricity → H2
    elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")
    elec_mwh_per_mwh_h2 = td.get_tech_param(elec_params, "electricity-input", 1.38)

    if product == "h2":
        return elec_mwh_per_mwh_h2

    # DRI Furnace: Iron ore + Hydrogen + Electricity → HBI
    dri_params = td.get_tech(tech_costs, "hydrogen direct iron reduction furnace")
    h2_per_t_hbi = td.get_tech_param(dri_params, "hydrogen-input", 2.1)
    dri_elec_per_t_hbi = td.get_tech_param(dri_params, "electricity-input", 1.03)
    h2_elec_per_t_hbi = h2_per_t_hbi * elec_mwh_per_mwh_h2

    if product == "hbi":
        return h2_elec_per_t_hbi + dri_elec_per_t_hbi

    if product == "steel":
        # Add EAF if configured, otherwise just HBI path
        eaf_source = config.get("eaf_electricity_source", "grid")
        if eaf_source == "renewable":
            eaf_params = td.get_tech(tech_costs, "electric arc furnace")
            eaf_elec_per_t_steel = td.get_tech_param(
                eaf_params, "electricity-input", 0.5
            )
            return h2_elec_per_t_hbi + dri_elec_per_t_hbi + eaf_elec_per_t_steel
        else:
            return h2_elec_per_t_hbi + dri_elec_per_t_hbi

    raise ValueError(f"Product '{product}' not recognized. Choose: h2, hbi, steel")


def apply_product_cutoff(network: pypsa.Network, product: str) -> None:
    """Remove supply chain components after the target product."""
    if product == "h2":
        # Keep only: electricity -> electrolyzer -> H2 storage
        # Remove: DRI, HBI, EAF, steel
        components_to_remove = [
            ("Link", "link_dri_furnace"),
            ("Link", "link_eaf"),
            ("Store", "store_hbi"),
            ("Store", "store_steel"),
        ]
    elif product == "hbi":
        # Keep: electricity -> electrolyzer -> H2 -> DRI -> HBI
        # Remove: EAF, steel
        components_to_remove = [
            ("Link", "link_eaf"),
            ("Store", "store_steel"),
        ]
    elif product == "steel":
        # Keep all: electricity -> electrolyzer -> H2 -> DRI -> HBI -> EAF -> steel
        components_to_remove = []
    else:
        raise ValueError(f"Product '{product}' not recognized")

    for comp_type, comp_name in components_to_remove:
        if comp_name in getattr(network, comp_type.lower() + "s", {}).index:
            logger.info(f"Removing {comp_type} {comp_name}")
            network.remove(comp_type, comp_name)


def prepare_network(
    skeleton_network_path: str,
    consolidated_renewables_path: str,
    tech_costs_path: str,
    local_demand_path: str,
    region: str,
    product: str,
    cost_year: int = 2030,
    config: dict = None,
    scenario: str = "reserved",
) -> Tuple[pypsa.Network, Dict]:
    """
    Prepare regional network with consolidated renewables.

    Parameters
    ----------
    skeleton_network_path : str
        Path to base network topology
    consolidated_renewables_path : str
        Path to consolidated renewables NetCDF
    tech_costs_path : str
        Path to technology cost database
    region : str
        Region name
    product : str
        Target product (h2, hbi, steel)
    cost_year : int
        Cost year for technology parameters
    config : dict
        Configuration dict
    scenario : str
        "reserved" (default): apply high-CF site reservation for domestic demand
        "unconstrained": no reservation; full renewable stack available (fallback scenario)

    Returns
    -------
    network : pypsa.Network
        Prepared PyPSA network
    audit_dict : dict
        Summary statistics
    """
    if config is None:
        config = {}

    logger.info("=" * 70)
    logger.info(f"Preparing network: region={region}, product={product}")
    logger.info("=" * 70)

    # Load skeleton
    logger.info("Loading skeleton network...")
    network = pypsa.Network(skeleton_network_path)
    network.name = f"base_{cost_year}_{region}_{product}"

    # Set region-specific discount rate
    interest_rates = config.get("interest_rate", {})

    # Get region-specific rate, or fall back to default
    if isinstance(interest_rates.get(region), dict):
        # Handle legacy component-level structure (flatten to use default)
        discount_rate = interest_rates[region].get(
            "default", interest_rates.get("default", 0.07)
        )
    else:
        discount_rate = interest_rates.get(region, interest_rates.get("default", 0.07))

    network.discount_rate = discount_rate
    logger.info(f"Region {region}: discount_rate = {discount_rate}")

    # Apply regional discount rate to all links and stores with costs
    for link_name in network.links.index:
        if network.links.loc[link_name, "overnight_cost"] > 0:
            network.links.loc[link_name, "discount_rate"] = discount_rate

    for store_name in network.stores.index:
        if network.stores.loc[store_name, "overnight_cost"] > 0:
            network.stores.loc[store_name, "discount_rate"] = discount_rate

    # Load tech costs
    logger.info("Loading technology costs...")
    tech_costs = td.load_tech_costs(tech_costs_path)

    # Load consolidated renewables for region
    logger.info("Loading consolidated renewables...")
    techs_dict, cf_ts, metadata = load_region_renewables_consolidated(
        consolidated_renewables_path, region
    )

    # Apply local demand reservation if configured
    # For scenario="reserved", reserve high-CF sites; for "unconstrained", skip reservation
    reserved_techs = None
    reserve_capacity_mw = config.get("reserve_local_demand_mw", 0)
    if reserve_capacity_mw <= 0 and scenario == "reserved":
        reserve_capacity_mw = load_local_electricity_demand_mw(
            local_demand_path, region
        )
        logger.info(
            f"Derived reservation target from local demand: {reserve_capacity_mw:.1f} MW"
        )
    logger.info(f"Scenario: {scenario} (scenario flag passed from Snakemake rule)")
    if reserve_capacity_mw > 0 or scenario == "reserved":
        logger.info(
            f"Applying local demand reservation for scenario={scenario}: "
            f"target {reserve_capacity_mw} MW"
        )
        reserved_techs = reserve_top_sites_by_highest_cf(
            techs_dict, cf_ts, reserve_capacity_mw, scenario=scenario
        )

    # Add renewable generators
    logger.info("Adding renewable generators...")
    gen_audit = add_renewable_generators(
        network, region, techs_dict, cf_ts, tech_costs, config, reserved_techs
    )

    # Apply product cutoff
    logger.info(f"Applying product cutoff for {product}...")
    apply_product_cutoff(network, product)

    # Build audit info
    audit_info = {
        "region": region,
        "product": product,
        "cost_year": cost_year,
        "scenario": scenario,
        "discount_rate": discount_rate,
        "renewables_metadata": metadata,
        "generators_audit": gen_audit,
        "network_stats": {
            "n_buses": len(network.buses),
            "n_generators": len(network.generators),
            "n_links": len(network.links),
            "n_stores": len(network.stores),
        },
    }

    logger.info("=" * 70)
    logger.info("Network prepared successfully:")
    logger.info(f"  - Buses: {len(network.buses)}")
    logger.info(f"  - Generators: {len(network.generators)}")
    logger.info(f"  - Capacity: {gen_audit['total_capacity_mw']:.0f} MW")
    logger.info("=" * 70)

    return network, audit_info


# ============================================================================
# SNAKEMAKE INTEGRATION
# ============================================================================

if __name__ == "__main__":
    # Check if running from Snakemake
    if snakemake is not None:
        # Snakemake inputs/outputs
        skeleton_path = snakemake.input.skeleton
        renewables_path = snakemake.input.renewables
        tech_costs_path = snakemake.input.tech_costs
        local_demand_path = snakemake.input.local_demand

        region = snakemake.params.region
        product = snakemake.params.product
        cost_year = (
            snakemake.wildcards.cost_year
            if hasattr(snakemake.wildcards, "cost_year")
            else 2030
        )
        scenario = (
            snakemake.wildcards.scenario
            if hasattr(snakemake.wildcards, "scenario")
            else "reserved"
        )

        output_path = snakemake.output[0]

        # Load config (if available)
        config_dict = snakemake.config if snakemake is not None else {}
    else:
        # Fallback for manual execution
        import sys

        if len(sys.argv) > 1:
            skeleton_path = sys.argv[1]
            renewables_path = sys.argv[2]
            tech_costs_path = sys.argv[3]
            region = sys.argv[4]
            product = sys.argv[5]
            output_path = sys.argv[6]
            cost_year = int(sys.argv[7]) if len(sys.argv) > 7 else 2030
            local_demand_path = None
            config_dict = {}
        else:
            raise ValueError("Provide paths and region/product as arguments")

    # Prepare network
    network, audit = prepare_network(
        skeleton_network_path=skeleton_path,
        consolidated_renewables_path=renewables_path,
        tech_costs_path=tech_costs_path,
        local_demand_path=local_demand_path,
        region=region,
        product=product,
        cost_year=cost_year,
        config=config_dict,
        scenario=scenario,
    )

    # Save network
    logger.info(f"Saving network to {output_path}")
    network.export_to_netcdf(output_path)

    logger.info("Network preparation complete")
