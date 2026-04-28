"""Prepare regional network: add renewables and configure supply chain per product."""

import logging
import pandas as pd
import numpy as np
import xarray as xr
import pypsa
import json
from typing import Dict, List, Tuple

import tech_database as td

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def region_to_iso3_codes(region: str, config: dict) -> List[str]:
    """Get ISO3 codes for region from config."""
    iso3_list = config.get("regions", {}).get(region, [])

    if not iso3_list:
        raise ValueError(
            f"Region '{region}' not found in config['regions'] or contains no ISO3 codes. "
            f"Available regions: {list(config.get('regions', {}).keys())}"
        )

    return iso3_list


# ============================================================================
# RENEWABLE PROFILES LOADING (NEW FORMAT)
# ============================================================================


def load_renewable_profiles(nc_path: str) -> xr.Dataset:
    """Load renewable profiles from new xarray format (single .nc file).

    Parameters
    ----------
    nc_path : str
        Path to renewable profiles netCDF file

    Returns
    -------
    xr.Dataset
        Dataset with dimensions [bus, technology, hour] and variables:
        - capacity_factor[bus, tech, hour]
        - p_nom_max[bus, tech]
        - avg_cf[bus, tech]
    """
    logger.info(f"Loading renewable profiles from {nc_path}...")

    try:
        dataset = xr.open_dataset(nc_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Renewable profiles file not found: {nc_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load renewable profiles: {e}") from e

    # Validate dimensions
    required_dims = {"bus", "technology", "hour"}
    actual_dims = set(dataset.dims.keys())
    if not required_dims.issubset(actual_dims):
        raise ValueError(
            f"Dataset missing required dimensions. Required: {required_dims}, "
            f"Found: {actual_dims}"
        )

    # Validate data variables
    required_vars = {"capacity_factor", "p_nom_max", "avg_cf"}
    actual_vars = set(dataset.data_vars.keys())
    if not required_vars.issubset(actual_vars):
        raise ValueError(
            f"Dataset missing required variables. Required: {required_vars}, "
            f"Found: {actual_vars}"
        )

    logger.info(
        f"✓ Loaded renewable profiles: {len(dataset.bus)} buses, "
        f"{len(dataset.technology)} technologies, {len(dataset.hour)} hours"
    )

    return dataset


def filter_by_region(dataset: xr.Dataset, region: str, config: dict) -> xr.Dataset:
    """Filter renewable profiles by region using ISO3 codes from bus_id.

    Parameters
    ----------
    dataset : xr.Dataset
        Unfiltered renewable profiles
    region : str
        Region name (key in config["regions"])
    config : dict
        Config dict with regions mapping

    Returns
    -------
    xr.Dataset
        Filtered dataset (subset of buses matching region's ISO3 codes)
    """
    # Get ISO3 codes for this region
    iso3_list = region_to_iso3_codes(region, config)
    logger.info(f"Region '{region}' maps to ISO3 codes: {iso3_list}")

    # Parse ISO3 from bus_id strings (format: "{ISO3}_{other_identifiers}")
    bus_ids = dataset.coords["bus"].values
    bus_iso3_codes = []

    for bus_id in bus_ids:
        bus_id_str = str(bus_id)
        # Extract ISO3 from bus_id (first component before underscore)
        iso3 = bus_id_str.split("_")[0]
        bus_iso3_codes.append(iso3)

    # Create mask for buses in this region
    bus_mask = np.isin(bus_iso3_codes, iso3_list)
    selected_buses = bus_ids[bus_mask]

    if len(selected_buses) == 0:
        raise ValueError(
            f"No buses found for region '{region}' with ISO3 {iso3_list}. "
            f"Available ISO3 codes in data: {np.unique(bus_iso3_codes)}"
        )

    # Filter dataset to selected buses
    filtered = dataset.sel(bus=selected_buses)

    logger.info(
        f"Filtered dataset to {len(selected_buses)} buses in region {region} "
        f"(from {len(bus_ids)} total)"
    )

    return filtered


def filter_by_technologies(dataset: xr.Dataset, config: dict) -> xr.Dataset:
    """Filter renewable technologies based on config.

    Parameters
    ----------
    dataset : xr.Dataset
        Renewable profiles with all technologies
    config : dict
        Config dict with optional `renewable_technologies` list

    Returns
    -------
    xr.Dataset
        Filtered dataset with only specified technologies
    """
    # Get allowed technologies from config, default to ["solar", "onwind"]
    allowed_techs = config.get("renewable_technologies", ["solar", "onwind"])
    logger.info(f"Allowed renewable technologies: {allowed_techs}")

    # Get available technologies in dataset
    available_techs = list(dataset.technology.values)
    logger.info(f"Available technologies in dataset: {available_techs}")

    # Find intersection of allowed and available
    techs_to_keep = [t for t in available_techs if str(t) in allowed_techs]

    if not techs_to_keep:
        raise ValueError(
            f"No renewable technologies available after filtering. "
            f"Allowed: {allowed_techs}, Available: {available_techs}"
        )

    # Filter dataset
    filtered = dataset.sel(technology=techs_to_keep)

    logger.info(
        f"Filtered to {len(techs_to_keep)} technologies: {techs_to_keep} "
        f"(from {len(available_techs)} available)"
    )

    return filtered


# ============================================================================
# ELECTRICITY BACK-PROPAGATION
# ============================================================================


def back_propagate_electricity_need(
    tech_costs: pd.Series, product: str, config: dict
) -> float:
    """Calculate renewable electricity requirement (MWh) per tonne of product.

    Back-propagates through supply chain efficiency chain:
    - steel (t): Electrolyzer(elec) + DRI(elec + H2) + optional EAF(elec)
    - hbi (t): Electrolyzer(elec) + DRI(elec + H2)
    - h2 (t): Electrolyzer(elec) only

    Parameters
    ----------
    tech_costs : pd.Series
        Technology cost database (MultiIndex by [tech_name, parameter])
    product : str
        Product: "steel", "hbi", or "h2"
    config : dict
        Config dict with optional overrides:
        - "electricity_per_tonne_{product}_mwh"
        - "eaf_electricity_source" ("grid" by default, "renewable" to include EAF electricity)

    Returns
    -------
    float
        Renewable electricity requirement in MWh per tonne of product
    """
    # Check for config override first
    override_key = f"electricity_per_tonne_{product}_mwh"
    if override_key in config:
        elec_need = config[override_key]
        logger.info(
            f"Using config override for {product}: "
            f"{override_key} = {elec_need:.4f} MWh/t"
        )
        return elec_need

    # Extract efficiencies from tech database
    # Electrolyzer: Electricity → H2
    elec_params = td.get_tech(tech_costs, "Alkaline electrolyzer large size")
    elec_mwh_per_mwh_h2 = td.get_tech_param(elec_params, "electricity-input", 1.38)
    logger.debug(
        f"Electrolyzer electricity input: {elec_mwh_per_mwh_h2:.4f} MWh/MWh H2"
    )

    if product == "h2":
        # H2 only: just electrolyzer electricity
        # Note: 1 MWh H2 ≈ 1 t H2 for energy accounting (MWh/MWh = MWh/t in energy terms)
        elec_need = elec_mwh_per_mwh_h2
        logger.info(
            f"Back-propagated electricity for H2: {elec_need:.4f} MWh/t H2 "
            f"(electrolyzer only)"
        )
        return elec_need

    # DRI Furnace: Iron ore + Hydrogen + Electricity → HBI
    dri_params = td.get_tech(tech_costs, "hydrogen direct iron reduction furnace")
    h2_per_t_hbi = td.get_tech_param(dri_params, "hydrogen-input", 2.1)
    dri_elec_per_t_hbi = td.get_tech_param(dri_params, "electricity-input", 1.03)
    logger.debug(f"DRI hydrogen input: {h2_per_t_hbi:.4f} t H2/t HBI")
    logger.debug(f"DRI electricity input: {dri_elec_per_t_hbi:.4f} MWh/t HBI")

    # Electricity for H2 production (via electrolyzer)
    h2_elec_per_t_hbi = h2_per_t_hbi * elec_mwh_per_mwh_h2

    if product == "hbi":
        # HBI: H2 production + DRI electricity
        elec_need = h2_elec_per_t_hbi + dri_elec_per_t_hbi
        logger.info(
            f"Back-propagated electricity for HBI: {elec_need:.4f} MWh/t HBI "
            f"(H2 production: {h2_elec_per_t_hbi:.4f}, DRI: {dri_elec_per_t_hbi:.4f})"
        )
        return elec_need

    if product == "steel":
        # EAF: HBI + Electricity → Steel
        eaf_params = td.get_tech(tech_costs, "electric arc furnace")
        eaf_elec_per_t_steel = td.get_tech_param(
            eaf_params, "electricity-input", 0.6395
        )
        logger.debug(f"EAF electricity input: {eaf_elec_per_t_steel:.4f} MWh/t Steel")

        eaf_source = config.get("eaf_electricity_source", "grid")

        # Steel renewable electricity: H2 production + DRI electricity.
        # If the EAF is configured to run on renewables, include its electricity too.
        elec_need = h2_elec_per_t_hbi + dri_elec_per_t_hbi
        if eaf_source == "renewable":
            elec_need += eaf_elec_per_t_steel
            logger.info(
                f"Back-propagated renewable electricity for Steel: {elec_need:.4f} MWh/t Steel "
                f"(H2 production: {h2_elec_per_t_hbi:.4f}, DRI: {dri_elec_per_t_hbi:.4f}, "
                f"EAF: {eaf_elec_per_t_steel:.4f})"
            )
        else:
            logger.info(
                f"Back-propagated renewable electricity for Steel: {elec_need:.4f} MWh/t Steel "
                f"(H2 production: {h2_elec_per_t_hbi:.4f}, DRI: {dri_elec_per_t_hbi:.4f}; "
                f"EAF electricity is grid-supplied)"
            )
        return elec_need

    raise ValueError(
        f"Product '{product}' not recognized. Choose from: 'h2', 'hbi', 'steel'"
    )


# ============================================================================
# TWO-STEP RENEWABLE FILTERING
# ============================================================================


def load_local_demand(local_demand_path: str, region: str, config: dict) -> float:
    """Load local electricity demand for region from CSV file.

    Parameters
    ----------
    local_demand_path : str
        Path to local demand CSV (columns: region, demand, el_share or similar)
    region : str
        Region name
    config : dict
        Configuration dict

    Returns
    -------
    float
        Local electricity demand in MWh/year. Returns 0 if region not found (with warning).
    """
    try:
        df = pd.read_csv(local_demand_path)
    except FileNotFoundError:
        logger.warning(
            f"Local demand file not found: {local_demand_path}. Using 0 MWh/year."
        )
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to load local demand: {e}. Using 0 MWh/year.")
        return 0.0

    # Look for region in dataframe (try common column names)
    region_col = None
    for col in ["region", "Region", "name", "Name"]:
        if col in df.columns:
            region_col = col
            break

    if region_col is None:
        logger.warning(
            f"Could not find region column in {local_demand_path}. Using 0 MWh/year."
        )
        return 0.0

    # Find demand value for this region
    region_data = df[df[region_col] == region]

    if region_data.empty:
        logger.warning(
            f"Region '{region}' not found in local demand data. "
            f"Available regions: {df[region_col].unique()}. Using 0 MWh/year."
        )
        return 0.0

    # Look for demand column (try common names)
    demand_col = None
    for col in ["demand", "Demand", "demand_mwh", "demand_MWh", "el_demand"]:
        if col in df.columns:
            demand_col = col
            break

    if demand_col is None:
        logger.warning(
            f"Could not find demand column in {local_demand_path}. Using 0 MWh/year."
        )
        return 0.0

    demand_mwh = float(region_data[demand_col].iloc[0])
    logger.info(
        f"Loaded local electricity demand for {region}: {demand_mwh:.1f} MWh/year"
    )

    return demand_mwh


def filter_renewable_generators_by_potential(
    dataset: xr.Dataset,
    region: str,
    product: str,
    tech_costs: pd.Series,
    config: dict,
    local_demand_mwh: float,
    max_product_demand_mt: float,
) -> Tuple[List[Dict], Dict]:
    """Two-step renewable filtering: reserve for domestic + select for product (per-tech).

    Parameters
    ----------
    dataset : xr.Dataset
        Filtered renewable profiles (already region + tech filtered)
    region : str
        Region name
    product : str
        Product (steel, hbi, h2)
    tech_costs : pd.Series
        Technology cost database
    config : dict
        Configuration dict
    local_demand_mwh : float
        Local electricity demand in MWh/year
    max_product_demand_mt : float
        Maximum product demand in Mt/year (for capacity target)

    Returns
    -------
    tuple
        (selected_generators, filter_audit)
        - selected_generators: List of dicts with bus, tech, p_nom_max, avg_cf
        - filter_audit: Dict with filtering statistics
    """
    logger.info("=" * 70)
    logger.info("TWO-STEP RENEWABLE FILTERING")
    logger.info("=" * 70)

    # Compute product renewable electricity need per tonne
    product_elec_per_t = back_propagate_electricity_need(tech_costs, product, config)

    # Target capacity (MW) for product production
    max_product_elec_mwh = max_product_demand_mt * 1e6 * product_elec_per_t
    max_product_elec_mw = max_product_elec_mwh / (365 * 24)

    # Multiplier from config (default 5)
    multiplier = config.get("renewable_coverage_multiplier", 5)
    target_capacity_mw = multiplier * max_product_elec_mw

    logger.info(f"{product.upper()} demand: {max_product_demand_mt:.1f} Mt")
    logger.info(
        f"{product.upper()} renewable electricity need: {product_elec_per_t:.4f} MWh/t → {max_product_elec_mw:.1f} MW average"
    )
    logger.info(
        f"Target renewable capacity ({multiplier}×): {target_capacity_mw:.1f} MW"
    )
    logger.info(
        f"Local demand: {local_demand_mwh:.1f} MWh/year → {local_demand_mwh / (365 * 24):.1f} MW average"
    )

    # STEP 1: Build bus-tech candidate list with all data
    candidates = []
    for bus_id in dataset.bus.values:
        for tech in dataset.technology.values:
            p_nom_max = float(
                dataset["p_nom_max"].sel(bus=bus_id, technology=tech).values
            )
            avg_cf = float(dataset["avg_cf"].sel(bus=bus_id, technology=tech).values)

            # Skip invalid combos
            if np.isnan(p_nom_max) or p_nom_max <= 0 or np.isnan(avg_cf) or avg_cf <= 0:
                continue

            candidates.append(
                {
                    "bus_id": str(bus_id),
                    "technology": str(tech),
                    "p_nom_max": p_nom_max,
                    "avg_cf": avg_cf,
                    "capacity_factor_ts": dataset["capacity_factor"]
                    .sel(bus=bus_id, technology=tech)
                    .values,
                    "potential": p_nom_max * avg_cf,  # Quality metric (MW × CF)
                }
            )

    if not candidates:
        logger.warning("No valid bus-tech candidates after filtering!")
        return [], {}

    logger.info(f"Total candidates: {len(candidates)}")

    # STEP 1: DOMESTIC RESERVATION
    logger.info("-" * 70)
    logger.info("STEP 1: RESERVE FOR DOMESTIC DEMAND")
    logger.info("-" * 70)

    # Sort globally by avg_cf (descending) for domestic reservation
    candidates_sorted_cf = sorted(candidates, key=lambda x: x["avg_cf"], reverse=True)

    local_demand_mw = local_demand_mwh / (365 * 24)
    reserved_generators = []
    reserved_capacity_mw = 0

    for candidate in candidates_sorted_cf:
        if reserved_capacity_mw >= local_demand_mw:
            break
        reserved_generators.append(candidate)
        reserved_capacity_mw += candidate["p_nom_max"]

    logger.info(
        f"Reserved {len(reserved_generators)} bus-tech combos for domestic demand"
    )
    logger.info(
        f"Reserved capacity: {reserved_capacity_mw:.1f} MW (target: {local_demand_mw:.1f} MW)"
    )

    # Create set of reserved IDs for filtering
    reserved_ids = {(g["bus_id"], g["technology"]) for g in reserved_generators}

    # STEP 2: STEEL SELECTION (PER-TECHNOLOGY)
    logger.info("-" * 70)
    logger.info("STEP 2: SELECT FOR STEEL PRODUCTION (PER-TECHNOLOGY)")
    logger.info("-" * 70)

    # Get non-reserved candidates
    non_reserved = [
        c for c in candidates if (c["bus_id"], c["technology"]) not in reserved_ids
    ]

    # Calculate per-technology potential shares
    tech_potentials = {}
    total_potential = 0
    for candidate in non_reserved:
        tech = candidate["technology"]
        potential = candidate["potential"]
        tech_potentials[tech] = tech_potentials.get(tech, 0) + potential
        total_potential += potential

    if total_potential <= 0:
        logger.warning("Total potential <= 0 in non-reserved generators!")
        total_potential = 1.0  # Avoid division by zero

    logger.info(f"Total non-reserved potential: {total_potential:.1f} MW·CF")

    # Per-tech targets (proportional to potential)
    tech_targets = {}
    for tech, pot in tech_potentials.items():
        share = pot / total_potential
        target_mw = share * target_capacity_mw
        tech_targets[tech] = target_mw
        logger.info(f"  {tech}: {share * 100:.1f}% share → target {target_mw:.1f} MW")

    # Select from each tech by avg_cf
    selected_generators = []
    selected_capacity_by_tech = {}

    for tech in sorted(tech_targets.keys()):
        target_mw = tech_targets[tech]

        # Get candidates for this tech, sort by avg_cf
        tech_candidates = sorted(
            [c for c in non_reserved if c["technology"] == tech],
            key=lambda x: x["avg_cf"],
            reverse=True,
        )

        tech_selected = []
        tech_capacity = 0

        for candidate in tech_candidates:
            if tech_capacity >= target_mw:
                break
            selected_generators.append(candidate)
            tech_selected.append(candidate)
            tech_capacity += candidate["p_nom_max"]

        selected_capacity_by_tech[tech] = tech_capacity
        logger.info(
            f"  {tech}: Selected {len(tech_selected)} generators, "
            f"{tech_capacity:.1f} MW (target: {target_mw:.1f} MW)"
        )

    total_selected_capacity = sum(selected_capacity_by_tech.values())

    logger.info("-" * 70)
    logger.info("FILTERING COMPLETE")
    logger.info("-" * 70)
    logger.info(f"Total generators selected: {len(selected_generators)}")
    logger.info(f"Total capacity selected: {total_selected_capacity:.1f} MW")
    logger.info(
        f"Coverage ratio: {total_selected_capacity / target_capacity_mw:.2f}× target"
    )

    # Build audit info
    filter_audit = {
        "region": region,
        "product": product,
        "local_demand_mwh": local_demand_mwh,
        "max_product_demand_mt": max_product_demand_mt,
        "product_elec_need_mwh_per_t": product_elec_per_t,
        "multiplier": multiplier,
        "target_capacity_mw": target_capacity_mw,
        "n_reserved_generators": len(reserved_generators),
        "reserved_capacity_mw": reserved_capacity_mw,
        "n_selected_generators": len(selected_generators),
        "selected_capacity_mw": total_selected_capacity,
        "selected_by_tech": selected_capacity_by_tech,
        "coverage_ratio": total_selected_capacity / target_capacity_mw
        if target_capacity_mw > 0
        else 0,
    }

    logger.info("=" * 70)

    return selected_generators, filter_audit


# ============================================================================
# EXTRACT INCREMENTAL GENERATOR SETS
# ============================================================================


def extract_incremental_generator_sets(
    selected_generators: List[Dict],
    product_demand_levels: List[float],
    product: str,
    tech_costs: pd.Series,
    config: dict,
) -> Dict[float, List[Dict]]:
    """Extract incremental generator subsets for each demand level.

    Given a ranked list of selected generators (from max demand), extracts smaller
    subsets for each demand level. Each subset is a proper subset of the next.

    Parameters
    ----------
    selected_generators : List[Dict]
        Ranked list from filter_renewable_generators_by_potential (at max demand)
    product_demand_levels : List[float]
        Product demand levels in Mt/year (e.g., [10, 200, 1000])
    product : str
        Product (steel, hbi, h2)
    tech_costs : pd.Series
        Technology cost database
    config : dict
        Configuration dict

    Returns
    -------
    dict
        Mapping: {demand_mt: [selected_generators_for_that_level]}
        Sets are nested: 10 Mt ⊆ 200 Mt ⊆ 1000 Mt
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING INCREMENTAL GENERATOR SETS")
    logger.info("=" * 70)

    # Compute renewable electricity need for this product
    elec_per_t = back_propagate_electricity_need(tech_costs, product, config)
    multiplier = config.get("renewable_coverage_multiplier", 5)

    # Calculate target MW for each demand level
    demand_targets = {}  # demand_mt -> target_mw
    for demand_mt in product_demand_levels:
        elec_mwh = demand_mt * 1e6 * elec_per_t
        elec_mw = elec_mwh / (365 * 24)
        target_mw = multiplier * elec_mw
        demand_targets[demand_mt] = target_mw
        logger.info(f"Demand {demand_mt:.1f} Mt → Target {target_mw:.1f} MW")

    # For each demand level, select generators up to its target
    incremental_sets = {}
    for demand_mt in sorted(product_demand_levels):
        target_mw = demand_targets[demand_mt]

        # Accumulate generators until reaching target
        subset = []
        accumulated_mw = 0
        for gen in selected_generators:
            if accumulated_mw >= target_mw:
                break
            subset.append(gen)
            accumulated_mw += gen["p_nom_max"]

        incremental_sets[demand_mt] = subset
        logger.info(
            f"  {demand_mt:.1f} Mt: {len(subset)} generators, "
            f"{accumulated_mw:.1f} MW (target {target_mw:.1f} MW)"
        )

    logger.info("=" * 70)
    return incremental_sets


def _serialize_dispatch_list(generators: List[Dict]) -> List[Dict]:
    """Return JSON-safe generator list metadata in selection order."""
    serialized = []
    for gen in generators:
        serialized.append(
            {
                "bus_id": str(gen["bus_id"]),
                "technology": str(gen["technology"]),
                "p_nom_max_mw": float(gen["p_nom_max"]),
                "avg_cf": float(gen["avg_cf"]),
            }
        )
    return serialized


# ============================================================================
# RENEWABLE GENERATOR ADDITION
# ============================================================================


def add_renewable_generators(
    network: pypsa.Network,
    dataset: xr.Dataset,
    tech_costs: pd.Series,
    config: dict,
    selected_generators: List[Dict] = None,
) -> Dict:
    """Add renewable generators from xarray dataset or pre-filtered list to electricity bus.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network to add generators to
    dataset : xr.Dataset
        Filtered renewable profiles with dimensions [bus, technology, hour]
    tech_costs : pd.Series
        Technology cost parameters
    config : dict
        Configuration dict
    selected_generators : List[Dict], optional
        Pre-filtered list from filter_renewable_generators_by_potential().
        If provided, ONLY these generators are added (faster).
        If None, all valid combinations from dataset are added.

    Returns
    -------
    dict
        Audit info with counts and statistics
    """
    # Map technology names to database keys for cost lookup
    tech_database_map = {
        "onwind": "onwind",
        "offwind-ac": "offwind",
        "solar": "solar-utility",
    }

    # Ensure carriers are defined before adding renewable generators.
    _ensure_carriers(
        network,
        ["renewable_electricity"] + [str(t) for t in dataset.technology.values],
    )

    # Use the network's discount_rate (which is set regionally in prepare_network)
    discount_rate = network.discount_rate

    # Validate data quality
    n_total_combos = len(dataset.bus) * len(dataset.technology)
    n_valid_combos = 0
    n_added_generators = 0
    total_p_nom_max = 0
    iso3_set = set()

    # If pre-filtered generators provided, use fast path
    if selected_generators is not None:
        logger.info(
            f"Adding {len(selected_generators)} pre-filtered generators (fast path)"
        )
        for gen_dict in selected_generators:
            bus_id = gen_dict["bus_id"]
            tech_str = gen_dict["technology"]
            p_nom_max = gen_dict["p_nom_max"]
            cf_timeseries = gen_dict["capacity_factor_ts"]
            avg_cf = gen_dict["avg_cf"]

            # Extract ISO3 from bus_id
            iso3 = str(bus_id).split("_")[0]
            iso3_set.add(iso3)

            # Handle timeseries NaNs
            if isinstance(cf_timeseries, np.ndarray):
                cf_timeseries = np.nan_to_num(cf_timeseries, nan=0.0)
            else:
                cf_timeseries = np.zeros(8760)

            # Get technology parameters from database
            db_tech_name = tech_database_map.get(tech_str, tech_str)
            tech_params = td.get_tech(tech_costs, db_tech_name)

            overnight_cost = (
                td.get_tech_param(tech_params, "investment", 0) * 1000
            )  # EUR/kW → EUR/MW
            lifetime = td.get_tech_param(tech_params, "lifetime", 20)
            fom_pct = td.get_tech_param(tech_params, "FOM", 0)
            fom_cost = overnight_cost * (fom_pct / 100) if overnight_cost > 0 else 0

            gen_name = f"renewable_{bus_id}_{tech_str}"

            # Add generator
            network.add(
                "Generator",
                gen_name,
                bus="renewable_electricity",
                carrier=tech_str,
                p_nom_extendable=True,
                p_nom=0,  # Start with no capacity; optimization will decide
                p_nom_max=p_nom_max,  # Upper ceiling from dataset (MW)
                p_max_pu=cf_timeseries,  # Hourly capacity factor (0-1)
                overnight_cost=overnight_cost,
                discount_rate=discount_rate,
                lifetime=lifetime,
                fom_cost=fom_cost,
            )

            n_added_generators += 1
            total_p_nom_max += p_nom_max

            logger.debug(
                f"Added generator {gen_name}: p_nom_max={p_nom_max:.1f} MW, "
                f"avg_cf={avg_cf:.3f}, overnight_cost={overnight_cost:.1f} EUR/MW"
            )

        n_valid_combos = len(selected_generators)
    else:
        # Original slow path: iterate through all xarray combos
        logger.info(
            f"Adding generators from full dataset ({n_total_combos} combos, slow path)"
        )
        for bus_id in dataset.bus.values:
            # Extract ISO3 from bus_id
            iso3 = str(bus_id).split("_")[0]
            iso3_set.add(iso3)

            for tech in dataset.technology.values:
                tech_str = str(tech)

                # Extract data for this bus-tech combination
                p_nom_max = float(
                    dataset["p_nom_max"].sel(bus=bus_id, technology=tech).values
                )
                avg_cf = float(
                    dataset["avg_cf"].sel(bus=bus_id, technology=tech).values
                )
                cf_timeseries = (
                    dataset["capacity_factor"].sel(bus=bus_id, technology=tech).values
                )

                # Skip invalid combinations (NaN or ≤0)
                if (
                    np.isnan(p_nom_max)
                    or p_nom_max <= 0
                    or np.isnan(avg_cf)
                    or avg_cf <= 0
                ):
                    continue

                n_valid_combos += 1

                # Handle timeseries NaNs
                if isinstance(cf_timeseries, np.ndarray):
                    cf_timeseries = np.nan_to_num(cf_timeseries, nan=0.0)
                else:
                    cf_timeseries = np.zeros(8760)

                # Get technology parameters from database
                db_tech_name = tech_database_map.get(tech_str, tech_str)
                tech_params = td.get_tech(tech_costs, db_tech_name)

                overnight_cost = (
                    td.get_tech_param(tech_params, "investment", 0) * 1000
                )  # EUR/kW → EUR/MW
                lifetime = td.get_tech_param(tech_params, "lifetime", 20)
                fom_pct = td.get_tech_param(tech_params, "FOM", 0)
                fom_cost = overnight_cost * (fom_pct / 100) if overnight_cost > 0 else 0

                gen_name = f"renewable_{bus_id}_{tech_str}"

                # Add generator
                network.add(
                    "Generator",
                    gen_name,
                    bus="renewable_electricity",
                    carrier=tech_str,
                    p_nom_extendable=True,
                    p_nom=0,  # Start with no capacity; optimization will decide
                    p_nom_max=p_nom_max,  # Upper ceiling from dataset (MW)
                    p_max_pu=cf_timeseries,  # Hourly capacity factor (0-1)
                    overnight_cost=overnight_cost,
                    discount_rate=discount_rate,
                    lifetime=lifetime,
                    fom_cost=fom_cost,
                )

                n_added_generators += 1
                total_p_nom_max += p_nom_max

                logger.debug(
                    f"Added generator {gen_name}: p_nom_max={p_nom_max:.1f} MW, "
                    f"avg_cf={avg_cf:.3f}, overnight_cost={overnight_cost:.1f} EUR/MW"
                )

    # Build audit info
    coverage_pct = (n_valid_combos / n_total_combos * 100) if n_total_combos > 0 else 0

    logger.info(
        f"Added {n_added_generators} renewable generators to network "
        f"({n_valid_combos}/{n_total_combos} valid combos, {coverage_pct:.1f}% coverage)"
    )
    logger.info(f"Total p_nom_max capacity: {total_p_nom_max:.1f} MW")

    if coverage_pct < 50:
        logger.warning(
            f"Low data coverage: {coverage_pct:.1f}% valid combos. "
            f"Consider checking data source."
        )

    return {
        "n_generators_added": n_added_generators,
        "n_valid_bus_tech_combos": n_valid_combos,
        "n_total_bus_tech_combos": n_total_combos,
        "coverage_pct": coverage_pct,
        "total_p_nom_max_mw": total_p_nom_max,
        "iso3_codes": sorted(list(iso3_set)),
        "technologies": sorted([str(t) for t in dataset.technology.values]),
    }


def _apply_discount_rate_to_components(
    network: pypsa.Network, discount_rate: float
) -> None:
    """Apply regional discount_rate to all cost-bearing components.

    Links, stores, and generators with costs must have discount_rate set for annualization.
    """
    # Apply to links
    for link_name, link_row in network.links.iterrows():
        has_cost = (
            pd.notna(link_row.get("overnight_cost")) and link_row["overnight_cost"] >= 0
        )
        if has_cost:
            network.links.at[link_name, "discount_rate"] = discount_rate

    # Apply to stores
    for store_name, store_row in network.stores.iterrows():
        has_cost = (
            pd.notna(store_row.get("overnight_cost"))
            and store_row["overnight_cost"] > 0
        )
        if has_cost:
            network.stores.at[store_name, "discount_rate"] = discount_rate

    # Apply to generators
    for gen_name, gen_row in network.generators.iterrows():
        has_cost = (
            pd.notna(gen_row.get("overnight_cost")) and gen_row["overnight_cost"] >= 0
        )
        if has_cost:
            network.generators.at[gen_name, "discount_rate"] = discount_rate

    logger.debug(
        f"Applied discount_rate={discount_rate:.4f} to all cost-bearing components"
    )


def _ensure_carriers(network: pypsa.Network, carrier_names: List[str]) -> None:
    """Add missing carriers to the network before components reference them."""
    for carrier_name in carrier_names:
        if carrier_name not in network.carriers.index:
            network.add("Carrier", carrier_name)


def _consistency_check(network: pypsa.Network) -> None:
    """Sanitize and check network consistency.

    PyPSA's sanitize() method automatically adds missing carriers and fixes consistency issues.
    """
    logger.info("Running network consistency check...")

    # PyPSA's built-in consistency check
    try:
        network.sanitize()
        logger.info("Network passed consistency check (sanitized)")
    except Exception as e:
        logger.warning(f"Network sanitization warning: {e}")


# ============================================================================
# SUPPLY CHAIN PRODUCT CUTOFF
# ============================================================================


def apply_product_cutoff(network: pypsa.Network, product: str) -> None:
    """Remove supply chain stages after target product (h2/hbi/steel)."""
    if product == "h2":
        logger.info("Product cutoff: Keeping electrolyzer only (H2 output)")

        # Remove conversion stages after hydrogen
        links_to_remove = ["dri", "eaf"]
        buses_to_remove = ["iron_ore", "hbi", "steel"]
        stores_to_remove = (
            ["hbi_storage"] if "hbi_storage" in network.stores.index else []
        )
        generators_to_remove = ["iron_ore"]

        for link_name in links_to_remove:
            if link_name in network.links.index:
                network.remove("Link", link_name)
                logger.debug(f"Removed link: {link_name}")

        for bus_name in buses_to_remove:
            try:
                network.remove("Bus", bus_name)
                logger.debug(f"Removed bus: {bus_name}")
            except ValueError:
                logger.debug(
                    f"Bus {bus_name} not found (already removed or not present)"
                )

        for store_name in stores_to_remove:
            if store_name in network.stores.index:
                network.remove("Store", store_name)
                logger.debug(f"Removed store: {store_name}")

        for gen_name in generators_to_remove:
            if gen_name in network.generators.index:
                network.remove("Generator", gen_name)
                logger.debug(f"Removed generator: {gen_name}")

    elif product == "hbi":
        logger.info("Product cutoff: Keeping electrolyzer + DRI (HBI output)")

        # Remove stages after HBI
        links_to_remove = ["eaf"]
        buses_to_remove = ["steel"]
        generators_to_remove = []

        for link_name in links_to_remove:
            if link_name in network.links.index:
                network.remove("Link", link_name)
                logger.debug(f"Removed link: {link_name}")

        for bus_name in buses_to_remove:
            try:
                network.remove("Bus", bus_name)
                logger.debug(f"Removed bus: {bus_name}")
            except ValueError:
                logger.debug(f"Bus {bus_name} not found")

        for gen_name in generators_to_remove:
            if gen_name in network.generators.index:
                network.remove("Generator", gen_name)

    elif product == "steel":
        logger.info("Product cutoff: Keeping full supply chain (Steel output)")
        # No removal; keep all stages
        pass

    else:
        raise ValueError(
            f"Product '{product}' not recognized. Choose from: 'h2', 'hbi', 'steel'"
        )


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================


def prepare_network(
    skeleton_network_path: str,
    renewable_nc_path: str,
    tech_costs_path: str,
    region: str,
    product: str,
    config: dict,
    local_demand_path: str = None,
) -> Tuple[pypsa.Network, Dict, Dict[float, List[Dict]]]:
    """Prepare network with incremental renewable filtering.

    Workflow:
    1. Load skeleton + renewables (filtered by region + tech)
    2. Call filter_renewable_generators_by_potential() with MAX demand
    3. Extract incremental subsets for each demand level (10 ⊆ 200 ⊆ 1000)
    4. Build FULL network with max demand set
    5. Apply product cutoff
    6. Return: (network, audit_info, incremental_sets)

    Parameters
    ----------
    skeleton_network_path : str
        Path to skeleton network
    renewable_nc_path : str
        Path to renewable profiles .nc file (NEW xarray format)
    tech_costs_path : str
        Path to technology costs CSV
    region : str
        Region name
    product : str
        Product (h2, hbi, or steel)
    config : dict
        Configuration dict with:
        - steel_demand_levels: [10, 200, 1000] Mt/year (applies to any product)
        - renewable_technologies: ["solar", "onwind"]
        - renewable_coverage_multiplier: 5
    local_demand_path : str, optional
        Path to local electricity demand CSV (columns: region, demand_mwh)

    Returns
    -------
    tuple
        (network, audit_info, incremental_sets)
        - network: PyPSA Network with full max-demand renewable set
        - audit_info: Dict with filtering + network stats
        - incremental_sets: {demand_mt: [selected_generators]} nested subsets
    """
    logger.info("=" * 70)
    logger.info(f"Preparing network for region={region}, product={product}")
    logger.info("=" * 70)

    # Step 0: Extract demand levels and max demand (config uses 'steel_demand_levels' for backward compat)
    product_demand_levels = config.get("steel_demand_levels", [10, 200, 1000])
    max_product_demand_mt = max(product_demand_levels)
    logger.info(f"Demand levels: {product_demand_levels} Mt/year")
    logger.info(f"Max demand: {max_product_demand_mt} Mt/year")

    # Step 1: Load skeleton network
    logger.info("Loading skeleton network...")
    network = pypsa.Network(skeleton_network_path)
    network.name = f"Skeleton-{region}-{product}"
    logger.info(
        f"Skeleton loaded: {len(network.buses)} buses, "
        f"{len(network.links)} links, {len(network.stores)} stores"
    )

    # Set snapshots
    cost_year = None
    if "snakemake" in globals():
        cost_year = getattr(snakemake.wildcards, "cost_year", None)

    if cost_year is not None:
        network.set_snapshots(
            pd.date_range(f"{cost_year}-01-01", periods=8760, freq="h")
        )
        logger.info(f"Set snapshots for cost_year={cost_year}")
    else:
        raise ValueError(
            "cost_year must be defined in snakemake wildcards for prepare_network"
        )

    # Step 1b: Set interest rate (discount rate)
    interest_rates = config.get("interest_rate", {})
    discount_rate = interest_rates.get(region, interest_rates.get("default", 0.07))
    network.discount_rate = discount_rate
    logger.info(f"Set discount rate: {discount_rate:.4f} for region {region}")

    # Step 2: Load tech costs
    logger.info("Loading technology costs...")
    tech_costs = td.load_tech_costs(tech_costs_path)

    # Step 3: Load renewable profiles (NEW FORMAT)
    logger.info(f"Loading renewable profiles for region {region}...")
    renewable_dataset = load_renewable_profiles(renewable_nc_path)

    # Step 3b: Filter by region
    renewable_dataset = filter_by_region(renewable_dataset, region, config)

    # Step 3c: Filter by allowed technologies
    renewable_dataset = filter_by_technologies(renewable_dataset, config)

    # Step 4: LOAD LOCAL DEMAND for renewable filtering
    local_demand_mwh = 0.0
    if local_demand_path:
        local_demand_mwh = load_local_demand(local_demand_path, region, config)
    else:
        logger.warning("local_demand_path not provided, using 0 MWh/year for filtering")

    # Step 5: FILTER RENEWABLES (with MAX demand)
    logger.info("Filtering renewable generators (max demand set)...")
    selected_generators, filter_audit = filter_renewable_generators_by_potential(
        dataset=renewable_dataset,
        region=region,
        product=product,
        tech_costs=tech_costs,
        config=config,
        local_demand_mwh=local_demand_mwh,
        max_product_demand_mt=max_product_demand_mt,
    )

    # Step 6: EXTRACT INCREMENTAL GENERATOR SETS
    logger.info("Extracting incremental generator sets...")
    incremental_sets = extract_incremental_generator_sets(
        selected_generators=selected_generators,
        product_demand_levels=product_demand_levels,
        product=product,
        tech_costs=tech_costs,
        config=config,
    )

    # Step 7: Add renewable generators (FULL SET from filtering)
    logger.info("Adding renewable generators to network (full max-demand set)...")
    gen_audit = add_renewable_generators(
        network=network,
        dataset=renewable_dataset,
        tech_costs=tech_costs,
        config=config,
        selected_generators=selected_generators,  # Pre-filtered list (fast path)
    )

    # Step 8: Apply product-specific cutoff
    logger.info(f"Applying product cutoff for {product}...")
    apply_product_cutoff(network=network, product=product)

    # Step 9: Build audit info
    audit_info = {
        "region": region,
        "product": product,
        "discount_rate": discount_rate,
        "iso3_list": gen_audit["iso3_codes"],
        "num_buses_in_region": len(renewable_dataset.bus),
        "num_technologies": len(gen_audit["technologies"]),
        "num_generators_added": gen_audit["n_generators_added"],
        "data_coverage_pct": gen_audit["coverage_pct"],
        "total_renewable_p_nom_max_mw": gen_audit["total_p_nom_max_mw"],
        "technologies": gen_audit["technologies"],
        "filter_audit": filter_audit,
        "incremental_set_counts": {
            demand_mt: len(gen_list) for demand_mt, gen_list in incremental_sets.items()
        },
        "dispatch_order_full": _serialize_dispatch_list(selected_generators),
        "dispatch_order_by_demand_mt": {
            str(int(demand_mt))
            if float(demand_mt).is_integer()
            else str(demand_mt): _serialize_dispatch_list(gen_list)
            for demand_mt, gen_list in incremental_sets.items()
        },
        "network_stats": {
            "num_buses": len(network.buses),
            "num_links": len(network.links),
            "num_generators": len(network.generators),
            "num_stores": len(network.stores),
        },
    }

    logger.info("Network preparation complete:")
    logger.info(f"  - Region: {region} (ISO3: {audit_info['iso3_list']})")
    logger.info(f"  - Buses in region: {audit_info['num_buses_in_region']}")
    logger.info(f"  - Generators added: {audit_info['num_generators_added']}")
    logger.info(f"  - Data coverage: {audit_info['data_coverage_pct']:.1f}%")
    logger.info(
        f"  - Total p_nom_max: {audit_info['total_renewable_p_nom_max_mw']:.1f} MW"
    )
    logger.info(f"  - Incremental sets: {audit_info['incremental_set_counts']}")
    logger.info(
        f"  - Network: {audit_info['network_stats']['num_buses']} buses, "
        f"{audit_info['network_stats']['num_generators']} generators"
    )

    # Apply regional discount_rate to all cost-bearing components
    _apply_discount_rate_to_components(network, discount_rate)

    # Run consistency check
    _consistency_check(network)

    logger.info("=" * 70)

    return network, audit_info, incremental_sets


# ============================================================================
# SNAKEMAKE INTEGRATION
# ============================================================================

if __name__ == "__main__":
    # Handle Snakemake or mock invocation
    if "snakemake" not in globals():
        # For testing: mock Snakemake

        class MockSnakemake:
            """Mock Snakemake object for testing."""

            def __init__(self):
                self.input = {
                    "skeleton": "../resources/networks/skeleton_2030.nc",
                    "renewable_nc": "../data/renewable_profiles/renewable_profiles_EU__20260420_142941.nc",
                    "costs": "../resources/technology_data/costs_2030.csv",
                    "local_demand": "../data/local_demand.csv",
                }
                self.output = {
                    "base_network": "test_base_network.nc",
                    "audit": "test_audit.json",
                    "incremental_sets": "test_incremental_sets.json",
                }
                self.wildcards = {
                    "region": "EU",
                    "product": "steel",
                    "cost_year": "2030",
                }
                self.config = {
                    "regions": {
                        "EU": ["DEU", "FRA", "ITA", "NLD"],
                        "Africa": ["EGY", "ZAF"],
                    },
                    "renewable_technologies": ["solar", "onwind"],
                    "renewable_coverage_multiplier": 5,
                    "steel_demand_levels": [10, 200, 1000],
                    "interest_rate": {"default": 0.07},
                }

        snakemake = MockSnakemake()

    # Prepare network
    try:
        local_demand_path = None
        if hasattr(snakemake.input, "local_demand") and snakemake.input.local_demand:
            local_demand_path = snakemake.input.local_demand

        network, audit_info, incremental_sets = prepare_network(
            skeleton_network_path=snakemake.input.skeleton,
            renewable_nc_path=snakemake.input.renewable_nc,
            tech_costs_path=snakemake.input.costs,
            region=snakemake.wildcards.region,
            product=snakemake.wildcards.product,
            config=snakemake.config,
            local_demand_path=local_demand_path,
        )

        # Save outputs
        network.export_to_netcdf(snakemake.output.base_network)
        logger.info(f"Network saved to {snakemake.output.base_network}")

        with open(snakemake.output.audit, "w") as f:
            json.dump(audit_info, f, indent=2, default=str)
        logger.info(f"Audit info saved to {snakemake.output.audit}")

        # Save incremental generator sets
        incremental_sets_serializable = {
            int(demand_mt): [
                {
                    "bus_id": gen["bus_id"],
                    "technology": gen["technology"],
                    "p_nom_max": float(gen["p_nom_max"]),
                    "avg_cf": float(gen["avg_cf"]),
                }
                for gen in gen_list
            ]
            for demand_mt, gen_list in incremental_sets.items()
        }
        with open(snakemake.output.incremental_sets, "w") as f:
            json.dump(incremental_sets_serializable, f, indent=2)
        logger.info(f"Incremental sets saved to {snakemake.output.incremental_sets}")

    except Exception as e:
        logger.error(f"Network preparation failed: {e}", exc_info=True)
        raise
