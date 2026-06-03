"""
Prepare regional network: load clustered renewables and configure supply chain.

This is the simplified Step 1 workflow that:
1. Loads the clustered renewable profiles directly (region, technology, class, time)
2. Creates PyPSA generators for each technology and class
3. Applies local demand reservation if configured
4. Adds supply chain (electrolyzer, DRI, optional EAF)
5. Applies product-specific cutoff

Usage (Snakemake rule):
    rule prepare_regional_network:
        input:
            skeleton = "resources/networks/skeleton.nc",
            renewables = "data/clustered_renewables.nc",
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
from typing import Any, Dict, Optional, Tuple, Iterable, cast
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import xarray as xr  # type: ignore
import pypsa  # type: ignore

import tech_database as td

from trade_chain_utils import (
    build_product_components,
    get_external_material_inputs,
)
from _helpers import setup_logging

snakemake: Any = globals().get("snakemake")

logger = setup_logging(__name__, snakemake=snakemake)


def load_regional_clustered_renewables(
    clustered_path: str,
    region: str,
) -> Tuple[Dict[str, np.ndarray], xr.DataArray, Dict]:
    """
    Load renewable data for a region from clustered NetCDF.

    Parameters
    ----------
    clustered_path : str
        Path to data/clustered_renewables.nc
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
    with xr.open_dataset(clustered_path) as ds:
        if region not in ds.region.values:
            available = ", ".join(ds.region.values)
            raise ValueError(f"Region '{region}' not found. Available: {available}")

        logger.info(f"Loading clustered renewables for {region}")

        # Select region (dims: technology, class)
        region_cap = ds["capacity"].sel(region=region)  # (tech, class)
        region_cf = ds["capacity_factor"].sel(region=region)  # (tech, class, time)

        # Extract technology names and data
        techs = list(region_cap.technology.values)
        technologies_dict = {}

        for tech in techs:
            cap = region_cap.sel(technology=tech).values
            # Drop NaN-padded trailing classes
            valid = ~np.isnan(cap)
            technologies_dict[tech] = cap[valid]
            logger.info(
                f"  {tech}: {valid.sum()} classes, {np.nansum(cap):.0f} MW total"
            )

        # Capacity factor time series (keep full structure for now)
        cf_ts = region_cf  # (tech, class, time)

        # Validate capacity factors: clamp to [0, 1] and replace NaN with 0
        # This prevents infeasibility warnings from PyPSA when p_max_pu goes negative or exceeds 1
        n_invalid_before = int(
            ((cf_ts < 0) | (cf_ts > 1) | cf_ts.isnull()).sum().values
        )
        cf_ts = cf_ts.clip(0, 1).fillna(0)
        if n_invalid_before > 0:
            logger.warning(f"Fixed {n_invalid_before} invalid CF values (clamped/NaN)")

        avg_cf_data = {}
        if "avg_cf" in ds.data_vars:
            for tech in techs:
                avg = ds["avg_cf"].sel(region=region, technology=tech).values
                valid = ~np.isnan(avg)
                avg_cf_data[tech] = avg[valid]

        metadata = {
            "region": region,
            "n_classes": {
                tech: int((~np.isnan(region_cap.sel(technology=tech).values)).sum())
                for tech in techs
            },
            "n_time": region_cf.sizes["time"],
            "n_technologies": len(techs),
            "technologies": techs,
            "time_start": pd.Timestamp(ds["time"].values[0]),
            "time_end": pd.Timestamp(ds["time"].values[-1]),
            "total_capacity_mw": float(region_cap.sum().values),
            "avg_cf": avg_cf_data,
        }

        logger.info(
            f"  Total capacity: {metadata['total_capacity_mw']:.0f} MW, "
            f"{metadata['n_time']} timesteps, {metadata['n_technologies']} technologies"
        )

    return technologies_dict, cf_ts, metadata


def load_local_electricity_demand_mw(
    local_demand_path: Optional[str],
    region: str,
) -> float:
    """Load regional electricity demand and convert it to average MW."""

    if not local_demand_path:
        return 0.0

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
) -> Optional[Dict[str, np.ndarray]]:
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
        "unreserved": skip reservation, return None

    Returns
    -------
    dict or None
        Dict of same structure as technologies_dict, with NaN for non-reserved sites.
        If scenario="unreserved", returns None (no reservation).
    """
    if scenario == "unreserved":
        logger.info(
            "Scenario=unreserved: skipping site reservation (all generators available)"
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
    reserved_techs: Optional[Dict[str, np.ndarray]] = None,
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

    # Ensure electricity bus exists (umbrella renewable carrier)
    elec_bus = "renewable_electricity"
    if elec_bus not in network.buses.index:
        network.add("Bus", elec_bus, carrier="renewable_electricity", unit="MW")

    # Get discount rate
    discount_rate = network.discount_rate

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

            # Ensure p_max_pu is valid (should be [0, 1] after data validation above)
            if np.any(np.isnan(p_max_pu)):
                logger.warning(
                    f"Generator {gen_name}: p_max_pu contains NaN values, filling with 0"
                )
                p_max_pu = np.nan_to_num(p_max_pu, nan=0.0)
            if np.any(p_max_pu < 0) or np.any(p_max_pu > 1):
                logger.warning(
                    f"Generator {gen_name}: p_max_pu out of bounds [0,1], clamping"
                )
                p_max_pu = np.clip(p_max_pu, 0, 1)

            # Determine tech-specific carrier while keeping generators on the
            # shared `renewable_electricity` bus. This preserves per-tech
            # statistics while modelling a common electricity bus.
            tech_to_carrier = {
                "pvplant": "renewable_pv",
                "windonshore": "renewable_wind_onshore",
                "windoffshore": "renewable_wind_offshore",
            }
            carrier_name = tech_to_carrier.get(tech, f"renewable_{tech}")

            # Add generator (extendable with ceiling)
            network.add(
                "Generator",
                gen_name,
                bus=elec_bus,
                carrier=carrier_name,
                p_nom_extendable=True,
                p_nom=0,  # Start with no capacity; optimization will decide
                p_nom_min=0,
                p_nom_max=p_nom_max,  # Upper ceiling from dataset (MW)
                p_max_pu=p_max_pu,  # Hourly capacity factor (0-1)
                overnight_cost=overnight_cost,
                discount_rate=discount_rate,
                lifetime=lifetime,
                fom_cost=fom_cost,
                tags={
                    "technology": tech,
                    "resource_tech": tech,
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


def _product_has_renewables(config: dict, product: str) -> bool:
    """Check if a product uses renewable electricity in its network.

    Returns True if the product should have renewable generators and reservation logic applied.
    Returns False if the product uses grid electricity only.
    """
    try:
        return bool(
            build_product_components(config, product).get("has_renewables", False)
        )
    except Exception:
        return False


def sanitize_and_fix(
    network: pypsa.Network, logger: Optional[logging.Logger] = None
) -> None:
    """Run `network.sanitize()` and apply small, safe fixes.

    Fixes applied:
    - Ensure renewable generators have `p_nom_min = 0`.
    - Clamp `p_nom_min` to `p_nom_max` when inconsistent.
    - Logs a summary of applied fixes.
    """
    if logger is None:
        _logger = globals()["logger"]
    else:
        _logger = logger

    _logger.info("Sanitizing network (PyPSA sanitize + post-fix checks)...")
    try:
        network.sanitize()
    except Exception as exc:
        _logger.warning(f"network.sanitize() raised an exception: {exc}")

    fixes = []

    # operate on a snapshot of the generators DataFrame to avoid SettingWithCopy
    if len(network.generators) == 0:
        _logger.info("No generators to check during sanitize_and_fix.")
        return

    gens = network.generators
    for gen in gens.index:
        try:
            carrier = gens.loc[gen, "carrier"]
        except Exception:
            carrier = None

        # Ensure renewable generators have zero minimum
        if isinstance(carrier, str) and carrier.startswith("renewable_"):
            try:
                current_pmin = (
                    gens.loc[gen, "p_nom_min"] if "p_nom_min" in gens.columns else None
                )
            except Exception:
                current_pmin = None
            # Set p_nom_min to 0 if not set or positive
            try:
                if current_pmin is None or (
                    pd.notna(current_pmin) and float(current_pmin) != 0.0
                ):
                    network.generators.loc[gen, "p_nom_min"] = 0.0
                    fixes.append(f"set p_nom_min=0 for {gen}")
            except Exception:
                # best-effort; continue
                pass

        # Clamp p_nom_min <= p_nom_max
        try:
            pmin = (
                network.generators.loc[gen, "p_nom_min"]
                if "p_nom_min" in network.generators.columns
                else None
            )
            pmax = (
                network.generators.loc[gen, "p_nom_max"]
                if "p_nom_max" in network.generators.columns
                else None
            )
            if pd.notna(pmin) and pd.notna(pmax):
                # If pmax < pmin, reduce pmin to pmax
                if float(cast(Any, pmax)) < float(cast(Any, pmin)):
                    network.generators.loc[gen, "p_nom_min"] = float(cast(Any, pmax))
                    fixes.append(f"clamped p_nom_min to p_nom_max for {gen}")
        except Exception:
            pass

    if fixes:
        _logger.info(f"sanitize_and_fix applied {len(fixes)} fixes: {fixes[:10]}")
    else:
        _logger.info("sanitize_and_fix applied no fixes")


def _get_components_for_product(config: dict, product: str) -> Tuple[set, set, set]:
    """Get links, stores, and buses for a product from the configured stage groups.

    Returns (keep_links, keep_stores, keep_buses) sets.
    """
    comp = build_product_components(config, product)
    keep_links = set(cast(Iterable[str], comp.get("links") or []))
    keep_stores = set(cast(Iterable[str], comp.get("stores") or []))
    keep_buses = set(cast(Iterable[str], comp.get("buses") or []))

    logger.info(
        f"Components for product={product}: links={keep_links}, stores={keep_stores}, buses={keep_buses}"
    )
    return keep_links, keep_stores, keep_buses


def apply_product_cutoff(
    network: pypsa.Network, product: str, config: Optional[dict] = None
) -> None:
    """Remove supply chain components beyond the target product.

    Uses config.product_components to determine which components to keep.
    Removes all links and stores not needed for the target product.
    Preserves the output buses for the product (e.g., 'steel' for steel product).
    Also removes orphaned buses (buses with no connected components).
    """
    if config is None:
        config = {}

    # Get the set of links, stores, and buses to keep for this product
    keep_links, keep_stores, keep_buses = _get_components_for_product(config, product)

    # Remove links not in the keep set
    for link_name in list(network.links.index):
        if link_name not in keep_links:
            try:
                network.remove("Link", link_name)
                logger.info(
                    f"Removed Link: {link_name} (not needed for product={product})"
                )
            except Exception as e:
                logger.warning(f"Could not remove Link {link_name}: {e}")

    # Remove stores not in the keep set
    for store_name in list(network.stores.index):
        if store_name not in keep_stores:
            try:
                network.remove("Store", store_name)
                logger.info(
                    f"Removed Store: {store_name} (not needed for product={product})"
                )
            except Exception as e:
                logger.warning(f"Could not remove Store {store_name}: {e}")

    # Remove orphaned buses (buses not connected to any remaining component)
    # BUT preserve buses listed in keep_buses (output bus for this product)
    for bus_name in list(network.buses.index):
        # Skip essential supply buses and product output buses
        if (
            bus_name in ["renewable_electricity", "grid_electricity"]
            or bus_name in keep_buses
        ):
            continue

        has_connection = False

        # Check if bus is used by any link (bus0, bus1, bus2, bus3)
        if len(network.links) > 0:
            for bcol in ["bus0", "bus1", "bus2", "bus3"]:
                if (
                    bcol in network.links.columns
                    and (network.links[bcol] == bus_name).any()
                ):
                    has_connection = True
                    break

        # Check generators
        if not has_connection and len(network.generators) > 0:
            if (network.generators["bus"] == bus_name).any():
                has_connection = True

        # Check stores
        if not has_connection and len(network.stores) > 0:
            if (network.stores["bus"] == bus_name).any():
                has_connection = True

        # Check loads
        if not has_connection and len(network.loads) > 0:
            if (network.loads["bus"] == bus_name).any():
                has_connection = True

        # Remove if orphaned
        if not has_connection:
            try:
                network.remove("Bus", bus_name)
                logger.info(f"Removed orphaned Bus: {bus_name}")
            except Exception as e:
                logger.warning(f"Could not remove Bus {bus_name}: {e}")


def prepare_network(
    skeleton_network_path: str,
    clustered_renewables_path: str,
    tech_costs_path: str,
    local_demand_path: Optional[str],
    region: str,
    product: str,
    cost_year: int = 2030,
    config: Optional[dict] = None,
    scenario: str = "reserved",
    route_label: Optional[str] = None,
) -> Tuple[pypsa.Network, Dict]:
    """Prepare regional network with clustered renewables.

    Parameters
    ----------
    skeleton_network_path : str
        Path to base network topology
    clustered_renewables_path : str
        Path to clustered renewables NetCDF
    tech_costs_path : str
        Path to technology cost database
    region : str
        Region name
    product : str
        Target product (h2, hbi, steel)
    cost_year : int
        Cost year for technology parameters
    config : dict, optional
        Configuration dict
    scenario : str
        "reserved" (default): apply high-CF site reservation for domestic demand
        "unreserved": no reservation; full renewable stack available (fallback scenario)
    route_label : str, optional
        If provided, slice skeleton to this stage only (e.g., "hbi", "steel")
        This enables independent per-stage solves for Option B semantics.

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
    logger.info(
        f"Preparing network: region={region}, product={product}, route_label={route_label}"
    )
    logger.info("=" * 70)

    # Load skeleton (prefer stage-group specific skeleton when available)
    logger.info("Loading skeleton network...")
    skeleton_to_load = skeleton_network_path
    if route_label:
        try:
            from pathlib import Path

            p = Path(skeleton_network_path)
            stem = p.stem
            suffix = p.suffix
            # Expect group-specific files like 'generic_model_2050_hbi.nc'
            candidate = p.with_name(f"{stem}_{route_label}{suffix}")
            if candidate.exists():
                logger.info(
                    f"Found group-specific skeleton for route_label={route_label}: {candidate}"
                )
                skeleton_to_load = str(candidate)
            else:
                logger.info(
                    f"No group-specific skeleton found for {route_label}; using {skeleton_network_path}"
                )
        except Exception:
            # Fallback to provided skeleton path
            skeleton_to_load = skeleton_network_path

    network = pypsa.Network(skeleton_to_load)
    network.name = f"base_{cost_year}_{region}_{product}"

    # STAGE SLICING: if route_label provided, slice skeleton to that stage only
    # Note: previously we skipped slicing when product == route_label (because
    # Snakemake params set `product` to the same value). Always slice when a
    # `route_label` is supplied to ensure per-stage networks are produced.
    if route_label:
        logger.info(f"Slicing skeleton to route_label={route_label}")

        # Use configured component resolver to derive which PyPSA components
        # (links, stores, buses) belong to this stage-group. This keeps the
        # slicing logic driven by `trade_chain` config and the canonical
        # TECH_COMPONENT_MAP in `trade_chain_utils.py`.
        try:
            keep_links, keep_stores, keep_buses = _get_components_for_product(
                config, route_label
            )
        except Exception as exc:
            logger.warning(
                f"Could not derive components for route_label={route_label}: {exc}; keeping full skeleton"
            )
            keep_links, keep_stores, keep_buses = set(), set(), set()

        # If resolver returned empty sets, warn and keep full skeleton
        if not (keep_links or keep_stores or keep_buses):
            logger.warning(
                f"Component resolver returned no components for {route_label}; keeping full skeleton"
            )
        else:
            # Remove links not in keep_links
            for link_name in list(network.links.index):
                if link_name not in keep_links:
                    try:
                        network.remove("Link", link_name)
                        logger.info(f"Removed Link: {link_name}")
                    except Exception as e:
                        logger.warning(f"Could not remove Link {link_name}: {e}")

            # Remove stores not in keep_stores
            for store_name in list(network.stores.index):
                if store_name not in keep_stores:
                    try:
                        network.remove("Store", store_name)
                        logger.info(f"Removed Store: {store_name}")
                    except Exception as e:
                        logger.warning(f"Could not remove Store {store_name}: {e}")

            # Add free external inputs ONLY for external materials of this
            # configured stage-group.
            external_material_inputs = set(
                get_external_material_inputs(config, route_label)
            )

            for bus_name in sorted(external_material_inputs):
                if bus_name not in network.buses.index:
                    logger.warning(
                        f"Expected material input bus missing during stage slicing: {bus_name}; skipping free input generator"
                    )
                    continue

                gen_name = f"{bus_name}_input"
                if gen_name not in network.generators.index:
                    try:
                        network.add(
                            "Generator",
                            gen_name,
                            bus=bus_name,
                            carrier=bus_name,
                            p_nom=1e10,
                            marginal_cost=0,
                        )
                        logger.info(
                            f"Added external free input generator from trade chain: {gen_name} on {bus_name}"
                        )
                    except Exception as e:
                        logger.warning(f"Could not add free input {gen_name}: {e}")

            logger.info(
                f"Skeleton sliced to {route_label}: {len(network.links)} links, {len(network.stores)} stores"
            )

    if snakemake.wildcards.wacc == "regional":
        logger.info("applying region specific wacc")
        wacc = pd.read_csv(snakemake.input.wacc, header=0)
        wacc.set_index("region", inplace=True)
        discount_rate = wacc.loc[region].values[0]

    elif snakemake.wildcards.wacc == "uniform":
        discount_rate = snakemake.params.uniform_interest_rate

    else:
        raise ValueError(
            f"Unrecognized wacc wildcard: {snakemake.wildcards.wacc}. "
            f"Expected 'regional' or 'uniform'."
        )

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

    # Load consolidated renewables for region (only if this stage needs renewables)
    # Check product_components config to see if product uses renewable_electricity
    product_uses_renewables = _product_has_renewables(config, route_label or product)

    if not product_uses_renewables:
        logger.info(
            f"Product '{route_label}' does not use renewable_electricity: "
            f"skipping renewable generator loading"
        )
        techs_dict = {}
        cf_ts = None
        metadata = {}
    else:
        logger.info("Loading consolidated renewables...")
        techs_dict, cf_ts, metadata = load_regional_clustered_renewables(
            clustered_renewables_path, region
        )
    # Apply local demand reservation if configured (only for products with renewable_electricity)
    # For scenario="reserved", reserve high-CF sites; for "unreserved", skip reservation
    reserved_techs = None
    if product_uses_renewables:
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
    else:
        logger.info(
            f"Skipping reservation: product '{route_label}' does not use renewables"
        )

    # Add renewable generators (only if techs_dict is not empty)
    logger.info("Adding renewable generators...")
    if techs_dict:
        gen_audit = add_renewable_generators(
            network, region, techs_dict, cf_ts, tech_costs, config, reserved_techs
        )
    else:
        logger.info(
            "Skipping renewable generator addition (no technologies for this stage)"
        )
        gen_audit = {
            "n_generators_added": 0,
            "total_capacity_mw": 0,
            "n_reserved": 0,
        }
    # Apply product cutoff only when using full skeleton.
    # When route_label is provided (dedicated stage-group skeleton), the network
    # is already scoped to the correct components, so cutoff is redundant.
    if route_label:
        logger.info(
            f"Using dedicated stage-group skeleton for {route_label}; skipping product cutoff"
        )
    else:
        logger.info(f"Applying product cutoff for {product}...")
        apply_product_cutoff(network, product, config=config)

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

    # Run centralized sanitization and small automatic fixes
    sanitize_and_fix(network, logger=logger)

    # Stage-slicing helper: produce a subnetwork containing only the specified process carrier
    def build_stage_subnetwork(n: pypsa.Network, process_carrier: str) -> pypsa.Network:
        """Return a deep copy of the network pruned to links with carrier == process_carrier

        Keeps:
        - Links whose `carrier` equals `process_carrier`.
        - Generators/stores attached to buses referenced by those links (e.g., raw resource suppliers).
        - Removes other conversion links and any isolated buses.

        The returned subnetwork is suitable for independent per-stage marginal solves (Option B semantics).
        """
        import copy

        sub = copy.deepcopy(n)

        # Remove links that are not the target process carrier
        for link_name in list(sub.links.index):
            carrier = sub.links.loc[link_name, "carrier"]
            if carrier != process_carrier:
                sub.remove("Link", link_name)

        # Remove generators not attached to remaining buses
        for gen_name in list(sub.generators.index):
            gen_bus = sub.generators.loc[gen_name, "bus"]
            if gen_bus not in sub.buses.index:
                try:
                    sub.remove("Generator", gen_name)
                except Exception:
                    pass

        # Remove stores not attached to remaining buses
        for store_name in list(sub.stores.index):
            store_bus = sub.stores.loc[store_name, "bus"]
            if store_bus not in sub.buses.index:
                try:
                    sub.remove("Store", store_name)
                except Exception:
                    pass

        # Remove isolated buses (no generators, no links, no stores)
        for bus_name in list(sub.buses.index):
            has_gen = (
                len(sub.generators.index[sub.generators["bus"] == bus_name]) > 0
                if len(sub.generators) > 0
                else False
            )
            has_store = (
                len(sub.stores.index[sub.stores["bus"] == bus_name]) > 0
                if len(sub.stores) > 0
                else False
            )
            has_link = False
            if len(sub.links) > 0:
                # check bus presence in any of the bus columns
                for link_name in sub.links.index:
                    row = sub.links.loc[link_name]
                    for bcol in ["bus0", "bus1", "bus2", "bus3"]:
                        if bcol in row.index and row.get(bcol) == bus_name:
                            has_link = True
                            break
                    if has_link:
                        break

            if not (has_gen or has_store or has_link):
                try:
                    sub.remove("Bus", bus_name)
                except Exception:
                    pass

        return sub

    # Validate carrier semantics before returning (buses ≠ process carriers; links == process carriers)
    def validate_network_carriers(n: pypsa.Network):
        """Validate that buses are commodity carriers and links are process carriers.

        Raises ValueError on semantic violations to prevent accidental upstream pricing.
        """
        # Define expected process carriers (conversion technologies)
        process_carriers = set(
            [
                "electrolysis",
                "direct_reduction_furnace",
                "electric_arc_furnace",
            ]
        )

        # Buses must not use process carriers
        invalid_buses = []
        for bus_name, row in n.buses.iterrows():
            carrier = row.get("carrier")
            if carrier in process_carriers:
                invalid_buses.append((bus_name, carrier))

        if invalid_buses:
            msgs = ", ".join([f"{b}({c})" for b, c in invalid_buses])
            raise ValueError(
                f"Invalid bus carriers found (process carriers on buses): {msgs}"
            )

        # Links should use process carriers; flag links that look like conversions but use commodity carriers.
        # Exclude storage-related links (charge/discharge) which legitimately use commodity carriers.
        storage_link_keywords = ("charge", "discharge", "storage")
        invalid_links = []
        for link_name, row in n.links.iterrows():
            carrier = row.get("carrier")
            # Skip storage-related links (e.g., batt_charge, batt_discharge)
            if any(kw in link_name.lower() for kw in storage_link_keywords):
                continue
            if carrier not in process_carriers:
                # A link that looks like a conversion should be a process carrier.
                # We conservatively flag any link that has multiple buses (bus0 and bus1) and isn't a process.
                n_buses = 0
                for bcol in ("bus0", "bus1", "bus2", "bus3"):
                    if bcol in row and not pd.isna(row.get(bcol)):
                        n_buses += 1
                if n_buses >= 2:
                    invalid_links.append((link_name, carrier))

        if invalid_links:
            msgs = ", ".join([f"{link}({carrier})" for link, carrier in invalid_links])
            raise ValueError(
                f"Invalid link carriers found (conversion links missing process carriers): {msgs}"
            )

    try:
        validate_network_carriers(network)
    except Exception as exc:
        logger.error(f"Carrier validation failed: {exc}")
        raise

    return network, audit_info


def add_labour_cost(n, labour_cost):

    logger.info("adding labour cost")

    carrier_labour_cost_dict = {
        "electrolysis": "ely_intensity in h/kW_ely",
        "direct_reduction_furnace": "dri_intensity in h/t_dri",
        "electric_arc_furnace": "eaf_intensity in h/t_steel",
    }

    regional_labour_cost = labour_cost.loc[snakemake.wildcards.region]
    wage = regional_labour_cost["steelworker_wage in euro/h"]

    for carrier in carrier_labour_cost_dict.keys():
        mask = n.links.carrier == carrier
        if mask.any():
            intensity = regional_labour_cost[carrier_labour_cost_dict[carrier]]

            if carrier == "electrolysis":
                fom_cost = wage * intensity * 1000
                n.links.loc[mask, "fom_cost"] += fom_cost
                logger.info(
                    f"Added labour cost as fom_cost to {carrier} links: {wage} €/h * {intensity} h/kW_ely * 1000 = {fom_cost:.2f} €/MW"
                )

            if carrier == "direct_reduction_furnace":
                marginal_cost = wage * intensity * n.links.loc[mask, "efficiency"]
                n.links.loc[mask, "marginal_cost"] += marginal_cost
                logger.info(
                    f"Added labour cost as marginal_cost to {carrier} links: {wage} €/h * {intensity} h/t_dri * efficiency"
                )

            if carrier == "electric_arc_furnace":
                marginal_cost = wage * intensity * n.links.loc[mask, "efficiency"]
                n.links.loc[mask, "marginal_cost"] += marginal_cost
                logger.info(
                    f"Added labour cost as marginal_cost to {carrier} links: {wage} €/h * {intensity} h/t_steel * efficiency"
                )
        else:
            logger.info(
                f"carrier {carrier} not in network, skipping labour cost addition for this carrier"
            )
    return n


# ============================================================================
# SNAKEMAKE INTEGRATION
# ============================================================================

if __name__ == "__main__":
    if snakemake is None:
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_regional_network",
            cost_year="2050",
            region="South_America",
            product="hbi",
            scenario="reserved",
            wacc="regional",
        )

    # Check if running from Snakemake
    if snakemake is not None:
        # Snakemake inputs/outputs
        skeleton_path = snakemake.input.skeleton
        renewables_path = snakemake.input.renewables
        tech_costs_path = snakemake.input.tech_costs
        local_demand_path = snakemake.input.local_demand

        region = snakemake.params.region
        product = snakemake.params.product
        route_label = (
            snakemake.params.route_label
            if hasattr(snakemake.params, "route_label")
            else None
        )  # route_label is the process stage name (e.g., "hbi", "steel")
        cost_year = (
            snakemake.wildcards.cost_year
            if hasattr(snakemake.wildcards, "cost_year")
            else 2050
        )
        scenario = (
            snakemake.wildcards.scenario
            if hasattr(snakemake.wildcards, "scenario")
            else "reserved"
        )

        output_path = snakemake.output[0]

        # Load config (if available)
        config_dict = snakemake.config if snakemake is not None else {}

    # Prepare network
    network, audit = prepare_network(
        skeleton_network_path=skeleton_path,
        clustered_renewables_path=renewables_path,
        tech_costs_path=tech_costs_path,
        local_demand_path=local_demand_path,
        region=region,
        product=product,
        cost_year=cost_year,
        config=config_dict,
        scenario=scenario,
        route_label=route_label,
    )

    # Add labour cost
    if snakemake.config["trade_chains"]["labour_cost"]:
        logger.info("Adding labour costs to network")
        # Load labour cost
        labour_cost = pd.read_csv(snakemake.input.labour_cost, header=0, index_col=0)
        network = add_labour_cost(network, labour_cost)

    elif not snakemake.config["trade_chains"]["labour_cost"]:
        logger.info("Labour cost addition skipped (labour_cost is False)")

    else:
        raise ValueError(
            f"Unrecognized labour_cost wildcard: {snakemake.config['trade_chains']['labour_cost']}. "
            f"Expected 'True' or 'False'."
        )

    # Save network
    logger.info(f"Saving network to {output_path}")
    network.export_to_netcdf(output_path)

    logger.info("Network preparation complete")
