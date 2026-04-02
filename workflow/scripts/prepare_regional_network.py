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
# RENEWABLE CLUSTER LOADING
# ============================================================================


def load_renewable_clusters_for_region(
    region: str, renewable_timeseries_path: str, config: dict
) -> Dict:
    """Load renewable clusters for region directly from netCDF (includes all metadata + timeseries)."""
    # Step 1: Get ISO3 codes for region
    iso3_list = region_to_iso3_codes(region, config)
    logger.info(f"Region '{region}' maps to ISO3 codes: {iso3_list}")

    # Step 2: Load netCDF dataset
    timeseries_ds = xr.open_dataset(renewable_timeseries_path)

    # Filter clusters by ISO3 country codes
    cluster_iso3 = timeseries_ds.coords["iso3"].values  # ISO3 per cluster
    cluster_ids = timeseries_ds.coords["cluster"].values  # Cluster IDs

    # Select clusters for this region's ISO3 codes
    cluster_mask = np.isin(cluster_iso3, iso3_list)
    selected_clusters = cluster_ids[cluster_mask]

    if len(selected_clusters) == 0:
        raise ValueError(
            f"No clusters found for region '{region}' with ISO3 {iso3_list}. "
            f"Available ISO3 in data: {np.unique(cluster_iso3)}"
        )

    logger.info(f"Found {len(selected_clusters)} clusters for region {region}")
    logger.info(f"Unique technologies: {timeseries_ds.technology.values}")

    # Step 3: Extract cluster data for each selected cluster
    clusters = []
    total_potential_mw = 0

    for cluster_id in selected_clusters:
        # Get data for this cluster across all technologies
        cluster_idx = list(cluster_ids).index(cluster_id)
        iso3 = cluster_iso3[cluster_idx]

        for tech in timeseries_ds.technology.values:
            # Extract 2D arrays from dataset
            renewable_potential = float(
                timeseries_ds["renewable_potential"]
                .sel(cluster=cluster_id, technology=tech)
                .values
            )
            p_nom_max = float(
                timeseries_ds["p_nom_max"]
                .sel(cluster=cluster_id, technology=tech)
                .values
            )
            avg_cf = float(
                timeseries_ds["avg_cf"].sel(cluster=cluster_id, technology=tech).values
            )
            cf_timeseries = (
                timeseries_ds["capacity_factor"]
                .sel(cluster=cluster_id, technology=tech)
                .values
            )

            # Handle NaN/missing values - default to 0
            renewable_potential = (
                0 if np.isnan(renewable_potential) else renewable_potential
            )
            p_nom_max = 0 if np.isnan(p_nom_max) else p_nom_max
            avg_cf = 0 if np.isnan(avg_cf) else avg_cf
            if isinstance(cf_timeseries, np.ndarray):
                cf_timeseries = np.nan_to_num(cf_timeseries, nan=0.0)
            else:
                cf_timeseries = (
                    np.zeros(8760) if np.isnan(cf_timeseries) else cf_timeseries
                )

            # Get geographic coordinates
            lat = float(timeseries_ds["lat"].sel(cluster=cluster_id).values)
            lon = float(timeseries_ds["lon"].sel(cluster=cluster_id).values)

            clusters.append(
                {
                    "cluster_id": f"{cluster_id}_{tech}",  # Unique ID combining cluster + technology
                    "iso3": iso3,
                    "technology": tech,
                    "renewable_potential_mw": renewable_potential,
                    "p_nom_max": p_nom_max,
                    "avg_cf": avg_cf,
                    "cf_timeseries": cf_timeseries,
                    "lat": lat,
                    "lon": lon,
                }
            )

            total_potential_mw += renewable_potential

    # Aggregate results
    iso3_codes = sorted(list(set(c["iso3"] for c in clusters)))
    technologies = sorted(list(set(c["technology"] for c in clusters)))

    logger.info(f"Total renewable potential for region: {total_potential_mw:.1f} MW")
    logger.info(f"Clusters loaded: {len(clusters)}")

    return {
        "clusters": clusters,
        "iso3_list": iso3_codes,
        "technologies": technologies,
        "total_potential_mw": total_potential_mw,
    }


# ============================================================================
# RENEWABLE GENERATOR ADDITION
# ============================================================================


def add_renewable_generators(
    network: pypsa.Network,
    renewable_clusters: Dict,
    tech_costs: pd.Series,
    config: dict,
) -> None:
    """Add renewable generators to electricity bus for all clusters.

    Renewable generators (wind, solar) produce electricity, so all have carrier="electricity".
    The technology type (onwind, offwind, solar) is tracked in the generator name.
    """
    clusters = renewable_clusters["clusters"]

    if not clusters:
        logger.warning("No clusters provided; no generators added")
        return

    # Map technology names to database keys for cost lookup
    tech_database_map = {
        "onwind": "onwind",
        "offwind-ac": "offwind",
        "solar": "solar-utility",
    }

    # Ensure electricity carrier is defined (all renewables produce electricity)
    if "electricity" not in network.carriers.index:
        network.add("Carrier", "electricity")

    # Use the network's discount_rate (which is set regionally in prepare_network)
    discount_rate = network.discount_rate

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        technology = cluster["technology"]
        p_nom_max = cluster["p_nom_max"]  # Use cluster-aggregated p_nom_max
        cf_ts = cluster["cf_timeseries"]

        # Get technology parameters from database (handles missing tech gracefully)
        db_tech_name = tech_database_map.get(technology, technology)
        tech_params = td.get_tech(tech_costs, db_tech_name)

        overnight_cost = (
            td.get_tech_param(tech_params, "investment", 0) * 1000
        )  # EUR/kW → EUR/MW
        lifetime = td.get_tech_param(tech_params, "lifetime", 20)
        fom_pct = td.get_tech_param(tech_params, "FOM", 0)
        fom_cost = overnight_cost * (fom_pct / 100) if overnight_cost > 0 else 0

        gen_name = f"renewable_{cluster_id}"

        # Add generator with cluster data
        # All renewables produce electricity (carrier="electricity")
        # Technology type (onwind, offwind, solar) is encoded in the generator name
        network.add(
            "Generator",
            gen_name,
            bus="electricity",
            carrier="electricity",
            p_nom_extendable=True,
            p_nom=0,  # Start with no capacity; optimization will decide
            p_nom_max=p_nom_max,  # Upper ceiling from cluster data (MW)
            p_max_pu=cf_ts,  # Hourly capacity factor from cluster data (0-1)
            overnight_cost=overnight_cost,
            discount_rate=discount_rate,
            lifetime=lifetime,
            fom_cost=fom_cost,
        )

        logger.debug(
            f"Added generator {gen_name}: p_nom_max={p_nom_max:.1f} MW, "
            f"overnight_cost={overnight_cost:.1f} EUR/MW"
        )

    logger.info(f"Added {len(clusters)} renewable generators to network")


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
    renewable_timeseries_path: str,
    tech_costs_path: str,
    region: str,
    product: str,
    config: dict,
) -> Tuple[pypsa.Network, Dict]:
    """Prepare network: add renewables, apply product cutoff. Returns (network, audit_info)."""
    logger.info("=" * 70)
    logger.info(f"Preparing network for region={region}, product={product}")
    logger.info("=" * 70)

    # Step 1: Load skeleton network
    logger.info("Loading skeleton network...")
    network = pypsa.Network(skeleton_network_path)
    network.name = f"Skeleton-{region}-{product}"
    logger.info(
        f"Skeleton loaded: {len(network.buses)} buses, "
        f"{len(network.links)} links, {len(network.stores)} stores"
    )

    # Set snapshots here using wildcard year coming from Snakemake
    cost_year = None
    if "snakemake" in globals():
        cost_year = getattr(snakemake.wildcards, "cost_year", None)

    if cost_year is not None:
        network.set_snapshots(
            pd.date_range(f"{cost_year}-01-01", periods=8760, freq="h")
        )
        logger.info(f"Set snapshots for cost_year={cost_year} in prepare_network")
    else:
        raise ValueError(
            "cost_year must be defined in snakemake wildcards for prepare_network"
        )

    # Step 1b: Set interest rate (discount rate) for the network
    interest_rates = config.get("interest_rate", {})
    discount_rate = interest_rates.get(region, interest_rates.get("default", 0.07))
    network.discount_rate = discount_rate
    logger.info(f"Set discount rate: {discount_rate:.4f} for region {region}")

    # Step 2: Load tech costs
    logger.info("Loading technology costs...")
    tech_costs = td.load_tech_costs(tech_costs_path)

    # Step 3: Load renewable clusters for region
    logger.info(f"Loading renewable clusters for region {region}...")
    renewable_clusters = load_renewable_clusters_for_region(
        region=region,
        renewable_timeseries_path=renewable_timeseries_path,
        config=config,
    )

    # Step 4: Add renewable generators
    logger.info("Adding renewable generators to network...")
    add_renewable_generators(
        network=network,
        renewable_clusters=renewable_clusters,
        tech_costs=tech_costs,
        config=config,
    )

    # Step 5: Apply product-specific cutoff
    logger.info(f"Applying product cutoff for {product}...")
    apply_product_cutoff(network=network, product=product)

    # Step 6: Build audit info
    # Count unique geographic cluster IDs (without technology suffix)
    unique_geographic_clusters = set()
    for cluster in renewable_clusters["clusters"]:
        # Extract base cluster ID (without technology)
        cluster_id = cluster["cluster_id"]
        base_cluster = "_".join(cluster_id.split("_")[:-1])  # Remove tech suffix
        unique_geographic_clusters.add(base_cluster)

    audit_info = {
        "region": region,
        "product": product,
        "discount_rate": discount_rate,
        "iso3_list": renewable_clusters["iso3_list"],
        "num_geographic_clusters": len(unique_geographic_clusters),
        "num_technologies": len(renewable_clusters["technologies"]),
        "total_renewable_potential_mw": renewable_clusters["total_potential_mw"],
        "network_stats": {
            "num_buses": len(network.buses),
            "num_links": len(network.links),
            "num_generators": len(network.generators),
            "num_stores": len(network.stores),
        },
    }

    logger.info("Network preparation complete:")
    logger.info(f"  - Region: {region} (ISO3: {audit_info['iso3_list']})")
    logger.info(f"  - Geographic clusters: {audit_info['num_geographic_clusters']}")
    logger.info(f"  - Technologies per cluster: {audit_info['num_technologies']}")
    logger.info(
        f"  - Total renewable potential: {audit_info['total_renewable_potential_mw']:.1f} MW"
    )
    logger.info(
        f"  - Network: {audit_info['network_stats']['num_buses']} buses, "
        f"{audit_info['network_stats']['num_generators']} generators"
    )

    # Apply regional discount_rate to all cost-bearing components
    _apply_discount_rate_to_components(network, discount_rate)

    # Run consistency check and sanitize
    _consistency_check(network)

    logger.info("=" * 70)

    return network, audit_info


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
                    "clusters_timeseries": "../data/renewable_clusters.nc",
                    "costs": "../resources/technology_data/costs_2030.csv",
                }
                self.output = {
                    "base_network": "test_base_network.nc",
                    "audit": "test_audit.json",
                }
                self.wildcards = {
                    "region": "Test_1",
                    "product": "steel",
                    "cost_year": "2030",
                }
                self.config = {
                    "regions": {"Test_1": ["NLD"], "Test_2": ["PRT"], "Test_3": ["IRL"]}
                }

        snakemake = MockSnakemake()

    # Prepare network
    try:
        network, audit_info = prepare_network(
            skeleton_network_path=snakemake.input.skeleton,
            renewable_timeseries_path=snakemake.input.clusters_timeseries,
            tech_costs_path=snakemake.input.costs,
            region=snakemake.wildcards.region,
            product=snakemake.wildcards.product,
            config=snakemake.config,
        )

        # Save outputs
        network.export_to_netcdf(snakemake.output.base_network)
        logger.info(f"Network saved to {snakemake.output.base_network}")

        with open(snakemake.output.audit, "w") as f:
            json.dump(audit_info, f, indent=2, default=str)
        logger.info(f"Audit info saved to {snakemake.output.audit}")

    except Exception as e:
        logger.error(f"Network preparation failed: {e}", exc_info=True)
        raise
