"""
Calculate regional Levelized Cost of X (LCOX) for multiple demand factors.

Optimized workflow (single network load, multiple constraint applications):
  1. Load base_network ONCE (renewables + product already configured)
  2. Load product-specific demands
  3. For each demand_factor in config:
     a. Copy network (in memory)
     b. Apply renewable constraint (demand_factor-specific)
     c. Add final loads
     d. Solve optimization
     e. Extract LCOX and save results_{demand_factor}.csv
     f. Export solved network_{demand_factor}.nc

Efficiency: Load base_network once, loop through constraints (not N separate calls)

Inputs (from Snakemake):
  - base_network: PyPSA network with renewables, prepared per region (netCDF)
  - steel_demand: Regional steel demand [Mt/year] (CSV)
  - local_demand: Regional local electricity demand [TWh/year] (CSV)

Outputs (generated for each demand_factor):
  - results_{demand_factor}.csv: LCOX point
  - network_{demand_factor}.nc: Optimized network
"""

import logging
from pathlib import Path
import pypsa
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# NETWORK FIX (pandas/xarray compatibility)
# ============================================================================


def load_network_with_string_fix(filepath):
    """Load PyPSA network and convert ArrowStringArray to object dtype.

    PyPSA 1.1.2 + xarray 2026.2.0 conflict: Force object dtype on string columns.
    """
    network = pypsa.Network(filepath)
    network.name = f"Loaded-{Path(filepath).stem}"

    # Convert ArrowStringArray to object dtype for all component dataframes
    for component_name in [
        "buses",
        "generators",
        "links",
        "stores",
        "lines",
        "transformers",
    ]:
        df = getattr(network, component_name, None)
        if df is not None and not df.empty:
            for col in df.select_dtypes(include=["string"]).columns:
                df[col] = df[col].astype("object")
            if hasattr(df.index, "dtype") and df.index.dtype.name == "string":
                df.index = df.index.astype("object")

    return network


# ============================================================================
# DEMAND LOADING
# ============================================================================


def load_demands_for_region(region, product, config):
    """Load product-specific demands for region.

    Returns dict with:
      - steel_demand_mt: Mt/year
      - steel_demand_mwh_per_h: MWh/h average
      - local_el_demand_mwh: TWh/year (if available for constraint calculation)
    """
    # Load steel production demand (if applicable for this product)
    if product in ["steel", "hbi", "eaf", "eaf-grid"]:
        try:
            steel_df = pd.read_csv(snakemake.input.steel_demand)
            # Find row matching region (case-insensitive)
            region_mask = steel_df["region"].str.lower() == region.lower()
            if not region_mask.any():
                raise ValueError(f"Region '{region}' not found in steel demand data")
            steel_demand_mt = steel_df[region_mask]["SteelProductionMt"].values[0]
        except Exception as e:
            logger.warning(f"Could not load steel demand for {region}: {e}")
            steel_demand_mt = 0
    else:
        steel_demand_mt = 0

    # Load local electricity demand (for renewable constraint calculation)
    try:
        local_df = pd.read_csv(snakemake.input.local_demand)
        region_mask = local_df["region"].str.lower() == region.lower()
        if region_mask.any():
            local_el_demand_mwh = local_df[region_mask]["demand"].values[
                0
            ]  # Already in MWh
        else:
            logger.warning(f"Region '{region}' not found in local demand data")
            local_el_demand_mwh = 0
    except Exception as e:
        logger.warning(f"Could not load local demand for {region}: {e}")
        local_el_demand_mwh = 0

    # Convert Mt/year to MWh/h (using regional electricity requirement)
    # For steel: 5.25 MWh/t (from config)
    electricity_per_steel_t = config.get("electricity_steel_ratio", 5.25)
    steel_demand_mwh = steel_demand_mt * electricity_per_steel_t
    steel_demand_mwh_per_h = steel_demand_mwh / 8760  # Annual → hourly average

    return {
        "steel_demand_mt": steel_demand_mt,
        "steel_demand_mwh_per_h": steel_demand_mwh_per_h,
        "local_el_demand_mwh": local_el_demand_mwh,
    }


# ============================================================================
# RENEWABLE CONSTRAINT
# ============================================================================


def apply_renewable_constraint(network, demand_factor, local_el_demand_mwh, config):
    """Constrain electrolyzer capacity based on demand_factor.

    NEW: Prioritize highest-CF renewables for local demand (least waste).

    Logic:
      1. Get all renewable generators with their timestep-averaged CF
      2. Sort by average CF (descending) - highest first
      3. Accumulate capacity from highest CF until >= local_demand
      4. Block these generators for local demand (set p_nom_max=0)
      5. Calculate remaining capacity available for steel
      6. Apply demand_factor to remaining capacity
      7. Constrain electrolyzer p_nom_max to this value

    Returns: audit dict with capacity breakdown and blocked generators
    """
    # Get all renewable generators (identified by name pattern "renewable_*")
    # All generators produce electricity, so we filter by naming convention
    renewable_gens = network.generators[
        network.generators.index.str.startswith("renewable_")
    ]

    if renewable_gens.empty:
        logger.warning("No renewable generators found in network")
        return {
            "total_renewable_capacity_mw": 0,
            "capacity_for_local_demand_mw": 0,
            "capacity_available_for_steel_mw": 0,
            "generators_blocked_for_local_demand": [],
        }

    # ====== STEP 1: Calculate average CF for each generator ======
    gen_cf_data = []

    for gen_name, gen_row in renewable_gens.iterrows():
        h_max_pu = gen_row["p_max_pu"]  # Hourly timeseries (0-1) or scalar

        # Handle both pandas Series, numpy array, and scalars
        if isinstance(h_max_pu, (int, float, np.number)):
            # Scalar CF - use directly
            avg_cf = float(h_max_pu)
        elif hasattr(h_max_pu, "values"):
            # Pandas Series
            cf_values = h_max_pu.values
            avg_cf = np.mean(cf_values) if len(cf_values) > 0 else 0
        else:
            # Numpy array or list
            cf_values = h_max_pu
            avg_cf = (
                np.mean(cf_values)
                if isinstance(cf_values, np.ndarray) and len(cf_values) > 0
                else float(cf_values)
            )
        p_nom_max = gen_row["p_nom_max"]

        gen_cf_data.append(
            {
                "gen_name": gen_name,
                "avg_cf": avg_cf,
                "p_nom_max": p_nom_max,
                "carrier": gen_row["carrier"],
            }
        )

    logger.debug(f"Found {len(gen_cf_data)} renewable generators")

    # ====== STEP 2: Sort by average CF (descending) - prioritize best ======
    gen_cf_data.sort(key=lambda x: x["avg_cf"], reverse=True)

    total_renewable_capacity = sum([g["p_nom_max"] for g in gen_cf_data])
    logger.info(f"Total renewable capacity: {total_renewable_capacity:.1f} MW")
    logger.info(
        f"Top 3 generators by CF: {[(g['gen_name'], format(g['avg_cf'], '.3f')) for g in gen_cf_data[:3]]}"
    )

    # ====== STEP 3: Block highest-CF generators for local demand ======
    # Convert annual local demand to hourly average [MWh/year] → [MW]
    local_el_demand_mwh / 8760  # Hourly average power needed

    capacity_accumulated = 0  # Track cumulative capacity factor contribution
    generators_for_local = []

    for gen_info in gen_cf_data:
        if capacity_accumulated >= local_el_demand_mwh:
            # We've accumulated enough to serve local demand, stop
            break

        gen_name = gen_info["gen_name"]
        avg_cf = gen_info["avg_cf"]
        p_nom_max = gen_info["p_nom_max"]

        # How much energy does this generator produce annually?
        annual_energy = avg_cf * p_nom_max * 8760  # MWh/year

        # How much do we still need?
        remaining_needed = local_el_demand_mwh - capacity_accumulated

        if annual_energy <= remaining_needed:
            # Use entire generator for local demand
            capacity_to_use = p_nom_max
            capacity_accumulated += annual_energy
        else:
            # Use partial generator to exactly meet local demand
            capacity_to_use = remaining_needed / (avg_cf * 8760)
            capacity_accumulated += remaining_needed

        generators_for_local.append(
            {
                "gen_name": gen_name,
                "avg_cf": avg_cf,
                "capacity_blocked_mw": capacity_to_use,
                "energy_provided_mwh": capacity_to_use * avg_cf * 8760,
            }
        )

        # Block this generator: set p_nom_max=0 so optimizer can't use for steel
        network.generators.at[gen_name, "p_nom_max"] = 0
        logger.info(
            f"  Blocked {gen_name:40s} (CF={avg_cf:.3f}, {capacity_to_use:7.1f} MW) → local demand"
        )

    logger.info(
        f"Allocated {len(generators_for_local)} generators for local demand ({capacity_accumulated:.0f} MWh/year)"
    )

    # ====== STEP 4: Calculate remaining renewable capacity (for steel) ======
    # Get remaining generators that are NOT blocked (p_nom_max > 0)
    remaining_renewable_gens = network.generators[
        (network.generators.index.str.startswith("renewable_"))
        & (network.generators["p_nom_max"] > 0)
    ]
    total_remaining_capacity = remaining_renewable_gens["p_nom_max"].sum()

    logger.info(
        f"Remaining renewable capacity for steel: {total_remaining_capacity:.1f} MW"
    )

    # ====== STEP 5: Apply demand_factor constraint ======
    available_for_steel = total_remaining_capacity * (demand_factor / 100.0)

    # ====== STEP 6: Constrain electrolyzer ======
    if "electrolyzer" in network.links.index:
        network.links.at["electrolyzer", "p_nom_max"] = available_for_steel
        logger.info(
            f"Constrained electrolyzer p_nom_max to {available_for_steel:.1f} MW "
            f"(demand_factor={demand_factor}%)"
        )
    else:
        logger.warning("Electrolyzer not found in network links")

    return {
        "total_renewable_capacity_mw": total_renewable_capacity,
        "capacity_for_local_demand_mw": capacity_accumulated
        / 8760,  # Convert back to MW
        "capacity_available_for_steel_mw": available_for_steel,
        "generators_blocked_for_local_demand": [
            g["gen_name"] for g in generators_for_local
        ],
        "num_generators_blocked": len(generators_for_local),
    }


# ============================================================================
# LOAD ADDITION
# ============================================================================


def add_loads_to_network(network, product, demands):
    """Store cumulative annual demand for flexible constraint injection.

    Rather than adding hourly Load components (which force rigid patterns),
    we store annual demand and inject it as a constraint during solve().
    This allows the network to decide flexibly WHEN to produce.
    """

    if product == "steel":
        bus_name = "steel"
        annual_steel_t = demands["steel_demand_mt"] * 1000
        annual_demand_units = annual_steel_t
        unit_str = "t"

    elif product == "hbi":
        bus_name = "hbi"
        annual_hbi_t = demands["steel_demand_mt"] * 1000
        annual_demand_units = annual_hbi_t
        unit_str = "t"

    elif product == "h2":
        bus_name = "hydrogen"
        annual_demand_units = demands["steel_demand_mwh_per_h"] * 8760
        unit_str = "MWh"

    elif product in ["eaf", "eaf-grid"]:
        bus_name = "steel"
        annual_steel_t = demands["steel_demand_mt"] * 1000
        annual_demand_units = annual_steel_t
        unit_str = "t"

    else:
        raise ValueError(f"Product '{product}' not recognized")

    if bus_name not in network.buses.index:
        raise ValueError(f"Bus '{bus_name}' not found in network")

    # Store as network parameter for constraint injection in solve()
    network.annual_demand = {
        "product": product,
        "bus": bus_name,
        "total_units": annual_demand_units,
        "unit": unit_str,
    }

    logger.info(
        f"Stored cumulative {product} demand: {annual_demand_units:.1f} {unit_str}/year (flexible timing)"
    )


# ============================================================================
# SOLVER
# ============================================================================


def solve_network(network, config):
    """Solve the PyPSA optimization with cumulative annual demand constraint.

    The constraint enforces: sum of production over all hours >= annual_demand
    This allows the network to decide flexibly WHEN to produce (not fixed hourly).
    """
    solver_name = config.get("solver", {}).get("name", "glpk")
    solver_options = config.get("solver_options", {}).get(
        config.get("solver", {}).get("options", "default"), {}
    )

    logger.info(f"Solving network with {solver_name}...")

    # Define constraint injection function (called after model building)
    def add_annual_constraint(network, snapshots):
        if not hasattr(network, "annual_demand"):
            return

        demand_info = network.annual_demand
        bus_name = demand_info["bus"]
        annual_demand = demand_info["total_units"]

        # Find generators connected to demand bus
        gens_on_bus = network.generators[network.generators["bus"] == bus_name].index

        if len(gens_on_bus) == 0:
            logger.warning(f"No generators on bus '{bus_name}' for annual constraint")
            return

        # Access PyPSA's linopy model variables
        p_var = network.model["Generator-p"]  # Shape: (snapshot, generator)

        # Sum generator output over all snapshots
        total_output = p_var.loc[:, gens_on_bus].sum()

        # Add constraint: total output >= annual demand
        network.model.add_constraints(
            total_output >= annual_demand, name=f"AnnualDemand_{bus_name}"
        )

        logger.info(
            f"✓ Constraint added: {bus_name} annual output >= {annual_demand:.0f} {demand_info['unit']}"
        )

    # Solve with constraint injection
    network.optimize(
        network.snapshots,
        solver_name=solver_name,
        solver_options=solver_options,
        multi_investment_periods=False,
        extra_functionality=add_annual_constraint,
    )

    logger.info("Network solved successfully")

    return network


# ============================================================================
# RESULTS EXTRACTION
# ============================================================================


def extract_lcox(network, product, demands, renewable_constraint_info, demand_factor):
    """Extract LCOX from optimized network.

    Returns DataFrame with one row containing all results + audit columns.
    """
    results_df = pd.DataFrame(
        columns=[
            "demand_factor [%]",
            "demand [{}]".format("t" if product != "h2" else "MWh"),
            "load [per h]",
            "cost [EUR]",
            "lcox [EUR/unit]",
            "renewable_capacity_total_mw",
            "renewable_capacity_for_local_mw",
            "renewable_capacity_for_steel_mw",
            "num_generators_blocked",
            "status",
        ]
    )

    try:
        obj_value = network.objective
        if obj_value is None or np.isnan(obj_value):
            raise ValueError("Optimization failed to return valid objective")

        demand_annual = (
            demands["steel_demand_mt"]
            if product != "h2"
            else demands["steel_demand_mwh_per_h"] * 8760
        )
        lcox = obj_value / demand_annual if demand_annual > 0 else np.inf

        results_df.loc[0] = [
            int(demand_factor * 100),  # Convert to percent
            demand_annual,
            demands["steel_demand_mwh_per_h"],
            obj_value,
            lcox,
            renewable_constraint_info["total_renewable_capacity_mw"],
            renewable_constraint_info["capacity_for_local_demand_mw"],
            renewable_constraint_info["capacity_available_for_steel_mw"],
            renewable_constraint_info.get("num_generators_blocked", 0),
            "feasible",
        ]
        logger.info(f"LCOX calculated: {lcox:.2f} EUR/unit")
        logger.info(
            f"  Generators blocked for local demand: {renewable_constraint_info.get('num_generators_blocked', 0)}"
        )

    except Exception as e:
        logger.error(f"Optimization infeasible or failed: {e}")
        demand_annual = (
            demands["steel_demand_mt"]
            if product != "h2"
            else demands["steel_demand_mwh_per_h"] * 8760
        )
        results_df.loc[0] = [
            int(demand_factor * 100),  # Convert to percent
            demand_annual,
            demands["steel_demand_mwh_per_h"],
            np.nan,
            np.nan,
            renewable_constraint_info["total_renewable_capacity_mw"],
            renewable_constraint_info["capacity_for_local_demand_mw"],
            renewable_constraint_info.get("capacity_available_for_steel_mw", 0),
            renewable_constraint_info.get("num_generators_blocked", 0),
            "infeasible",
        ]

    return results_df


# ============================================================================
# ADJUSTMENT (LEGACY - KEPT FOR COMPATIBILITY)
# ============================================================================


def adjust_part_load(network, config):
    """Adjust part-load limits for links based on config."""
    part_load = config.get("part_load", {})
    if not part_load:
        return

    for carrier, min_pu in part_load.items():
        mask = network.links["carrier"] == carrier
        if mask.any():
            network.links.loc[mask, "p_min_pu"] = min_pu
            logger.debug(f"Set part-load limit for {carrier}: p_min_pu={min_pu}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "calculate_regional_lcox",
            cost_year="2030",
            region="Europe",
            product="steel",
        )

    # ==================== SETUP ====================
    logger.info("=" * 70)
    logger.info(
        f"LCOX Calculation: region={snakemake.wildcards.region}, "
        f"product={snakemake.wildcards.product}"
    )
    logger.info("=" * 70)

    # Load pre-prepared base network ONCE (key efficiency gain)
    logger.info("Loading base network...")
    base_network = load_network_with_string_fix(snakemake.input.base_network)
    logger.info(
        f"Network loaded: {len(base_network.buses)} buses, "
        f"{len(base_network.generators)} generators, {len(base_network.links)} links"
    )

    # Load demands for this region and product
    logger.info("Loading demands...")
    demands = load_demands_for_region(
        region=snakemake.wildcards.region,
        product=snakemake.wildcards.product,
        config=snakemake.config,
    )
    logger.info(f"Steel demand: {demands['steel_demand_mt']:.1f} Mt/year")
    logger.info(
        f"Local electricity demand: {demands['local_el_demand_mwh']:.1f} MWh/year"
    )

    # ==================== SINGLE DEMAND FACTOR PROCESSING ====================
    # Get demand_factor from Snakemake wildcard (in percent: 1, 10, 50, etc.)
    demand_factor_percent = int(snakemake.wildcards.demand_factor)
    demand_factor = (
        demand_factor_percent / 100.0
    )  # Convert to decimal (0.01, 0.1, 0.5, etc.)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Processing demand_factor={demand_factor_percent}% ({demand_factor})")
    logger.info(f"{'=' * 70}")

    # Create a copy of base network for this constraint scenario
    network = base_network.copy()
    network.name = f"LCOX-{snakemake.wildcards.region}-{snakemake.wildcards.product}-DF{demand_factor_percent}%"

    # Preserve discount_rate from base network (needed for cost annuitization)
    network.discount_rate = base_network.discount_rate

    # Apply renewable constraint based on demand_factor
    logger.info(
        f"Applying renewable constraint (demand_factor={demand_factor_percent}%)..."
    )
    constraint_info = apply_renewable_constraint(
        network=network,
        demand_factor=demand_factor,
        local_el_demand_mwh=demands["local_el_demand_mwh"],
        config=snakemake.config,
    )

    logger.info(
        f"  Total renewable capacity: {constraint_info['total_renewable_capacity_mw']:.1f} MW"
    )
    logger.info(
        f"  Capacity for local demand: {constraint_info['capacity_for_local_demand_mw']:.1f} MW"
    )
    logger.info(
        f"  Capacity available for steel: {constraint_info['capacity_available_for_steel_mw']:.1f} MW"
    )

    # Add loads to network
    logger.info("Adding loads to network...")
    add_loads_to_network(
        network=network, product=snakemake.wildcards.product, demands=demands
    )

    # Adjust part-load (if configured)
    if snakemake.config.get("part_load"):
        adjust_part_load(network, snakemake.config)

    # Solve this constraint scenario
    solve_network(network, snakemake.config)

    # Extract LCOX results
    logger.info("Extracting results...")
    results_df = extract_lcox(
        network=network,
        product=snakemake.wildcards.product,
        demands=demands,
        renewable_constraint_info=constraint_info,
        demand_factor=demand_factor,
    )

    # ==================== SAVE RESULTS ====================
    logger.info("=" * 70)
    logger.info("Saving results...")
    logger.info("=" * 70)

    # Save CSV result
    results_df.to_csv(snakemake.output.results, index=False)
    logger.info(f"Results saved: {snakemake.output.results}")

    # Save network
    network.export_to_netcdf(snakemake.output.network)
    logger.info(f"Network saved: {snakemake.output.network}")

    logger.info("=" * 70)
    logger.info(f"LCOX calculation complete for demand_factor={demand_factor_percent}%")
    logger.info("=" * 70)
