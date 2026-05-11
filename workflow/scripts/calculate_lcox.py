"""
Calculate regional Levelized Cost of X (LCOX) for a single product demand level.

Supports: steel, hbi, h2 (flexible product support)

Workflow (single demand level per invocation):
  1. Load base_network (renewables + product already configured)
  2. Load product-specific demands
  3. Apply incremental generator selection (delete generators outside demand level set)
  4. Add hourly loads based on demand
  5. Solve optimization
  6. Extract LCOX and save results_{demand_level}.csv
  7. Export solved network_{demand_level}.nc

Parallelization: Each demand level is a separate Snakemake job, enabling parallel execution.

Inputs (from Snakemake):
  - base_network: PyPSA network with renewables, prepared per region (netCDF)
  - incremental_sets: Pre-filtered generator sets per demand level (JSON)
  - local_demand: Regional local electricity demand [MWh/year] (CSV)

Outputs (generated for each demand level):
  - results_{demand_level}.csv: LCOX point for that demand level
  - network_{demand_level}.nc: Optimized network
"""

import logging
import os
from pathlib import Path
from typing import Any
import pypsa
import pandas as pd
import numpy as np
import xarray as xr

snakemake: Any = globals().get("snakemake")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if snakemake is not None and getattr(snakemake, "log", None):
    log_path = Path(snakemake.log[0])
else:
    log_path = Path("../logs") / "calculate_lcox.log"

log_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ============================================================================
# DEMAND LOADING
# ============================================================================


def load_demands_for_region(region, config):
    """Load local electricity demand for region.

    Returns dict with:
      - local_el_demand_mwh: MWh/year (for renewable constraint calculation)
    """
    # Load local electricity demand (for renewable constraint calculation)
    try:
        local_df = pd.read_csv(snakemake.input.local_demand)
        region_mask = local_df["region"].str.lower() == region.lower()
        if region_mask.any():
            total_energy_mwh = local_df[region_mask]["demand"].values[
                0
            ]  # MWh final energy
            el_share = (
                local_df[region_mask]["el_share"].values[0] / 100
            )  # Convert % to fraction
            local_el_demand_mwh = total_energy_mwh * el_share  # Apply electricity share
        else:
            logger.warning(f"Region '{region}' not found in local demand data")
            local_el_demand_mwh = 0
    except Exception as e:
        logger.warning(f"Could not load local demand for {region}: {e}")
        local_el_demand_mwh = 0

    return {
        "local_el_demand_mwh": local_el_demand_mwh,
    }


# ============================================================================
# RENEWABLE CONSTRAINT
# ============================================================================


# ============================================================================
# INCREMENTAL RENEWABLE SELECTION (upstream filtering, Phase 3)
# ============================================================================


def apply_incremental_generator_selection(network, product_demand_mt, incremental_sets):
    """Delete renewable generators NOT in the incremental set for this demand level.

    NEW WORKFLOW (Phase 3):
    - Base network contains ALL generators from max-demand filtering
    - Incremental sets pre-computed in prepare_regional_network.py
    - For each demand level, delete generators outside that level's set

    OLD WORKFLOW (Phase 2) REMOVED:
    - apply_renewable_constraint() - blocked highest-CF for local demand
    - Now: filtering done once upstream, no per-demand redundancy

    Parameters
    ----------
    network : pypsa.Network
        Network with full renewable set (from filtering at max demand)
    product_demand_mt : float
        Current product demand level in Mt/year
    incremental_sets : dict
        Mapping: {demand_mt: [selected_generators]}
        Each generator dict has: bus_id, technology, p_nom_max, avg_cf

    Returns
    -------
    dict
        Audit info with generator deletion stats
    """
    logger.info("=" * 70)
    logger.info("APPLYING INCREMENTAL GENERATOR SELECTION")
    logger.info("=" * 70)

    # Get the selected generators for this demand level
    if product_demand_mt not in incremental_sets:
        logger.warning(
            f"Demand level {product_demand_mt} Mt not in incremental sets: {list(incremental_sets.keys())}"
        )
        return {
            "total_generators_before": len(network.generators),
            "generators_deleted": 0,
            "generators_kept": len(network.generators),
            "selected_for_demand": 0,
        }

    selected_generators = incremental_sets[product_demand_mt]

    # Build set of generator names for this demand level
    # Generator names are formatted as: renewable_{bus_id}_{technology}
    selected_gen_names = set()
    for gen_dict in selected_generators:
        gen_name = f"renewable_{gen_dict['bus_id']}_{gen_dict['technology']}"
        selected_gen_names.add(gen_name)

    logger.info(
        f"Selected generators for {product_demand_mt} Mt: {len(selected_gen_names)}"
    )

    # Get all renewable generators in network
    renewable_gens = network.generators[
        network.generators.index.str.startswith("renewable_")
    ]

    # Find generators to delete (those NOT in selected set)
    generators_to_delete = [
        gen_name
        for gen_name in renewable_gens.index
        if gen_name not in selected_gen_names
    ]

    logger.info(
        f"Deleting {len(generators_to_delete)} generators not in incremental set"
    )

    # Delete generators
    for gen_name in generators_to_delete:
        network.remove("Generator", gen_name)
        logger.debug(f"Deleted generator: {gen_name}")

    logger.info(
        f"Kept {len(selected_gen_names)} generators for demand level {product_demand_mt} Mt"
    )
    logger.info("=" * 70)

    return {
        "total_generators_before": len(renewable_gens),
        "generators_deleted": len(generators_to_delete),
        "generators_kept": len(selected_gen_names),
        "selected_for_demand": len(selected_gen_names),
    }


# ============================================================================
# LOAD ADDITION
# ============================================================================


def add_loads_to_network(network, product, demands):
    """Add hourly Load components for fixed product demand.

    Converts annual demand to hourly load: hourly_load = annual_demand / 8760
    This represents a constant hourly demand throughout the year.
    """

    if product == "steel":
        bus_name = "steel"
        # Steel is measured in t/year, convert to t/h (hourly)
        hourly_demand_t = demands["product_demand_mt"] * 1e6 / 8760  # Mt/year → t/h
        unit_str = "t/h"

    elif product == "hbi":
        bus_name = "hbi"
        # HBI is measured in t/year, convert to t/h (hourly)
        hourly_demand_t = demands["product_demand_mt"] * 1e6 / 8760  # Mt/year → t/h
        unit_str = "t/h"

    elif product == "h2":
        bus_name = "hydrogen"
        # H2 is measured in MWh/year, convert to MW (hourly average)
        hourly_demand_mwh = demands[
            "product_demand_mwh_per_h"
        ]  # Already hourly average
        unit_str = "MW"

    elif product in ["eaf", "eaf-grid"]:
        bus_name = "steel"
        # Steel is measured in t/year, convert to t/h (hourly)
        hourly_demand_t = demands["product_demand_mt"] * 1e6 / 8760  # Mt/year → t/h
        unit_str = "t/h"

    else:
        raise ValueError(f"Product '{product}' not recognized")

    if bus_name not in network.buses.index:
        raise ValueError(f"Bus '{bus_name}' not found in network")

    # Add constant hourly load to the bus
    load_name = f"{product}_demand"
    if product == "h2":
        p_set = hourly_demand_mwh
    else:
        p_set = hourly_demand_t

    network.add(
        "Load",
        load_name,
        bus=bus_name,
        p_set=p_set,  # Constant hourly demand
    )

    # For steel/HBI: set HBI storage initial energy to 24 hours of hourly load
    if product.lower() in ["steel", "hbi"]:
        if "hbi_storage" in network.stores.index:
            hbi_e_initial = 24 * hourly_demand_t  # 24 hours of buffer
            network.stores.at["hbi_storage", "e_initial"] = hbi_e_initial
            logger.info(
                f"Set HBI storage e_initial to {hbi_e_initial:.2f} t (24h buffer for {hourly_demand_t:.4f} t/h demand)"
            )

    logger.info(
        f"Added hourly load for {product}: {load_name} = {p_set:.4f} {unit_str} (constant all hours)"
    )


def inspect_network(network, product):
    """Print network structure for debugging infeasibility."""
    logger.info("\n" + "=" * 80)
    logger.info("NETWORK INSPECTION - Connectivity & Status")
    logger.info("=" * 80)

    logger.info(f"Buses ({len(network.buses)}): {list(network.buses.index)}")
    logger.info(f"\nLoads ({len(network.loads)}):")
    for load_name, load_row in network.loads.iterrows():
        logger.info(
            f"  {load_name:30s} -> bus={load_row['bus']:15s} p_set={load_row['p_set']:.1f}"
        )

    logger.info(f"\nLinks ({len(network.links)}):")
    for link_name, link_row in network.links.iterrows():
        logger.info(
            f"  {link_name:15s}: {link_row['bus0']:12s} -> {link_row['bus1']:12s}  p_nom_ext={link_row['p_nom_extendable']} p_nom_max={link_row['p_nom_max']:.0e}"
        )

    logger.info(f"\nStores ({len(network.stores)}):")
    for store_name, store_row in network.stores.iterrows():
        logger.info(f"  {store_name:20s} -> {store_row['bus']:15s}")

    # Check isolated buses
    all_buses = set(network.buses.index)
    connected = (
        set(network.generators["bus"].unique())
        | set(network.links["bus0"].unique())
        | set(network.links["bus1"].unique())
        | set(network.loads["bus"].unique())
        | set(network.stores["bus"].unique())
    )
    isolated = all_buses - connected
    if isolated:
        logger.warning(f"⚠ Isolated buses: {isolated}")
    logger.info("=" * 80 + "\n")


def _convert_arrow_strings(network):
    """Convert ArrowStringArray columns/indices to regular object dtype.

    Workaround for PyPSA incompatibility with pandas ArrowStringArray.
    Uses PyPSA's component structure to properly access all dataframes.
    Based on: https://github.com/PyPSA/PyPSA/issues/1585
    """
    for c in network.components:
        df = c.static
        if not df.empty:
            # Convert index if it's ArrowStringArray
            if isinstance(df.index.values, pd.arrays.ArrowStringArray):
                c.static.index = pd.Index(df.index.astype(object))
            # Convert columns if they're ArrowStringArray
            for col in df.columns:
                if isinstance(df[col].values, pd.arrays.ArrowStringArray):
                    c.static[col] = df[col].astype(object)
        # Convert time-varying data
        for key in c.dynamic:
            dyn_df = c.dynamic[key]
            if isinstance(dyn_df, pd.DataFrame) and not dyn_df.empty:
                # Convert column index if it's ArrowStringArray
                if isinstance(dyn_df.columns.values, pd.arrays.ArrowStringArray):
                    c.dynamic[key].columns = pd.Index(dyn_df.columns.astype(object))


def _convert_bool_attrs_to_int(network):
    """Convert boolean attributes to integers for netCDF4 compatibility.

    netCDF4 does not support boolean types for attributes.
    Convert True -> 1, False -> 0.
    """
    # PyPSA uses either .attrs or internal _attrs depending on version
    attr_container = None
    if hasattr(network, "attrs"):
        attr_container = network.attrs
    elif hasattr(network, "_attrs"):
        attr_container = network._attrs

    if attr_container is None:
        logger.warning("Network object has no attribute container for attrs")
        return

    for key, value in list(attr_container.items()):
        if isinstance(value, (bool, np.bool_)):
            attr_container[key] = int(value)


def _patch_linopy_dataset_compat():
    """Patch linopy's local Dataset alias to tolerate Dataset inputs.

    Newer xarray releases reject `xr.Dataset(data_vars=<Dataset>)` during
    linopy model construction. Linopy still performs this conversion when
    transposing expressions, so we intercept xarray's Dataset constructor and
    reinterpret `Dataset(ds)` as a Dataset copy.
    """

    try:
        import xarray.core.dataset as xarray_dataset_module
    except Exception as exc:
        logger.warning(
            f"Could not import xarray.Dataset for compatibility patch: {exc}"
        )
        return

    dataset_cls = getattr(xarray_dataset_module, "Dataset", None)
    if dataset_cls is None or getattr(dataset_cls, "_shift_compat_patched", False):
        return

    original_init = dataset_cls.__init__

    def _dataset_init_compat(self, *args, **kwargs):
        if args and isinstance(args[0], xr.Dataset):
            source_ds = args[0]
            args = ()
            kwargs = dict(kwargs)
            kwargs.setdefault("data_vars", source_ds.data_vars)
            kwargs.setdefault("coords", source_ds.coords)
            kwargs.setdefault("attrs", source_ds.attrs)
        return original_init(self, *args, **kwargs)

    _dataset_init_compat._shift_compat_patched = True  # type: ignore[attr-defined]
    dataset_cls.__init__ = _dataset_init_compat
    logger.info("Applied linopy/xarray Dataset compatibility patch")


def _compute_infeasibility_diagnostics(network, output_dir):
    """Compute infeasibility diagnostics for an infeasible network and write IIS if available."""

    # Attempt to run linopy infeasibility diagnostics
    if hasattr(network.model, "compute_infeasibilities"):
        try:
            infeasible_labels = network.model.compute_infeasibilities()
            logger.info(
                f"Linopy compute_infeasibilities() returned {len(infeasible_labels)} entries"
            )
        except Exception as e:
            logger.warning(f"Could not compute linopy infeasibilities: {e}")
            infeasible_labels = None
    else:
        logger.warning("Network model does not support compute_infeasibilities()")
        infeasible_labels = None

    # Write IIS from backend Gurobi model if available
    gurobi_model = None
    if hasattr(network.model, "backend") and hasattr(network.model.backend, "model"):
        gurobi_model = network.model.backend.model

    if gurobi_model is not None:
        try:
            if hasattr(gurobi_model, "computeIIS"):
                try:
                    gurobi_model.computeIIS()
                    logger.info("Gurobi IIS computed")
                except Exception as iis_err:
                    logger.warning(f"Could not compute IIS on Gurobi model: {iis_err}")

            model_ilp_path = os.path.join(
                output_dir, f"infeasibility_{network.name}.ilp"
            )
            gurobi_model.write(model_ilp_path)
            logger.info(f"IIS .ilp written to: {model_ilp_path}")

            for c in gurobi_model.getConstrs():
                if c.IISConstr:
                    logger.info(f"IIS constraint: {c.ConstrName}")
            for v in gurobi_model.getVars():
                if v.IISLB or v.IISUB:
                    logger.info(f"IIS var: {v.VarName} IISLB={v.IISLB} IISUB={v.IISUB}")
        except Exception as ilp_err:
            logger.warning(f"Could not write IIS .ilp: {ilp_err}")
    else:
        logger.warning("Gurobi backend model not available for IIS .ilp write")

    # Write text infeasibility report if available
    if infeasible_labels:
        if hasattr(network.model, "format_infeasibilities"):
            try:
                infeas_report = network.model.format_infeasibilities()
            except Exception as e:
                infeas_report = f"format_infeasibilities failed: {e}"
        elif hasattr(network.model, "print_infeasibilities"):
            try:
                import io
                import sys

                _buf = io.StringIO()
                _old_stdout = sys.stdout
                sys.stdout = _buf
                network.model.print_infeasibilities()
                sys.stdout = _old_stdout
                infeas_report = _buf.getvalue()
            except Exception as pi_err:
                sys.stdout = _old_stdout
                logger.warning(f"Could not run print_infeasibilities(): {pi_err}")
                infeas_report = "Infeasible constraints identified, but could not capture output of print_infeasibilities()."
        else:
            infeas_report = "Infeasible constraints identified, but format_infeasibilities() and print_infeasibilities() are unavailable."

        infeas_path = os.path.join(output_dir, f"infeasibilities_{network.name}.txt")
        with open(infeas_path, "w", encoding="utf-8") as f:
            f.write(f"Infeasible constraints for network {network.name}:\n")
            f.write("=" * 80 + "\n\n")
            f.write(infeas_report)
        logger.info(f"Infeasibility report written to: {infeas_path}")
    else:
        logger.warning("Model is infeasible but no specific constraints identified")


def solve_network(network, config):
    """Solve the PyPSA optimization with hourly fixed demand.

    The network has hourly Load components with constant p_set.
    Solver minimizes cost to satisfy these fixed hourly demands.
    """
    # Convert arrow strings to regular strings before optimization
    _convert_arrow_strings(network)
    _patch_linopy_dataset_compat()

    solver_cfg = config.get("solver", {})
    solver_name = os.getenv("SHIFT_SOLVER", solver_cfg.get("name", "glpk"))
    solver_options_key = os.getenv(
        "SHIFT_SOLVER_OPTIONS", solver_cfg.get("options", "default")
    )
    solver_options = config.get("solver_options", {}).get(solver_options_key, {})

    logger.info(f"Solving network with {solver_name}...")
    logger.info(f"Solver options: {solver_options}")

    # Add output logging for Gurobi to see what's happening
    if solver_name.lower() == "gurobi" and "OutputFlag" not in solver_options:
        solver_options = {**solver_options, "OutputFlag": 1}  # Enable Gurobi output

    # Solve without constraint injection (hourly loads already in network)
    try:
        status = network.optimize(
            network.snapshots,
            solver_name=solver_name,
            solver_options=solver_options,
            multi_investment_periods=False,
        )

        logger.info(f"Optimization status: {status}")

        status_ok = status == 0 or status == ("ok", "optimal") or status == "optimal"

        if not status_ok:
            logger.warning(f"Non-optimal status ({status})")
            if network.objective is not None:
                logger.info(f"  Objective value: {network.objective}")
            else:
                logger.warning("  Objective is None (no feasible solution found)")
                logger.warning(
                    "Model is infeasible - check network structure and constraints"
                )

                # Use linopy's built-in infeasibility diagnostics
                if solver_name.lower() == "gurobi" and snakemake.params.compute_iis:
                    try:
                        output_dir = os.path.dirname(snakemake.output.network)
                        _compute_infeasibility_diagnostics(
                            network=network,
                            output_dir=output_dir,
                        )
                    except Exception as iis_e:
                        logger.warning(f"Could not compute infeasibilities: {iis_e}")
                elif solver_name.lower() == "gurobi":
                    logger.info("compute_iis flag false, skipping IIS diagnostics")
    except Exception as e:
        logger.error(f"Solver exception: {e}")
        raise

    return network


# ============================================================================
# RESULTS EXTRACTION
# ============================================================================


def extract_lcox(network, product, demands):
    """Extract LCOX from optimized network.

    Returns DataFrame with product-specific columns for supply curve.
    """
    # Define product-specific column names
    if product.lower() in ["steel", "hbi"]:
        load_col = "load [t/h]"
        cost_col = "lcox [EUR/t]"
    elif product.lower() in ["h2"]:
        load_col = "load [MW]"
        cost_col = "lcox [EUR/MWh]"
    else:
        load_col = "load [per h]"
        cost_col = "lcox [EUR/unit]"

    results_df = pd.DataFrame(
        columns=[
            "demand [t]",
            load_col,
            "cost [EUR]",
            cost_col,
        ]
    )

    try:
        obj_value = network.objective
        if obj_value is None or np.isnan(obj_value):
            raise ValueError("Optimization failed to return valid objective")

        demand_annual_t = demands["product_demand_mt"] * 1e6  # Mt → t
        hourly_load_t = demand_annual_t / 8760
        lcox = obj_value / demand_annual_t if demand_annual_t > 0 else np.inf

        results_df.loc[0] = [
            demand_annual_t,
            hourly_load_t,
            obj_value,
            lcox,
        ]
        logger.info(
            f"LCOX calculated: {lcox:.2f} {cost_col.split('[')[1].split(']')[0]}"
        )

    except Exception as e:
        logger.error(f"Optimization infeasible or failed: {e}")
        demand_annual_t = demands["product_demand_mt"] * 1e6
        hourly_load_t = demand_annual_t / 8760
        results_df.loc[0] = [
            demand_annual_t,
            hourly_load_t,
            np.nan,
            np.nan,
        ]

    return results_df


if __name__ == "__main__":
    if snakemake is None:
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "calculate_regional_lcox",
            cost_year="2030",
            region="Europe",
            product="steel",
        )

    # ==================== SETUP ====================
    logger.info("=" * 70)
    # Get the specific demand level for THIS invocation (passed by Snakemake)
    product_demand_mt = float(snakemake.params.product_demand_mt)
    scenario = (
        snakemake.wildcards.scenario
        if hasattr(snakemake.wildcards, "scenario")
        else "reserved"
    )
    logger.info(
        f"LCOX Calculation: region={snakemake.wildcards.region}, "
        f"product={snakemake.wildcards.product}, "
        f"scenario={scenario}, "
        f"demand={product_demand_mt} Mt/year."
    )
    logger.info("=" * 70)

    # Load pre-prepared base network once
    logger.info("Loading base network...")
    base_network = pypsa.Network(snakemake.input.base_network)
    logger.info(
        f"Network loaded: {len(base_network.buses)} buses, "
        f"{len(base_network.generators)} generators, {len(base_network.links)} links"
    )

    # Load local electricity demand for this region
    logger.info("Loading demands...")
    demands = load_demands_for_region(
        region=snakemake.wildcards.region,
        config=snakemake.config,
    )

    logger.info(
        f"Local electricity demand: {demands['local_el_demand_mwh']:.1f} MWh/year"
    )

    # ==================== PROCESS SINGLE DEMAND LEVEL ====================
    electricity_per_product_t = snakemake.config.get("electricity_steel_ratio", 5.25)

    logger.info(f"Processing: {product_demand_mt} Mt/year")

    # ==================== NETWORK SETUP ====================
    # Create a copy of base network
    network = base_network.copy()
    network.name = f"LCOX-{snakemake.wildcards.region}-{snakemake.wildcards.product}-{product_demand_mt}"

    # Ensure snapshot year is set by upstream network preparation;
    # do not override if already set.
    if network.snapshots is None or len(network.snapshots) == 0:
        cost_year = int(snakemake.wildcards.cost_year)
        network.set_snapshots(
            pd.date_range(f"{cost_year}-01-01", periods=8760, freq="h")
        )
        logger.info(
            f"Set snapshots for cost_year={cost_year} (fallback in calculate_lcox)"
        )
    else:
        logger.info(
            f"Snapshots pre-set in network (len={len(network.snapshots)}), not overriding in calculate_lcox"
        )

    # Preserve discount_rate from base network (needed for cost annuitization)
    network.discount_rate = base_network.discount_rate

    # Calculate renewable electricity needed for this demand level
    scaled_product_demand_mwh_per_h = (
        product_demand_mt * 1e6 * electricity_per_product_t / 8760
    )

    logger.info(f"Product demand: {product_demand_mt:.1f} Mt/year")
    logger.info(
        f"Renewable electricity required: {scaled_product_demand_mwh_per_h * 8760:.1f} MWh/year"
    )

    # Create scaled demands dict for this demand level
    scaled_demands = demands.copy()
    scaled_demands["product_demand_mt"] = product_demand_mt
    scaled_demands["product_demand_mwh_per_h"] = scaled_product_demand_mwh_per_h

    # Load incremental generator sets and apply filtering
    logger.info("Loading incremental generator sets...")
    # Skip incremental filtering: use all generators for all demand levels
    logger.info("Using all generators (no incremental filtering applied)")

    # Add hourly load for steel output
    # (This also sets HBI storage e_initial inside add_loads_to_network)
    logger.info("Adding hourly load to network...")
    add_loads_to_network(
        network=network, product=snakemake.wildcards.product, demands=scaled_demands
    )

    # Debug: Print network structure
    logger.info(
        "\n--- Network Structure for Demand Level {:.1f} Mt/year ---".format(
            product_demand_mt
        )
    )
    logger.info(f"Buses: {list(network.buses.index)}")
    logger.info(f"Generators: {len(network.generators)} total")
    for gen in network.generators.index:
        p_max = network.generators.at[gen, "p_nom_max"]
        logger.info(f"  {gen}: p_nom_max={p_max:.1f} MW")
    logger.info(f"Links: {list(network.links.index)}")
    for link in network.links.index:
        p_nominal = network.links.at[link, "p_nom"]
        logger.info(f"  {link}: p_nom={p_nominal:.1f} MW")
    logger.info(f"Stores: {list(network.stores.index)}")
    logger.info(f"Loads: {list(network.loads.index)}")

    # Solve
    logger.info("Optimizing network...")
    if snakemake.config.get("debug_network_inspection", False):
        inspect_network(network, snakemake.wildcards.product)  # Debug inspection
    try:
        solve_network(network, snakemake.config)
        optimization_status = (
            "optimal"
            if network.objective is not None and not np.isnan(network.objective)
            else "infeasible"
        )
    except Exception as e:
        logger.warning(
            f"Solver error for product demand {product_demand_mt} Mt/year: {e}"
        )
        optimization_status = "error"

    if optimization_status != "optimal":
        logger.warning(
            f"Optimization {optimization_status} for product demand {product_demand_mt} Mt/year - returning NaN values"
        )

    # Extract LCOX results
    logger.info("Extracting results...")
    results_df = extract_lcox(
        network=network,
        product=snakemake.wildcards.product,
        demands=scaled_demands,
    )

    # Save results for this demand level
    result_file = snakemake.output.results
    network_file = snakemake.output.network

    logger.info("Saving results...")
    results_df.to_csv(result_file, index=False)
    logger.info(f"Results saved: {result_file}")

    # Save network only if optimization succeeded
    if optimization_status == "optimal":
        try:
            _convert_bool_attrs_to_int(network)
            network.export_to_netcdf(network_file)
            logger.info(f"Network saved: {network_file}")
        except Exception as e:
            logger.warning(f"Could not save network: {e}")
    else:
        logger.warning(
            f"Skipping network export due to solver status: {optimization_status}"
        )

    logger.info("=" * 70)

    if optimization_status == "optimal":
        logger.info("Demand level completed successfully!")
    else:
        logger.warning(
            f"Demand level completed with solver status: {optimization_status}"
        )
    logger.info("=" * 70)
