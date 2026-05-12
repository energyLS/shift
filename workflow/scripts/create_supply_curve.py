import os
from typing import Any
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pypsa

snakemake: Any = globals().get("snakemake")

matplotlib.use("Agg")


def get_steel_demand(region):

    steel_demand = pd.read_csv(snakemake.input.steel_demand, header=0)

    steel_demand_region = steel_demand.loc[
        steel_demand.region == region
    ].SteelProductionMt

    return steel_demand_region


def get_final_demand(region):

    final_demand_data = pd.read_csv(snakemake.input.local_demand, header=0)
    final_demand = final_demand_data.loc[final_demand_data.region == region].demand

    return final_demand


def get_stage_ratios_from_skeleton():
    """Extract stage conversion ratios from the PyPSA supply-chain skeleton.

    Returns ratios as input-per-output:
      - ore_per_hbi
      - hbi_per_steel
      - ore_per_steel
    """
    network = pypsa.Network(snakemake.input.skeleton)

    ore_per_hbi = None
    hbi_per_steel = None

    if "dri" in network.links.index:
        dri_eff = float(network.links.at["dri", "efficiency"])
        if dri_eff > 0:
            ore_per_hbi = 1.0 / dri_eff

    if "eaf" in network.links.index:
        eaf_eff = float(network.links.at["eaf", "efficiency"])
        if eaf_eff > 0:
            hbi_per_steel = 1.0 / eaf_eff

    ore_per_steel = None
    if ore_per_hbi is not None and hbi_per_steel is not None:
        ore_per_steel = ore_per_hbi * hbi_per_steel

    return {
        "ore_per_hbi": ore_per_hbi,
        "hbi_per_steel": hbi_per_steel,
        "ore_per_steel": ore_per_steel,
    }


def get_stage_metadata(product, stage_ratios):
    """Return stage input/output metadata for the current supply-curve product."""
    if product == "hbi":
        return {
            "stage_input_commodity": "iron_ore",
            "stage_output_commodity": "hbi",
            "stage_input_per_output": stage_ratios["ore_per_hbi"],
        }
    if product == "steel":
        return {
            "stage_input_commodity": "hbi",
            "stage_output_commodity": "steel",
            "stage_input_per_output": stage_ratios["hbi_per_steel"],
        }
    return {
        "stage_input_commodity": "",
        "stage_output_commodity": product,
        "stage_input_per_output": None,
    }


def create_supply_curve():
    """
    Create supply curve from reserved and unreserved scenario LCoX results.

    Loads results from two distinct optimization scenarios:
      - reserved: highest-CF sites reserved for domestic demand
            - unreserved: full renewable stack available (optional/fallback)

    Combines results, validates they differ, and produces CSV/PDF outputs.
    """
    stage_ratios = get_stage_ratios_from_skeleton()
    stage_meta = get_stage_metadata(product, stage_ratios)

    def find_process_label_for_product(config, product_name):
        chains = config.get("trade_chains") or []
        for chain in chains:
            for stage in chain.get("stages", []):
                if stage.get("output_commodity") == product_name:
                    return stage.get("process_label")
        return None

    # process_label derived from config (which stage produces this product)
    process_label_config = (
        find_process_label_for_product(snakemake.config, product) or product
    )
    # detect whether this script was invoked per-stage (Snakemake wildcard `process_label`) or as full-product
    invoked_with_process_label = hasattr(snakemake.wildcards, "process_label")
    # For later tagging use the config-derived label
    process_label = process_label_config

    reserved_files = snakemake.input.lco_reserved
    print("reserved scenario files:", reserved_files)
    df_reserved = pd.concat(
        (pd.read_csv(f, sep=",") for f in reserved_files), ignore_index=True
    )
    print("reserved scenario data loaded")

    unreserved_files = snakemake.input.lco_unreserved
    if unreserved_files and len(unreserved_files) > 0:
        print("unreserved scenario files:", unreserved_files)
        df_unreserved = pd.concat(
            (pd.read_csv(f, sep=",") for f in unreserved_files), ignore_index=True
        )
        print("unreserved scenario data loaded")
        df_merged = df_reserved.copy()
        df_sub = df_unreserved.copy()
    else:
        print("unreserved scenario not provided; using reserved for both outputs")
        df_merged = df_reserved.copy()
        df_sub = df_reserved.copy()

    infeasible_rows = df_merged[
        df_merged[columns["cost per unit"]] == "infeasible"
    ].index
    df_merged = df_merged.drop(infeasible_rows)
    df_sub = df_sub.drop(infeasible_rows)
    print("deleted infeasible rows to prepare for plotting")

    df_all_demand = pd.read_csv(snakemake.input.local_demand, header=0)
    df_local_demand = df_all_demand.loc[
        df_all_demand["region"] == snakemake.wildcards["region"]
    ]

    if product == "hydrogen":
        conversion_factor = 0.75
    elif product in ["steel", "eaf", "hbi", "eaf-grid"]:
        conversion_factor = 1 / snakemake.config["electricity_steel_ratio"]
    else:
        raise ValueError(f"product {product} not recognized for supply curve plotting")

    local_load = float(
        df_local_demand["demand"].values[0]
        * df_local_demand["el_share"].values[0]
        / 100
    )
    product_subtract = local_load * conversion_factor

    print(f"local el load is: {local_load} MWh")
    print(
        f"product substraction due to local el load is: {product_subtract} {columns['product_unit']}"
    )

    df_sub[columns["demand"]] = df_sub[columns["demand"]].subtract(product_subtract)
    df_sub.loc[df_sub[columns["demand"]] < 0, columns["demand"]] = 0
    print("local el load has been subtracted from global supply")

    df_merged["stage_input_commodity"] = stage_meta["stage_input_commodity"]
    df_merged["stage_output_commodity"] = stage_meta["stage_output_commodity"]
    df_merged["stage_input_per_output"] = stage_meta["stage_input_per_output"]
    df_merged["process_label"] = process_label

    df_sub["stage_input_commodity"] = stage_meta["stage_input_commodity"]
    df_sub["stage_output_commodity"] = stage_meta["stage_output_commodity"]
    df_sub["stage_input_per_output"] = stage_meta["stage_input_per_output"]
    df_sub["process_label"] = process_label

    df_merged["stage_marginal_cost_per_unit"] = df_merged[
        columns["cost per unit"]
    ].astype(float)
    df_sub["stage_marginal_cost_per_unit"] = df_sub[columns["cost per unit"]].astype(
        float
    )

    # For per-stage runs (when invoked with Snakemake wildcard `process_label`)
    # we must NOT include upstream input prices. Those are only for full-chain product
    # aggregation. If this script is invoked as a stage, set upstream input costs
    # to zero to preserve Option B semantics.
    if invoked_with_process_label:
        iron_ore_total_cost = 0
    else:
        if product == "steel":
            ore_ratio = stage_ratios["ore_per_steel"]
            if ore_ratio is None:
                ore_ratio = snakemake.config["iron_ore"]["ore_to_steel_ratio"]
            iron_ore_total_cost = (
                snakemake.config["iron_ore"]["marginal_cost"] * ore_ratio
            )
        elif product == "hbi":
            ore_ratio = stage_ratios["ore_per_hbi"]
            if ore_ratio is None:
                ore_ratio = snakemake.config["iron_ore"]["ore_to_steel_ratio"]
            iron_ore_total_cost = (
                snakemake.config["iron_ore"]["marginal_cost"] * ore_ratio
            )
        elif product in ["hydrogen", "eaf", "eaf-grid"]:
            iron_ore_total_cost = 0
        else:
            raise ValueError(
                f"product {product} not recognized for supply curve plotting"
            )

    df_merged["iron_ore_cost_per_unit"] = iron_ore_total_cost
    df_sub["iron_ore_cost_per_unit"] = iron_ore_total_cost
    df_merged["total_cost_per_unit"] = (
        df_merged["stage_marginal_cost_per_unit"] + df_merged["iron_ore_cost_per_unit"]
    )
    df_sub["total_cost_per_unit"] = (
        df_sub["stage_marginal_cost_per_unit"] + df_sub["iron_ore_cost_per_unit"]
    )

    if product == "steel":
        df_sub = df_merged.copy()

    df_merged.to_csv(snakemake.output.supply, index=False)

    try:
        unreserved_path = snakemake.output.supply_unreserved
    except Exception:
        unreserved_path = None

    if (
        unreserved_path
        and str(unreserved_path).endswith(".csv")
        and len(snakemake.input.lco_unreserved) > 0
    ):
        df_sub.to_csv(unreserved_path, index=False)
        print(f"Saved unreserved supply curve: {unreserved_path}")
    else:
        print(
            "Skipping supply_unreserved output (unreserved scenario not provided or not enabled)"
        )

    if unreserved_path:
        try:
            if not os.path.exists(unreserved_path):
                df_sub.to_csv(unreserved_path, index=False)
                print(f"Wrote fallback supply_unreserved file: {unreserved_path}")
        except Exception:
            pass

    y_merged = df_merged["stage_marginal_cost_per_unit"]
    y_sub = df_sub["stage_marginal_cost_per_unit"]

    plt.plot(
        df_merged[columns["demand"]].astype(int) / (1e6),
        y_merged,
        linestyle="-",
        marker="o",
        label="supply (reserved)",
    )

    if len(snakemake.input.lco_unreserved) > 0:
        plt.plot(
            df_sub[columns["demand"]].astype(int) / (1e6),
            y_sub,
            linestyle="--",
            color="C1",
            marker="o",
            markerfacecolor="none",
            label="supply (unreserved)",
        )
        plt.axvline(
            product_subtract / (1e6),
            label="local electricity demand (converted to product)",
            linestyle="--",
            color="C1",
        )

    if product == "steel":
        y_merged_total = df_merged["total_cost_per_unit"]
        y_sub_total = df_sub["total_cost_per_unit"]
        plt.plot(
            df_merged[columns["demand"]].astype(int) / (1e6),
            y_merged_total,
            linestyle="-",
            color="C3",
            marker="s",
            label="supply total (marginal + ore, reserved)",
        )
        if len(snakemake.input.lco_unreserved) > 0:
            plt.plot(
                df_sub[columns["demand"]].astype(int) / (1e6),
                y_sub_total,
                linestyle="--",
                color="C4",
                marker="s",
                markerfacecolor="none",
                label="supply total (marginal + ore, unreserved)",
            )

    if product == "hydrogen":
        final_demand = get_final_demand(snakemake.wildcards["region"])
        plt.axvline(
            x=final_demand.values[0] / (1e6), linestyle="-", label="final energy demand"
        )
        plt.axvline(
            x=0.6 * final_demand.values[0] / (1e6),
            linestyle=":",
            label="60% final energy demand",
        )
        plt.axvline(
            x=0.2 * final_demand.values[0] / (1e6),
            linestyle="-.",
            label="20% final energy demand",
        )
    elif product in ["steel", "eaf", "eaf-grid"]:
        steel_demand = get_steel_demand(snakemake.wildcards["region"])
        plt.axvline(x=steel_demand.values[0], linestyle=":", label="local steel demand")
    elif product == "hbi":
        steel_demand = get_steel_demand(snakemake.wildcards["region"])
        hbi_per_steel = stage_ratios["hbi_per_steel"] or 1.0
        plt.axvline(
            x=steel_demand.values[0] * hbi_per_steel,
            linestyle=":",
            label="local steel demand (HBI-equivalent)",
        )

    plt.title(
        f"levelized cost of {product} production in {snakemake.wildcards['region']}"
    )
    plt.ylim(columns["ylim"])
    plt.xlabel(columns["xlabel"])
    plt.ylabel(columns["cost per unit"])
    plt.legend(loc="lower right")
    plt.savefig(snakemake.output.supply_curve, format="pdf", bbox_inches="tight")

    return


# Setup columns and product before function execution (needed for both Snakemake and main)
if snakemake is None:
    from _helpers import mock_snakemake

    snakemake = mock_snakemake(
        "create_supply_curve",
        cost_year="2030",
        region="South_South_America",
        product="steel",
    )


# Derive product from process_label (new approach)
def _derive_product_from_process_label(process_label, config):
    """Reverse-lookup: process_label → output_commodity (product)."""
    chains = config.get("trade_chains") or []
    for chain in chains:
        stages = chain.get("stages", [])
        for s in stages:
            if s.get("process_label") == process_label:
                return s.get("output_commodity")
    # Fallback: if no chain found, assume process_label is product
    return process_label


# Get process_label from wildcards; fallback to product for backward compatibility
if hasattr(snakemake.wildcards, "process_label"):
    process_label = snakemake.wildcards["process_label"]
    product = _derive_product_from_process_label(process_label, snakemake.config)
    print(f"Using process_label={process_label}, derived product={product}")
else:
    # Backward compatibility: use product wildcard
    product = snakemake.wildcards["product"]
    process_label = None
    print(f"Using product={product} (no process_label provided)")

if product == "hydrogen":
    columns = {
        "demand factor": "demand factor [%]",
        "demand": "demand [t]",
        "load": "load [MW]",
        "total cost": "cost [EUR]",
        "cost per unit": "lcox [EUR/MWh]",
        "xlabel": "Demand in TWh",
        "product_unit": "MWh",
        "ylim": (0, 100),
    }
elif product in ["steel", "eaf", "hbi", "eaf-grid"]:
    columns = {
        "demand factor": "demand factor [%]",
        "demand": "demand [t]",
        "load": "load [t/h]",
        "total cost": "cost [EUR]",
        "cost per unit": "lcox [EUR/t]",
        "xlabel": "Demand in Mt",
        "product_unit": "t",
        "ylim": (0, 900),
    }
else:
    raise ValueError(f"product {product} not recognized for supply curve plotting")

if __name__ == "__main__":
    create_supply_curve()
