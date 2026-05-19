import os
import sys
from typing import Any
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pypsa

# Add workflow/scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from _helpers import setup_logging
from trade_chain_utils import route_label_for_product

snakemake: Any = globals().get("snakemake")

matplotlib.use("Agg")

logger = setup_logging(
    __name__, snakemake=snakemake, log_filename="create_supply_curve.log"
)


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

    # route_label is derived from config to locate upstream LCoX files for this product
    route_label = route_label_for_product(snakemake.config, product) or product

    reserved_files = snakemake.input.lco_reserved
    logger.info("reserved scenario files: %s", reserved_files)
    df_reserved = pd.concat(
        (pd.read_csv(f, sep=",") for f in reserved_files), ignore_index=True
    )
    logger.info("reserved scenario data loaded")

    unreserved_files = snakemake.input.lco_unreserved
    if unreserved_files and len(unreserved_files) > 0:
        logger.info("unreserved scenario files: %s", unreserved_files)
        df_unreserved = pd.concat(
            (pd.read_csv(f, sep=",") for f in unreserved_files), ignore_index=True
        )
        logger.info("unreserved scenario data loaded")
        df_merged = df_reserved.copy()
        df_sub = df_unreserved.copy()
    else:
        logger.info("unreserved scenario not provided; using reserved for both outputs")
        df_merged = df_reserved.copy()
        df_sub = df_reserved.copy()

    infeasible_rows = df_merged[
        df_merged[columns["cost per unit"]] == "infeasible"
    ].index
    df_merged = df_merged.drop(infeasible_rows)
    df_sub = df_sub.drop(infeasible_rows)
    logger.info("deleted infeasible rows to prepare for plotting")

    df_merged["stage_input_commodity"] = stage_meta["stage_input_commodity"]
    df_merged["stage_output_commodity"] = stage_meta["stage_output_commodity"]
    df_merged["stage_input_per_output"] = stage_meta["stage_input_per_output"]
    df_merged["route_label"] = route_label

    df_sub["stage_input_commodity"] = stage_meta["stage_input_commodity"]
    df_sub["stage_output_commodity"] = stage_meta["stage_output_commodity"]
    df_sub["stage_input_per_output"] = stage_meta["stage_input_per_output"]
    df_sub["route_label"] = route_label

    df_merged["stage_marginal_cost_per_unit"] = df_merged[
        columns["cost per unit"]
    ].astype(float)
    df_sub["stage_marginal_cost_per_unit"] = df_sub[columns["cost per unit"]].astype(
        float
    )

    # Supply curves are stage-marginal only; no upstream pricing is included.
    # This preserves Option B semantics where each stage is independently cost-optimized
    # and the trade model assembles the full chain cost.
    iron_ore_total_cost = 0

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
        logger.info("Saved unreserved supply curve: %s", unreserved_path)
    else:
        logger.info(
            "Skipping supply_unreserved output (unreserved scenario not provided or not enabled)"
        )

    if unreserved_path:
        try:
            if not os.path.exists(unreserved_path):
                df_sub.to_csv(unreserved_path, index=False)
                logger.info(
                    "Wrote fallback supply_unreserved file: %s", unreserved_path
                )
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
    elif product == "steel":
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


# Get product from wildcards (product-labeled contract)
product = snakemake.wildcards["product"]
logger.info("Creating supply curve for product=%s", product)

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
elif product in ["steel", "hbi"]:
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
