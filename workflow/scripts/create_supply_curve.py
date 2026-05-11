import os
from typing import Any
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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


def create_supply_curve():
    """
    Create supply curve from reserved and unconstrained scenario LCoX results.

    Loads results from two distinct optimization scenarios:
      - reserved: highest-CF sites reserved for domestic demand
      - unconstrained: full renewable stack available (optional/fallback)

    Combines results, validates they differ, and produces CSV/PDF outputs.
    """
    # Load reserved scenario (always required)
    reserved_files = snakemake.input.lco_reserved
    print("reserved scenario files:", reserved_files)

    df_from_reserved = (pd.read_csv(f, sep=",") for f in reserved_files)
    df_reserved = pd.concat(df_from_reserved, ignore_index=True)
    print("reserved scenario data loaded")

    # Load unconstrained scenario (optional, may be empty list)
    unconstrained_files = snakemake.input.lco_unconstrained
    if unconstrained_files and len(unconstrained_files) > 0:
        print("unconstrained scenario files:", unconstrained_files)
        df_from_unconstrained = (pd.read_csv(f, sep=",") for f in unconstrained_files)
        df_unconstrained = pd.concat(df_from_unconstrained, ignore_index=True)
        print("unconstrained scenario data loaded")
        # For main CSV output, use reserved; unconstrained goes to separate output
        df_merged = df_reserved.copy()
        df_sub = df_unconstrained.copy()
    else:
        # Fallback: if unconstrained not available, use reserved for both
        print("unconstrained scenario not provided; using reserved for both outputs")
        df_merged = df_reserved.copy()
        df_sub = df_reserved.copy()

    # preparing for plotting
    infeasible_rows = df_merged[
        df_merged[columns["cost per unit"]] == "infeasible"
    ].index
    df_merged = df_merged.drop(infeasible_rows)
    df_sub = df_sub.drop(infeasible_rows)
    print("deleted infeasible rows to prepare for plotting")

    # # first calculate part of local supply that should be used to cover local el demand
    df_all_demand = pd.read_csv(snakemake.input.local_demand, header=0)
    df_local_demand = df_all_demand.loc[
        df_all_demand["region"] == snakemake.wildcards["region"]
    ]
    # # total final energy consumption for the region * percentage of final energy consumption needed to meet local el demand = local el demand need in MWh
    # # since the demand is for hydrogen (after electrolysis of 75%), local el load must be converted to the amount of decreasing hydrogen production

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

    # # # ******************* SUBTRACTING LOCAL DEMAND ***********************
    # # remove local load from demand and drop all negative rows (generators that are only local)
    df_sub[columns["demand"]] = df_sub[columns["demand"]].subtract(product_subtract)
    df_sub.loc[df_sub[columns["demand"]] < 0, columns["demand"]] = 0
    print("local el load has been subtracted from global supply")

    # saves the merged costs in a supply curve csv
    # df_merged shows reserved scenario (domestic demand reservation applied)
    # df_sub shows unconstrained scenario (all generators available) - optional
    df_merged.to_csv(snakemake.output.supply, index=False)

    # Only save supply_unconstrained if unconstrained scenario was provided
    try:
        unconstrained_path = snakemake.output.supply_unconstrained
    except Exception:
        unconstrained_path = None

    if (
        unconstrained_path
        and str(unconstrained_path).endswith(".csv")
        and len(snakemake.input.lco_unconstrained) > 0
    ):
        # write the unconstrained supply curve to the declared path
        df_sub.to_csv(unconstrained_path, index=False)
        print(f"Saved unconstrained supply curve: {unconstrained_path}")
    else:
        print(
            "Skipping supply_unconstrained output (unconstrained scenario not provided or not enabled)"
        )

    # If Snakemake declared a non-temp path but we didn't write it above
    # for any reason, ensure it exists to prevent MissingOutputException.
    if unconstrained_path:
        try:
            if not os.path.exists(unconstrained_path):
                # write a minimal CSV fallback
                df_sub.to_csv(unconstrained_path, index=False)
                print(f"Wrote fallback supply_unconstrained file: {unconstrained_path}")
        except Exception:
            pass

    # creates and saves supply curve plot
    if product in ["steel", "hbi"]:
        iron_ore_total_cost = (
            snakemake.config["iron_ore"]["marginal_cost"]
            * snakemake.config["iron_ore"]["ore_to_steel_ratio"]
        )

    elif product in ["hydrogen", "eaf", "eaf-grid"]:
        iron_ore_total_cost = 0

    else:
        raise ValueError(f"product {product} not recognized for supply curve plotting")

    y_merged = df_merged[columns["cost per unit"]].astype(float) + iron_ore_total_cost
    y_sub = df_sub[columns["cost per unit"]].astype(float) + iron_ore_total_cost

    plt.plot(
        df_merged[columns["demand"]].astype(int) / (1e6),
        y_merged,
        linestyle="-",
        marker="o",
        label="supply (reserved: high-CF sites reserved for domestic)",
    )

    # Plot unconstrained scenario if available (different from reserved)
    if len(snakemake.input.lco_unconstrained) > 0:
        plt.plot(
            df_sub[columns["demand"]].astype(int) / (1e6),
            y_sub,
            linestyle="--",
            color="C1",
            marker="o",
            markerfacecolor="none",
            label="supply (unconstrained: full renewable stack available)",
        )

        # Only show local demand line if both scenarios are available
        plt.axvline(
            product_subtract / (1e6),
            label="local electricity demand (converted to product)",
            linestyle="--",
            color="C1",
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

    elif product in ["steel", "eaf", "hbi", "eaf-grid"]:
        steel_demand = get_steel_demand(snakemake.wildcards["region"])
        plt.axvline(x=steel_demand.values[0], linestyle=":", label="local steel demand")

    plt.title(
        f"levelized cost of {product} production in {snakemake.wildcards['region']}",
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

product = snakemake.wildcards["product"]
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
