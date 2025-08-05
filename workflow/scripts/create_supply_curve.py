import pandas as pd
import matplotlib.pyplot as plt


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
    # input: "resources/lcoh/{region}/results_{demand_factor}.csv",
    # outputs: supply="resources/supply_curves/{region}_hydrogen.csv", supply_curve="resources/supply_curves/{region}_hydrogen.pdf"
    all_files = snakemake.input.lco_product_data
    print("files to merge:", all_files)

    df_from_each_file = (pd.read_csv(f, sep=",", index_col=0) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_sub = df_merged.copy()
    print("merged file has been created")

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
    elif product == "steel":
        conversion_factor = 1 / snakemake.config["electricity_steel_ratio"]

    local_load = float(df_local_demand["demand"] * df_local_demand["el_share"] / 100)

    product_subtract = local_load * conversion_factor

    print(f"local el load is: {local_load} MWh")
    print(
        f"product substraction due to local el load is: {product_subtract} {columns['product_unit']}"
    )

    # # # ******************* SUBTRACTING LOCAL DEMAND ***********************
    # # remove local load from demand and drop all negative rows (generators that are only local)
    df_sub[columns["demand"]] = df_sub[columns["demand"]].subtract(product_subtract)
    df_sub[columns["demand"]][df_sub[columns["demand"]] < 0] = 0
    print("local el load has been subtracted from global supply")

    # saves the merged costs in a supply curve csv
    df_sub.to_csv(snakemake.output.supply)

    # creates and saves supply curve plot
    if product == "steel":
        iron_ore_total_cost = snakemake.config["iron_ore"]["marginal_cost"] * snakemake.config["iron_ore"]["ore_to_steel_ratio"]
        y_merged = df_merged[columns["cost per unit"]].astype(float) + iron_ore_total_cost
        y_sub = df_sub[columns["cost per unit"]].astype(float) + iron_ore_total_cost
    else:
        y_merged = df_merged[columns["cost per unit"]].astype(float)
        y_sub = df_sub[columns["cost per unit"]].astype(float)

    plt.plot(
        df_merged[columns["demand"]].astype(int) / (1e6),
        y_merged,
        linestyle="-",
        marker="o",
        label="supply",
    )

    # the subtracted plot
    plt.plot(
        df_sub[columns["demand"]].astype(int) / (1e6),
        y_sub,
        linestyle="--",
        color="C1",
        marker="o",
        markerfacecolor="none",
        label="supply w. local el. demand subtracted",
    )

    plt.axvline(
        product_subtract / (1e6),
        label="local energy demand for el.",
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

    elif product == "steel":
        steel_demand = get_steel_demand(snakemake.wildcards["region"])
        plt.axvline(
            x=0.6 * steel_demand.values[0], linestyle=":", label="local steel demand"
        )

    plt.title(
        f"levelized cost of {product} production in {snakemake.wildcards['region']}",
    )
    plt.ylim(columns["ylim"])
    plt.xlabel(columns["xlabel"])
    plt.ylabel(columns["cost per unit"])
    plt.legend()
    plt.savefig(snakemake.output.supply_curve, format="pdf", bbox_inches="tight")

    return


if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "create_supply_curve",
            cost_year="2030",
            region="Europe",
            product="steel",
        )

    product = snakemake.wildcards["product"]
    if product == "hydrogen":
        columns = {
            "demand factor": "demand factor [%]",
            "demand": "demand [MWh]",
            "load": "load [MW]",
            "total cost": "cost [EUR]",
            "cost per unit": "LCOH [EUR/MWh]",
            "xlabel": "Demand in TWh",
            "product_unit": "MWh",
            "ylim": (0, 100),
        }
    elif product == "steel":
        columns = {
            "demand factor": "demand factor [%]",
            "demand": "demand [t]",
            "load": "load [t/h]",
            "total cost": "cost [EUR]",
            "cost per unit": "LCOS [EUR/t]",
            "xlabel": "Demand in Mt",
            "product_unit": "t",
            "ylim": (0, 900),
        }

    create_supply_curve()
