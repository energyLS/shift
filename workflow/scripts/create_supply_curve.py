import pandas as pd
import matplotlib.pyplot as plt


def create_supply_curve():
    # input: "resources/lcoh/{region}/results_{demand_factor}.csv",
    # outputs: supply="resources/supply_curves/{region}_hydrogen.csv", supply_curve="resources/supply_curves/{region}_hydrogen.pdf"
    all_files = snakemake.input.lco_product_data
    print("files to merge:", all_files)

    df_from_each_file = (pd.read_csv(f, sep=",", index_col=0) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    print("merged file has been created")

    # preparing for plotting
    infeasible_rows = df_merged[
        df_merged[columns["cost per unit"]] == "infeasible"
    ].index
    df_merged = df_merged.drop(infeasible_rows)
    print("deleted infeasible rows to prepare for plotting")

    # # first calculate part of local supply that should be used to cover local el demand
    df_all_demand = pd.read_csv(snakemake.input.local_demand, header=0)
    df_local_demand = df_all_demand.loc[
        df_all_demand["region"] == snakemake.wildcards["region"]
    ]
    # # total final energy consumption for the region * percentage of final energy consumption needed to meet local el demand = local el demand need in MWh
    # # since the demand is for hydrogen (after electrolysis of 75%), local el load must be converted to the amount of decreasing hydrogen production

    # TODO get local steel demand
    if product == "hydrogen":
        local_load = (
            float(df_local_demand["demand"] * df_local_demand["el_share"] / 100) * 0.75
        )
    elif product == "steel":
        local_load = 0
    # # print("local el load is: ", local_load)

    # # # ******************* SUBTRACTING LOCAL DEMAND ***********************
    # # remove local load from demand and drop all negative rows (generators that are only local)
    df_merged[columns["demand"]] = df_merged[columns["demand"]].subtract(local_load)
    df_merged[columns["demand"]][df_merged[columns["demand"]] < 0] = 0
    print("local load has been subtracted from global supply")

    # saves the merged costs in a supply curve csv
    df_merged.to_csv(snakemake.output.supply)

    # creates and saves supply curve plot
    plt.plot(
        df_merged[columns["demand"]].astype(int) / (1e6),
        df_merged[columns["cost per unit"]].astype(int),
        linestyle="-",
        marker="o",
    )
    plt.title(
        f"levelized cost of {product} production in {snakemake.wildcards['region']}",
    )
    # plt.ylim((0, 100))
    plt.xlabel(columns["xlabel"])
    plt.ylabel(columns["cost per unit"])
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
        }
    elif product == "steel":
        columns = {
            "demand factor": "demand factor [%]",
            "demand": "demand [t]",
            "load": "load [t/h]",
            "total cost": "cost [EUR]",
            "cost per unit": "LCOS [EUR/t]",
            "xlabel": "Demand in Mt",
        }

    create_supply_curve()
