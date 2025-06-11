import pandas as pd
import matplotlib.pyplot as plt

def create_supply_curve():
# input: "resources/lcoh/{region}/results_{demand_factor}.csv",
# outputs: supply="resources/supply_curves/{region}_hydrogen.csv", supply_curve="resources/supply_curves/{region}_hydrogen.pdf"
    all_files = snakemake.input.lcoh_data
    print("files to merge:", all_files)

    df_from_each_file = (pd.read_csv(f, sep=',',index_col=0) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    print("merged file has been created")

    # preparing for plotting
    infeasible_rows = df_merged[df_merged["LCOH [EUR/MWh]"] == "infeasible"].index
    df_merged = df_merged.drop(infeasible_rows)
    print("deleted infeasible rows to prepare for plotting")

    # # first calculate part of local supply that should be used to cover local el demand
    df_all_demand = pd.read_csv(snakemake.input.local_demand,header = 0)
    df_local_demand = df_all_demand.loc[df_all_demand["region"]==snakemake.wildcards['region']]
    # # total final energy consumption for the region * percentage of final energy consumption needed to meet local el demand = local el demand need in MWh
    # # since the demand is for hydrogen (after electrolysis of 75%), local el load must be converted to the amount of decreasing hydrogen production
    local_load = float(df_local_demand["demand"]*df_local_demand["el_share"]/100)*0.75
    # # print("local el load is: ", local_load)

    # # # ******************* SUBTRACTING LOCAL DEMAND ***********************
    # # remove local load from demand and drop all negative rows (generators that are only local)
    df_merged["demand [MWh]"] = df_merged["demand [MWh]"].subtract(local_load)
    df_merged["demand [MWh]"][df_merged["demand [MWh]"]<0] = 0
    print("local load has been subtracted from global supply")
    
    # saves the merged costs in a supply curve csv
    df_merged.to_csv(snakemake.output.supply)

    # creates and saves supply curve plot
    plt.plot(df_merged["demand [MWh]"].astype(int)/(1e6),df_merged["LCOH [EUR/MWh]"].astype(int),linestyle='-', marker='o')
    plt.title("levelized cost of hydrogen production in {}".format(snakemake.wildcards['region']),)
    plt.ylim((0,100))
    plt.xlabel("demand [TWh]")
    plt.ylabel("LCOH [EUR/MWh]")
    plt.savefig(snakemake.output.supply_curve, format="pdf", bbox_inches="tight") 

    return

if __name__ == "__main__":
    create_supply_curve()