import pandas as pd
import matplotlib.pyplot as plt

def create_supply_curve_with_demand():
# input: "resources/lcoh/{region}/results_{demand_factor}.csv",
# outputs: supply="resources/supply_curves/{region}_hydrogen.csv", supply_curve="resources/supply_curves/{region}_hydrogen.pdf"
    all_files = snakemake.input.lcoh_data
    print("files to merge:", all_files)

    final_demand_data = pd.read_csv(snakemake.input.final_demand_data,header=0)
    final_demand = float(final_demand_data.loc[final_demand_data.region == "{}".format(snakemake.wildcards['region'])]["demand"])
    #final_green is wrong, this is final_el
    final_green = final_demand*float(final_demand_data.loc[final_demand_data.region == "{}".format(snakemake.wildcards['region'])]["el_share"])/100
    print("demand data",final_demand)

    df_from_each_file = (pd.read_csv(f, sep=',',index_col=0) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_sub = df_merged.copy()
    print("merged file has been created")

    # first calculate part of local supply that should be used to cover local el demand
    df_all_demand = pd.read_csv(snakemake.input.final_demand_data,header = 0)
    df_local_demand = df_all_demand.loc[df_all_demand["region"]==snakemake.wildcards['region']]
    # total final energy consumption for the region * percentage of final energy consumption needed to meet local el demand = local el demand need in MWh
    # since the demand is for hydrogen (after electrolysis of 75%), local el load must be converted to the amount of decreasing hydrogen production
    local_load = float(df_local_demand["demand"]*df_local_demand["el_share"]/100)*0.75
    print("local el load is: ", local_load)

    # ******************* SUBTRACTING LOCAL DEMAND ***********************
    #  remove local load from demand and drop all negative rows (generators that are only local)
    df_sub["demand [MWh]"] = df_sub["demand [MWh]"].subtract(local_load)
    df_sub["demand [MWh]"][df_sub["demand [MWh]"]<0] = 0
    print("local load has been subtracted from global supply")
    
    # preparing for plotting
    infeasible_rows = df_merged[df_merged["LCOH [EUR/MWh]"] == "infeasible"].index
    df_merged = df_merged.drop(infeasible_rows)
    df_sub = df_sub.drop(infeasible_rows)
    print("deleted infeasible rows to prepare for plotting")

    # creates and saves supply curve plot
    plt.plot(df_merged["demand [MWh]"].astype(int)/(1e6),df_merged["LCOH [EUR/MWh]"].astype(int),linestyle='-', marker='o',label="supply")
    plt.axvline(x=final_demand/(1e6),linestyle='-',label="final demand")
    plt.axvline(x=0.2*final_demand/(1e6),linestyle='-.',label="20 final demand")
    plt.axvline(x=0.6*final_demand/(1e6),linestyle=':',label="60 final demand")
    
    # the subtracted plot
    plt.plot(df_sub["demand [MWh]"].astype(float)/(1e6),df_sub["LCOH [EUR/MWh]"].astype(float),linestyle='--', color='C1',marker='o',markerfacecolor="none",label="subracted supply")
    plt.axvline(local_load/(1e6),label="local demand",linestyle='--',color='C1')

    #plt.axvline(x=final_green/(1e6),linestyle='-',label="final demand RE",color='springgreen')
    #plt.axvline(x=0.2*final_green/(1e6),linestyle='--',label="20 final demand RE",color='springgreen')
    #plt.axvline(x=0.6*final_green/(1e6),linestyle='-.',label="60 final demand RE",color='springgreen')
    
    plt.ylabel("LCOH [EUR/MWh]")
    plt.ylim((0,100))
    plt.title("levelized cost of hydrogen production in {}".format(snakemake.wildcards['region']))
    plt.xlabel("demand [TWh]")
    plt.legend()
    plt.savefig(snakemake.output.supply_curve, format="pdf", bbox_inches="tight") 

    return

if __name__ == "__main__":
    create_supply_curve_with_demand()