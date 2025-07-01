import pypsa
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# from pyomo.environ import Constraint
import xarray as xr


def calc_annuity(i_rate, lifetime):
    # calculating annuity factor with interest rate and lifetime
    return i_rate / (1 - (1 + i_rate) ** (-lifetime))


def calc_cap_cost(costs, tech, i_rate):
    # select the relevant part of the dataframe
    sub_data = costs.loc[costs["technology"] == tech]

    # save values (not all tech has FOM, so setting FOM to zero in that case)
    if len(sub_data["value"].loc[sub_data["parameter"] == "FOM"].values) > 0:
        FOM = sub_data["value"].loc[sub_data["parameter"] == "FOM"].values
    else:
        FOM = 0

    CAPEX = sub_data["value"].loc[sub_data["parameter"] == "investment"].values
    lifetime = sub_data["value"].loc[sub_data["parameter"] == "lifetime"].values

    # calculate parameters
    annuity = calc_annuity(i_rate, lifetime)

    # returns cap costs in EUR/MW
    return (annuity + FOM / 100) * CAPEX * 1e3


# inputs are solar potentials, wind potentials, costs and load
def building_model(ds, dw, dc, load, h_cost):
    # create network + buses + carriers
    network = pypsa.Network()

    network.set_snapshots(pd.to_datetime(ds.time.to_pandas()))

    # defining the buses
    network.add("Bus", "bus el", carrier="el")
    network.add("Bus", "bus hydrogen", carrier="hydrogen")
    # network.add("Bus","bus water", carrier = "water")

    # adding carriers
    network.add("Carrier", "el")
    network.add("Carrier", "hydrogen")
    network.add("Carrier", "wind")
    network.add("Carrier", "solar")
    # network.add("Carrier","water")

    # adding wind and solar generators on el bus
    interest_rate = 0.075
    wind_cost = calc_cap_cost(dc, "onwind", interest_rate)
    # offshore_wind_cost = calc_cap_cost(dc,"offwind",interest_rate)
    solar_cost = calc_cap_cost(dc, "solar-utility", interest_rate)
    battery_cost = calc_cap_cost(dc, "battery storage", interest_rate)

    # hydrogen cost can either be 0 or real cost
    if h_cost == False:
        hydrogen_storage_cost = 0
    else:
        hydrogen_storage_cost = calc_cap_cost(
            dc, "hydrogen storage tank incl. compressor", interest_rate
        )

    # for every class in solar data
    print("---------------------------- starting with solar data ")
    for i in range(0, len(ds.capacity)):
        # "time":slice("2013-01-01 00:00", "2013-01-30 14:00"),
        sol_df = ds.sel({"class": ds["class"][i]})

        # costs taken from dae: solar costs
        network.add(
            "Generator",
            "PV {}".format(i),
            bus="bus el",
            carrier="solar",
            # p_nom = 8, #leave it commented out, start cap should be zero
            p_nom_extendable=True,
            p_nom_max=sol_df["capacity"]
            .to_pandas()
            .item(),  # this will be ds.capacities
            p_max_pu=sol_df["capacity factor"].to_pandas(),  # this will be ds.profiles
            capital_cost=solar_cost[0],  # EUR/MW, this will be read in from costs file
        )

    # for every class in onshore wind data
    print("---------------------------- starting with wind data ")
    for i in range(0, len(dw.capacity)):
        wind_df = dw.sel({"class": dw["class"][i]})

        network.add(
            "Generator",
            "on_wind turbine {}".format(i),
            bus="bus el",
            carrier="wind",
            # p_nom = 8, #this is capacity
            p_nom_extendable=True,
            # p_nom_min = 8,
            p_nom_max=wind_df["capacity"].to_pandas().item(),
            p_max_pu=wind_df[
                "capacity factor"
            ].to_pandas(),  # read in from potentials file
            capital_cost=wind_cost[
                0
            ],  # EUR/MW, read in from costs file and calculated in above function
        )

    # for every class in offshore wind data
    # for i in range (0,len(dww.capacity)):
    #     offshore_wind_df = dww.sel({"class":dww["class"][i]})

    #     network.add(
    #         "Generator",
    #         "off_wind turbine {}".format(i),
    #         bus="bus el",
    #         carrier="wind",
    #         #p_nom = 8, #this is capacity
    #         p_nom_extendable=True,
    #         #p_nom_min = 8,
    #         p_nom_max = offshore_wind_df["capacity"].to_pandas(),
    #         p_max_pu= offshore_wind_df["capacity factor"].to_pandas(), #read in from potentials file
    #         capital_cost= offshore_wind_cost  #EUR/MW, read in from costs file and calculated in above function
    #     )

    # adding storage
    # battery storage investment cost = 75EUR/kWh
    network.add(
        "Store",
        "battery",
        bus="bus el",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=battery_cost[0],
    )  # EUR/MWh

    # hydrogen storage
    network.add(
        "Store",
        "hydrogen",
        bus="bus hydrogen",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=hydrogen_storage_cost,
    )  # EUR/MWh

    # p_set unit in MW
    network.add("Load", "load", bus="bus hydrogen", p_set=load)
    print("network load: ", load)

    # values for electrolysis link from cost outputs in 2050 (dae data)
    network.add(
        "Link",
        "electrolysis",
        bus0="bus el",
        bus1="bus hydrogen",
        efficiency=0.75,  # per unit
        capital_cost=calc_cap_cost(dc, "electrolysis", interest_rate)[0],  # EUR/MW
        p_nom_extendable=True,
    )
    return network


def save_lcoh(solved_network):
    # creating dataframe for saving
    res = pd.DataFrame(
        columns=[
            "demand factor [%]",
            "demand [MWh]",
            "load [MW]",
            "cost [EUR]",
            "LCOH [EUR/MWh]",
        ]
    )

    try:
        solved_network.objective
    except:
        # if infeasible
        print("saving infeasible network")
        res.loc[res.shape[0]] = [
            snakemake.wildcards["demand_factor"],
            load * 8760,
            load,
            "infeasible",
            "infeasible",
        ]
    else:
        # if feasible
        print("saving feasible network")
        res.loc[res.shape[0]] = [
            snakemake.wildcards["demand_factor"],
            load * 8760,
            load,
            solved_network.objective,
            solved_network.objective / (load * 8760),
        ]

    # saving network and dataframe
    solved_network.export_to_netcdf(snakemake.output.network)
    res.to_csv(snakemake.output.results)
    return


if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "model_lcoh", cost_year="2030", demand_factor=20, region="Europe"
        )

    # making dataframes from inputs
    dc = pd.read_csv(snakemake.input.costs, header=0)
    d = xr.open_dataset(snakemake.input.supply_data)

    # subselecting each technology and cleaning for "0 and nan" - capacity values
    ds = d.sel({"technology": "pvplant"})
    ds_cleaned = ds.where(ds.capacity > 0.0, drop=True)
    dw = d.sel({"technology": "windonshore"})
    dw_cleaned = dw.where(dw.capacity > 0.0, drop=True)
    # dww = d.sel({"technology":"windoffshore"})
    # dww_cleaned = dww.where(dww.capacity > 0.0,drop=True)

    max_load = (
        int(
            (ds_cleaned.capacity * ds_cleaned["capacity factor"]).sum(
                dim=["time", "class"]
            )
            + (dw_cleaned.capacity * dw_cleaned["capacity factor"]).sum(
                dim=["time", "class"]
            )
        )
        / 8760
        * 0.75
    )
    print("max load, (solar+wind)/8760*0.75:", max_load)

    # calculating load
    # load = int(ds_cleaned.capacity.max()+dw_cleaned.capacity.max())*(int(snakemake.wildcards['demand_factor'])/100)
    load = float(
        (
            (ds_cleaned.capacity * ds_cleaned["capacity factor"]).sum(
                dim=["time", "class"]
            )
            + (dw_cleaned.capacity * dw_cleaned["capacity factor"]).sum(
                dim=["time", "class"]
            )
            # +
            # (
            # dww_cleaned.capacity * dww_cleaned["capacity factor"]
            # ).sum(dim=["time","class"])
        )
        / 8760
        * (int(snakemake.wildcards["demand_factor"]) / 100)
    )
    print("load:", load)
    print("diff:", max_load - load)
    print("data loaded successfully")

    # building model
    print("building model")
    network = building_model(
        ds_cleaned, dw_cleaned, dc, load, snakemake.config["hydrogen_storage_cost"]
    )

    # solving model
    print("solving model")
    network.optimize(
        network.snapshots,
        solver_name="gurobi",
        solver_options={
            "crossover": 0,
            "method": 2,
            "BarConvTol": 1.0e-5,
            "OptimalityTol": 1.0e-5,
        },
    )
    # , "barHomogeneous":1, "FeasibilityTol": 1.e-5,
    print("network was solved succesfully")

    # saving results and calculating LCOH
    print("saving results and calculating lcoh")
    save_lcoh(network)
