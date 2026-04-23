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

    if tech in ["hydrogen direct iron reduction furnace", "electric arc furnace"]:
        capital_cost = (annuity + FOM / 100) * CAPEX
    else:
        capital_cost = (annuity + FOM / 100) * CAPEX * 1e3

    return capital_cost


def rename_trace_carriers(n):

    # Index name and new carrier
    carrier_rename_dict = {
        "electrolysis (exp)": "electrolysis",
        "battery inverter (charging, exp)": "battery inverter",
        "battery inverter (discharging, exp)": "battery inverter",
        "hydrogen direct iron reduction furnace": "hydrogen direct iron reduction furnace",
        "electric arc furnace": "electric arc furnace",
    }

    nice_names = {
        "electrolysis": "electrolysis",
        "battery inverter": "battery inverter",
        "hydrogen direct iron reduction furnace": "hydrogen direct iron reduction furnace",
        "electric arc furnace": "electric arc furnace",
    }
    colors = snakemake.config["colors"]

    # Deduplicate carrier values while preserving insertion order, then build
    # the parallel nice_name and color lists from the same unique sequence.
    unique_carriers = list(dict.fromkeys(carrier_rename_dict.values()))

    n.add(
        "Carrier",
        unique_carriers,
        nice_name=[nice_names[carrier] for carrier in unique_carriers],
        color=[colors[carrier] for carrier in unique_carriers],
    )

    for idx, new_carrier in carrier_rename_dict.items():
        n.links.loc[idx, "carrier"] = new_carrier

    # Adjust colors of all carriers, overwriting the TRACE colors
    for carrier in n.carriers.index:
        n.carriers.loc[carrier, "color"] = colors[carrier]

    # Add carrier to DRI generator
    n.generators.loc["iron ore DRI-ready (exp)", "carrier"] = "iron ore"

    return n


def remove_shipping_importer_components(n):
    # Remove trace shipping components
    n.remove(
        "Link",
        ["ship loading (exp)", "ship unloading (imp)"],
    )
    n.remove("Bus", ["berth (exp)", "berth (imp)", "steel (imp)"])
    n.remove("Store", ["steel storage (exp)", "steel storage (imp)"])

    return n


# inputs are solar potentials, wind potentials, costs and load
def building_model(n, region, ds, dw, dc, load, h_cost, iron_ore_cost):

    if product != "eaf-grid":

        # Country specific wacc
        base_interest_rate = snakemake.params.interest_rate

        if snakemake.wildcards.wacc == "regional":
            print(f"applying region specific wacc")
            wacc = pd.read_csv(snakemake.input.wacc, header=0)
            wacc.set_index("region", inplace=True)
            regional_wacc = wacc.loc[region].values[0]
            interest_rate = regional_wacc
            # Adjust capital_cost of all pre-loaded TRACE components to the
            # regional WACC.  Technologies not found in the costs table (e.g.
            # the 1/1000 stabiliser entries) are left untouched.
            n = adjust_trace_wacc(n, base_interest_rate, regional_wacc, dc)
        elif snakemake.wildcards.wacc == "uniform":
            interest_rate = base_interest_rate
        else:
            raise ValueError("wacc wildcard not recognized, choose 'regional' or 'uniform'")

        # adding wind and solar generators on el bus
        wind_cost = calc_cap_cost(dc, "onwind", interest_rate)
        # offshore_wind_cost = calc_cap_cost(dc,"offwind",interest_rate)
        solar_cost = calc_cap_cost(dc, "solar-utility", interest_rate)

        # for every class in solar data
        print("---------------------------- starting with solar data ")
        for i in range(0, len(ds.capacity)):
            # "time":slice("2013-01-01 00:00", "2013-01-30 14:00"),
            sol_df = ds.sel({"class": ds["class"][i]})

            # costs taken from dae: solar costs
            n.add(
                "Generator",
                "pv {}".format(i),
                bus="electricity (exp)",
                carrier="pv",
                p_nom_extendable=True,
                p_nom_max=sol_df["capacity"].to_pandas().item()
                * pv_p_nom_max_cor,  # this will be ds.capacities
                p_max_pu=sol_df["capacity factor"]
                .to_pandas()
                .clip(lower=0),  # this will be ds.profiles
                capital_cost=solar_cost[
                    0
                ],  # EUR/MW, this will be read in from costs file
            )

        # for every class in onshore wind data
        print("---------------------------- starting with wind data ")
        for i in range(0, len(dw.capacity)):
            wind_df = dw.sel({"class": dw["class"][i]})

            n.add(
                "Generator",
                "onwind {}".format(i),
                bus="electricity (exp)",
                carrier="wind",
                p_nom_extendable=True,
                p_nom_max=wind_df["capacity"].to_pandas().item() * onwind_p_nom_max_cor,
                p_max_pu=wind_df["capacity factor"]
                .to_pandas()
                .clip(lower=0),  # read in from potentials file
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

    elif product == "eaf-grid":
        # adding only electricity grid on el bus for eaf-grid case
        n.add(
            "Generator",
            "grid-electricity",
            bus="electricity (exp)",
            carrier="electricity",
            p_nom_extendable=True,
            p_nom_max=np.inf,
            capital_cost=snakemake.config["grid_electricity"]["capital_cost"],  # EUR/MW
            marginal_cost=snakemake.config["grid_electricity"][
                "marginal_cost"
            ],  # EUR/MW
        )

    else:
        raise ValueError("product not recognized, choose steel, hbi, eaf, eaf-grid")

    # hydrogen cost can either be 0 or real cost. Real cost is the default of the imported network
    if h_cost == False:
        n.stores.at[
            "hydrogen storage tank type 1 including compressor (exp)", "capital_cost"
        ] = 0
    else:
        pass

    if iron_ore_cost == False:
        n.generators.at["iron ore DRI-ready (exp)", "marginal_cost"] = 0
    else:
        pass

    # Remove trace shipping components
    n = remove_shipping_importer_components(n)

    if product == "steel":

        # p_set unit in MW
        n.add("Load", "load", bus="steel (exp)", carrier="steel", p_set=load)

    elif product == "hbi":

        # Remove steel components from the network
        n.remove(
            "Link",
            ["electric arc furnace"],
        )
        n.remove("Bus", ["steel (exp)"])
        n.remove("Carrier", ["steel", "electric arc furnace"])

        # p_set unit in MW
        n.add(
            "Load",
            "load",
            bus="hot briquetted iron (exp)",
            carrier="hot briquetted iron",
            p_set=load,
        )

    elif product in ["eaf", "eaf-grid"]:

        # Remove components up to hbi and leave eaf/steel components
        n.remove(
            "Link",
            ["electrolysis (exp)", "hydrogen direct iron reduction furnace"],
        )
        n.remove(
            "Bus",
            ["hydrogen (g) (exp)", "hydrogen (g) storage (exp)", "iron ore (exp)"],
        )
        n.remove(
            "Carrier",
            [
                "hydrogen",
                "iron ore",
                "electrolysis",
                "hydrogen direct iron reduction furnace",
            ],
        )
        n.remove(
            "Store",
            [
                "hydrogen storage tank type 1 including compressor (exp)",
                "HBI storage (exp)",
            ],
        )

        n.remove("Generator", ["iron ore DRI-ready (exp)"])

        # Add Generator as HBI input  (at no cost)
        n.add(
            "Generator",
            "hbi input",
            bus="hot briquetted iron (exp)",
            carrier="hot briquetted iron",
            p_nom_extendable=True,
            capital_cost=0.1,
            marginal_cost=0.1,
        )

        # p_set unit in MW
        n.add("Load", "load", bus="steel (exp)", carrier="steel", p_set=load)

    else:
        raise ValueError("product not recognized, choose steel, hbi, eaf, eaf-grid")

    print("network load: ", load)

    return n


def save_lcox(solved_network):
    # creating dataframe for saving
    res = pd.DataFrame(
        columns=[
            "demand factor [%]",
            "demand [t]",
            "load [t/h]",
            "cost [EUR]",
            "LCOX [EUR/t]",
        ]
    )

    try:
        obj = solved_network.objective
        if obj is None:
            raise AttributeError
    except AttributeError:
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


def solve_network(n):

    solver_name = snakemake.config["solver"]["name"]
    options = snakemake.config["solver_options"][snakemake.config["solver"]["options"]]

    print("solving model")
    n.optimize(n.snapshots, solver_name=solver_name, solver_options=options)
    # , "barHomogeneous":1, "FeasibilityTol": 1.e-5,
    print("network was solved succesfully")

    return n


def prepare_re(d):

    # subselecting each technology and cleaning for "0 and nan" - capacity values
    ds = d.sel({"technology": "pvplant"})
    ds_cleaned = ds.where(ds.capacity > 0.0, drop=True)
    dw = d.sel({"technology": "windonshore"})
    dw_cleaned = dw.where(dw.capacity > 0.0, drop=True)
    # dww = d.sel({"technology":"windoffshore"})
    # dww_cleaned = dww.where(dww.capacity > 0.0,drop=True)

    return ds_cleaned, dw_cleaned


def calculate_load(ds_cleaned, dw_cleaned, pv_p_nom_max_cor, onwind_p_nom_max_cor):

    # calculating (max) load
    max_load = (
        int(
            (
                ds_cleaned.capacity * pv_p_nom_max_cor * ds_cleaned["capacity factor"]
            ).sum(dim=["time", "class"])
            + (
                dw_cleaned.capacity
                * onwind_p_nom_max_cor
                * dw_cleaned["capacity factor"]
            ).sum(dim=["time", "class"])
            # +
            # (
            # dww_cleaned.capacity * dww_cleaned["capacity factor"]
            # ).sum(dim=["time","class"])
        )
        / 8760
        / snakemake.config["electricity_steel_ratio"]
    )

    load = max_load * (float(snakemake.wildcards["demand_factor"]) / 100)

    print(
        f"max load hydrogen, (solar+onwind corrected)/{snakemake.config["electricity_steel_ratio"]}: {max_load:.1f}"
    )
    print(f"load steel with demand factor: {load:.1f}")

    return load


def adjust_trace_wacc(n, base_interest_rate, regional_wacc, costs):
    # TODO This is only applied to links
    """
    Rescale the capital_cost of every PyPSA component in *n* from
    base_interest_rate to regional_wacc.

    For each component whose carrier exactly matches a technology entry in the
    costs dataframe, capital_cost is fully recalculated with the new rate
    (CAPEX, FOM and lifetime are looked up from costs, identical to how
    calc_cap_cost works for wind/solar).

    Components whose carrier is not found in costs (e.g. the placeholder
    1/1000 stabiliser costs, or iron-ore generators) are left unchanged.
    Ensure carrier names are aligned with the costs technology column via
    rename_trace_carriers() before calling this function.

    Parameters
    ----------
    n : pypsa.Network
    base_interest_rate : float      – rate used when the TRACE network was built
    regional_wacc      : float      – new, region-specific rate to apply
    costs              : pd.DataFrame – technology costs table (same ``dc``)

    Returns
    -------
    n : pypsa.Network  (modified in place and returned for convenience)
    """
    if base_interest_rate == regional_wacc:
        print("adjust_wacc: base and regional rate are identical – skipping.")
        return n

    available_techs = set(costs["technology"].unique())

    component_frames = [
        # ("Generator", n.generators),
        ("Link", n.links),
        # ("Store", n.stores),
        # ("StorageUnit", n.storage_units),
    ]

    for comp_type, df in component_frames:
        if df.empty:
            continue
        for idx in df.index:
            carrier = df.at[idx, "carrier"]
            if carrier not in available_techs:
                print("Not found in costs, skipping: ", idx)
                continue
            old = df.at[idx, "capital_cost"]
            new = calc_cap_cost(costs, carrier, regional_wacc)
            new = float(new.flat[0]) if hasattr(new, "__len__") else float(new)
            df.at[idx, "capital_cost"] = new
            print(
                f"  adjust_wacc [{comp_type}] '{idx}' (carrier='{carrier}'): "
                f"capital_cost {old:.2f} → {new:.2f} EUR/MW  "
                f"(i {base_interest_rate:.4f} → {regional_wacc:.4f})"
            )

    return n


def adjust_part_load(n):

    print(f"adjusting part-load limits for {snakemake.config["part_load"].keys()}")

    for carrier in snakemake.config["part_load"].keys():

        n.links.loc[
            n.links.carrier == carrier,
            "p_min_pu",
        ] = snakemake.config[
            "part_load"
        ][carrier]

    return n


if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "model_lcox",
            cost_year="2050",
            region="Middle_East",
            product="eaf-grid",
            demand_factor=10,
            wacc="regional",
        )

    # making dataframes from inputs
    dc = pd.read_csv(snakemake.input.costs, header=0)
    d = xr.open_dataset(snakemake.input.supply_data)

    # load TRACE steel model
    n = pypsa.Network(snakemake.input.trace)
    n = rename_trace_carriers(n)
    n = adjust_part_load(n)

    # Get correction factors and product
    pv_p_nom_max_cor = snakemake.config["pv_p_nom_max_cor"]
    onwind_p_nom_max_cor = snakemake.config["onwind_p_nom_max_cor"]
    product = snakemake.wildcards.product

    # preparing RE data
    ds_cleaned, dw_cleaned = prepare_re(d)

    # calculating load
    load = calculate_load(
        ds_cleaned, dw_cleaned, pv_p_nom_max_cor, onwind_p_nom_max_cor
    )

    # building model
    print("adding RE to network")
    n = building_model(
        n,
        snakemake.wildcards.region,
        ds_cleaned,
        dw_cleaned,
        dc,
        load,
        snakemake.config["hydrogen_storage_cost"],
        snakemake.config["iron_ore_cost_in_supply_chain"],
    )

    # solving model
    n = solve_network(n)

    # saving results and calculating LCOX
    print("saving results and calculating lcoX")
    save_lcox(n)
