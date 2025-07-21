import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use("bmh")


# inputs are transportation costs, supply curves, trade options and load demand for all regions
def building_model(supply_curves, demands, bus_location):
    # this function creates network, carrier and a bus for each region
    # with a load and all supply possibilities added

    # create network
    network = pypsa.Network()

    # adding carriers
    network.add("Carrier", "hydrogen")

    c = 0
    # for each region we are creating a bus with all the potentials and load
    for r in range(0, len(supply_curves)):

        # getting the supply curve for one region
        region_file = supply_curves[c]
        region_data = pd.read_csv(region_file, header=0)
        filename = os.path.basename(region_file)
        # Extract region name
        region_name = filename.split("_hydrogen")[0]

        print("building generators and loads for ", region_name)

        # defining the bus with region name
        network.add(
            "Bus",
            "bus {}".format(region_name),
            carrier="hydrogen",
            x=float(
                bus_location.loc[bus_location["region_name"] == region_name]["long"]
            ),  # long
            y=float(
                bus_location.loc[bus_location["region_name"] == region_name]["lat"]
            ),  # lat
        )

        # defining the demand for the region
        load = int(demands.loc[demands["region"] == region_name]["demand"]) * float(
            snakemake.wildcards["demand"]
        )
        print(
            f"Load set via snakemake.wildcard to {float(snakemake.wildcards['demand'])*100}% of regional final energy demand."
        )

        network.add(
            "Load",
            "load {}".format(region_name),
            bus="bus {}".format(region_name),
            p_set=load,
        )

        # defining the supply opportunities for the region (apart from last supply as that is the 75% infeasible one)
        for s in range(0, len(region_data) - 1):
            if s == 0:
                p_nom_supply = float(region_data["demand [MWh]"][s])
            else:
                p_nom_supply = float(region_data["demand [MWh]"][s]) - float(
                    region_data["demand [MWh]"][s - 1]
                )
            M_cost_supply = float(region_data["LCOH [EUR/MWh]"][s])

            network.add(
                "Generator",
                "hydrogen supply {}_{}".format(
                    region_name, region_data["demand factor [%]"][s]
                ),
                bus="bus {}".format(region_name),
                carrier="hydrogen",
                p_nom_extendable=True,
                p_nom_max=p_nom_supply,  # MWh, demand = potential supply
                marginal_cost=M_cost_supply,  # EUR/MWh
                capital_cost=1 / 1000,  # to prevent optimisation shennanigans
            )

        # update counter for next region
        c += 1

    return network


def create_links(transport_costs, trade_options):

    # for in range of length of input csv with all the different links, region_from = column , region_to = column 2
    # create links with the correct corresponding costs

    # marginal and fixed cost for the different type of transport
    ship_mc = float(
        transport_costs.loc[transport_costs["transport_type"] == "shipping"][
            "marginal_cost"
        ]
    )
    pipe_mc = float(
        transport_costs.loc[transport_costs["transport_type"] == "pipeline"][
            "marginal_cost"
        ]
    )
    ship_c = float(
        transport_costs.loc[transport_costs["transport_type"] == "shipping"][
            "fixed_cost"
        ]
    )
    input_demand = 0.42  # MWh/km for LH2, IEA future of hydrogen 2019
    boat_capacity = 363000  # MWh for LH2, IEA future of hydrogen 2019
    speed = 30  # km/h, IEA future of hydrogen 2019
    BOG = 0.2 / 100  # %/day, IEA future of hydrogen 2019

    print("ship + pipe cost", ship_mc, ship_c, pipe_mc)

    # if there should be a link, create a link
    # do this for both shipping and pipeline
    for r in range(0, len(trade_options)):
        # checking if the row connects with shipping
        if trade_options["shipping"][r] == 1:
            r_from = trade_options["region_from"][r]
            r_to = trade_options["region_to"][r]
            total_cost = ship_c + int(
                float(trade_options["shipping_distance [km]"][r]) * ship_mc
            )

            # calculating efficiency
            days_at_sea = (
                float(trade_options["shipping_distance [km]"][r]) / speed
            ) / 24
            tot_BOG = 1 - (1 - BOG) ** days_at_sea
            tot_fuel_demand = (
                (2 * float(trade_options["shipping_distance [km]"][r]))
                * input_demand
                / boat_capacity
            )
            eff = 1 - max(tot_BOG, tot_fuel_demand)

            network.add(
                "Link",
                "shipping {}-{}".format(r_from, r_to),
                bus0="bus {}".format(r_from),
                bus1="bus {}".format(r_to),
                efficiency=eff,  # %, calculated above
                marginal_cost=total_cost,  # EUR/MWh
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
                p_nom_extendable=True,
            )
            print("shipping link made from {} to {} - eff {}".format(r_from, r_to, eff))

        # checking if the row connects with pipeline
        if trade_options["pipeline"][r] == 1:
            r_from = trade_options["region_from"][r]
            r_to = trade_options["region_to"][r]
            p_cost = int(float(trade_options["pipeline_distance [km]"][r]) * pipe_mc)
            filling_demand = 1.5 / 100  # DEA, energy transport datasheet, 2050, %
            losses = 1.7 / 100  # DEA, energy transport datasheet 2022, 2050, %/1000km
            eff = (1 - filling_demand) * (1 - losses) ** (
                float(trade_options["pipeline_distance [km]"][r]) / 1000
            )

            network.add(
                "Link",
                "pipeline {}-{}".format(r_from, r_to),
                bus0="bus {}".format(r_from),
                bus1="bus {}".format(r_to),
                efficiency=eff,  # calculated above
                marginal_cost=p_cost,  # EUR/MWh
                p_nom_extendable=True,
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
            )
            print("pipeline link made from {} to {} - eff {}".format(r_from, r_to, eff))

    return


def save_trade_network(solved_network):

    sol = pd.DataFrame(columns=["type", "variable", "value", "unit"])
    # add objective cost
    sol.loc[sol.shape[0]] = ["objective", "cost", solved_network.objective, "EUR"]
    print("added objective cost to sol")

    # add all generators with name and production value
    df_gen = solved_network.generators.p_nom_opt.T.to_frame()
    df_gen.reset_index(inplace=True)
    df_gen = df_gen.rename(columns={"Generator": "variable", "now": "value"})
    df_gen.insert(0, "type", "generator")
    df_gen.insert(3, "unit", "MWh")
    sol = pd.concat([sol, df_gen], ignore_index=True)
    print("added generators to sol")

    # add all links with names and flows
    df_links = solved_network.links.p_nom_opt.T.to_frame()
    df_links.reset_index(inplace=True)
    df_links = df_links.rename(columns={"Link": "variable", "p_nom_opt": "value"})
    df_links.insert(0, "type", "link")
    df_links.insert(3, "unit", "MWh")
    sol = pd.concat([sol, df_links], ignore_index=True)
    print("added links to sol")

    # add all bus (balance) - who is importing/exporting
    df_bus = solved_network.buses_t.p.T
    df_bus.reset_index(inplace=True)
    df_bus = df_bus.rename(columns={"Bus": "variable", "now": "value"})
    df_bus.insert(0, "type", "bus")
    df_bus.insert(3, "unit", "MW")
    sol = pd.concat([sol, df_bus], ignore_index=True)
    print("added bus_balances to sol")

    # how much of capacity is actually being used per bus?
    df_bus_cap = (
        (
            solved_network.generators.groupby(["bus"]).p_nom_opt.sum()
            / solved_network.generators.groupby(["bus"]).p_nom_max.sum()
        )
        * 100
    ).to_frame()
    df_bus_cap.reset_index(inplace=True)
    df_bus_cap = df_bus_cap.rename(
        columns={df_bus_cap.columns[0]: "variable", df_bus_cap.columns[1]: "value"}
    )
    df_bus_cap.insert(0, "type", "used bus capacity")
    df_bus_cap.insert(3, "unit", "%")
    sol = pd.concat([sol, df_bus_cap], ignore_index=True)
    print("added bus_capacities to sol")

    sol.to_csv(snakemake.output.trade_result)
    network.export_to_netcdf(snakemake.output.trade_network)

    return


def plot_trade_network(n):
    # creating color dataframe for type of transportation method
    df_link = n.links.type.astype(str).to_frame()
    df_link.reset_index(inplace=True)
    df_link = df_link.rename(
        columns={df_link.columns[0]: "Link", df_link.columns[1]: "color"}
    )
    # setting all as default to green
    df_link["color"] = "lightgreen"
    # shipping links are changed to blue
    df_link.loc[df_link["Link"].str.contains("shipping"), "color"] = "skyblue"
    df_link.set_index("Link", inplace=True)

    # creating figure
    fig = plt.figure()
    region_gen = n.generators.groupby(["bus"]).p_nom_opt.sum()
    region_load = n.loads.groupby(["bus"]).p_set.sum()
    link_flow = n.links.p_nom_opt.astype(int)
    n.plot(
        bus_sizes=1e-8 * region_load,
        bus_colors="seagreen",
        bus_alpha=1,
        link_widths=0,
        branch_components=["Link"],
    )  # the load at bus in green
    n.plot(
        bus_sizes=1e-8 * region_gen,
        bus_colors="lightsteelblue",
        bus_alpha=0.7,
        link_widths=1e-9 * link_flow,
        branch_components=["Link"],
        link_colors=df_link["color"],
    )  # the gen at bus in light blue

    legend_elements = [
        plt.Line2D([0], [0], color="lightgreen", label="pipeline"),
        plt.Line2D([0], [0], color="lightblue", label="shipping"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="load",
            markerfacecolor="seagreen",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="generation",
            markerfacecolor="lightsteelblue",
            markersize=10,
        ),
    ]
    fig.legend(handles=legend_elements, frameon=False)

    # fig.suptitle("scenario:{}-{}-{}".format(snakemake.wildcards["cost_year"],snakemake.wildcards["transport_cost"],snakemake.wildcards["demand"]))
    fig.savefig(snakemake.output.trade_plot, format="pdf")
    return


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "model_trade", transport_cost="irena", cost_year="2030", demand=0.2
        )

    print("starting up with all regions--- ")
    # making dataframes
    transport_costs = pd.read_csv(snakemake.input.transport_costs, header=0)
    trade_options = pd.read_csv(snakemake.input.trade_options, header=0)
    supply_curves = snakemake.input.supply_curves
    bus_locations = pd.read_csv(snakemake.input.bus_locations, header=0)
    loads = pd.read_csv(snakemake.input.demand)
    print("data loaded successfully")

    # building model
    print("building model")
    network = building_model(supply_curves, loads, bus_locations)

    # building transport network connecting the individual buses
    print("building transportation links")
    create_links(transport_costs, trade_options)

    # solving model
    print("solving model")
    network.optimize(
        network.snapshots,
        solver_name="gurobi",
        solver_options={
            "crossover": 0,
            "method": 2,
            "BarConvTol": 1.0e-5,
            "FeasibilityTol": 1.0e-5,
            "OptimalityTol": 1.0e-5,
            "barHomogeneous": 1,
        },
    )
    print("network was solved")

    # saving results and calculating LCOH
    print("saving results as network+csv and pdf")
    save_trade_network(network)
    plot_trade_network(network)
