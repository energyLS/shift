import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use("bmh")


# inputs are transportation costs, supply curves, trade options and load demand for all regions
def building_model(supply_curves, demands, bus_location, product):
    # this function creates network, carrier and a bus for each region
    # with a load and all supply possibilities added

    # create network
    network = pypsa.Network()

    # adding carriers
    network.add(
        "Carrier", name=product, color=snakemake.config["plot"]["colors"][product]
    )

    # for each region we are creating a bus with all the potentials and load
    for r in range(0, len(supply_curves)):

        # getting the supply curve for one region
        region_file = supply_curves[r]
        region_data = pd.read_csv(region_file, header=0)
        filename = os.path.basename(region_file)
        # Extract region name
        region_name = filename.split("_" + product)[0]

        print("building generators and loads for ", region_name)

        # define the steel bus with region name
        network.add(
            "Bus",
            region_name,
            carrier=product,
            x=float(
                bus_location.loc[bus_location["region_name"] == region_name]["long"]
            ),  # long
            y=float(
                bus_location.loc[bus_location["region_name"] == region_name]["lat"]
            ),  # lat
        )

        # Define the iron ore carrier
        network.add(
            "Carrier",
            name="iron_ore",
            color=snakemake.config["plot"]["colors"]["iron_ore"],
        )

        network.add(
            "Carrier",
            name="shipping_iron_ore",
            color=snakemake.config["plot"]["colors"]["iron_ore_shipping"],
        )

        # Define the steel shipping carrier
        network.add(
            "Carrier",
            name="shipping_" + product,
            color=snakemake.config["plot"]["colors"][product + "_shipping"],
        )

        # define the iron ore bus with region name
        network.add(
            "Bus",
            region_name + "_ore",
            carrier="iron_ore",
            x=float(
                bus_location.loc[bus_location["region_name"] == region_name]["long"]
            ),  # long
            y=float(
                bus_location.loc[bus_location["region_name"] == region_name]["lat"]
            ),  # lat
        )

        # Define iron ore generators feeding iron ore buses in each region
        iron_ore_limit = (
            iron_ore.loc[iron_ore["region"] == region_name][
                "IronOreProductionMt"
            ].values[0]
            * 1e6
            * snakemake.config["iron_ore"]["potential_allowance"]
        )  # Limit in t_ore

        network.add(
            "Generator",
            "{}_ore".format(region_name),
            bus=region_name + "_ore",
            carrier="iron_ore",
            p_nom_extendable=True,
            p_nom_max=iron_ore_limit,  # t_ore
            marginal_cost=snakemake.config["iron_ore"]["marginal_cost"],  # EUR/t_ore
            capital_cost=1 / 1000,  # to prevent optimisation shenanigans
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
            region_name,
            bus=region_name,
            p_set=load,
        )

        # defining the supply opportunities for the region (apart from last supply as that is the 75% infeasible one)
        for s in range(0, len(region_data) - 1):
            if s == 0:
                p_nom_supply = float(region_data[f"demand [{unit}]"][s])
            else:
                p_nom_supply = float(region_data[f"demand [{unit}]"][s]) - float(
                    region_data[f"demand [{unit}]"][s - 1]
                )
            M_cost_supply = float(region_data[f"{cost_descriptor} [EUR/{unit}]"][s])

            if product == "hydrogen":
                network.add(
                    "Generator",
                    "{} supply {}_{}".format(
                        product, region_name, region_data["demand factor [%]"][s]
                    ),
                    bus=region_name,
                    carrier=product,
                    p_nom_extendable=True,
                    p_nom_max=p_nom_supply,  # MWh or t, demand = potential supply
                    marginal_cost=M_cost_supply,  # EUR/MWh or EUR/t
                    capital_cost=1 / 1000,  # to prevent optimisation shennanigans
                )

            elif product == "steel":
                network.add(
                    "Link",
                    "{} supply {}_{}".format(
                        product, region_name, region_data["demand factor [%]"][s]
                    ),
                    bus0=region_name + "_ore",
                    bus1=region_name,
                    carrier=product,
                    p_nom_max=p_nom_supply
                    * snakemake.config["iron_ore"][
                        "ore_to_steel_ratio"
                    ],  # t, demand = potential supply
                    p_nom_extendable=True,
                    efficiency=1 / snakemake.config["iron_ore"]["ore_to_steel_ratio"],
                    marginal_cost=M_cost_supply
                    / snakemake.config["iron_ore"][
                        "ore_to_steel_ratio"
                    ],  # Note: marginal_cost are referred to bus0, hence we need to consider efficiency to apply €/t_steel value
                    capital_cost=1 / 1000,  # to prevent optimisation shenanigans
                )
            else:
                raise ValueError("Product must be either 'steel' or 'hydrogen'.")

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


    ship_steel_mc = (
        transport_costs.loc[transport_costs["transport_type"] == "shipping_steel"][
            "marginal_cost"
        ].values[0]
    )

    ship_iron_ore_mc = (
        transport_costs.loc[transport_costs["transport_type"] == "shipping_iron_ore"][
            "marginal_cost"
        ].values[0]
    )


    input_demand = 0.42  # MWh/km for LH2, IEA future of hydrogen 2019
    boat_capacity = 363000  # MWh for LH2, IEA future of hydrogen 2019
    speed = 30  # km/h, IEA future of hydrogen 2019
    BOG = 0.2 / 100  # %/day, IEA future of hydrogen 2019

    print("ship + pipe cost", ship_mc, ship_c, pipe_mc)
    print(f"shipping cost steel {ship_steel_mc} EUR/(t*km)")
    print(f"shipping cost iron ore {ship_iron_ore_mc} EUR/(t*km)")


    # if there should be a link, create a link
    # do this for both shipping and pipeline
    for r in range(0, len(trade_options)):
        # checking if the row connects with shipping
        if trade_options["shipping"][r] == 1:
            r_from = trade_options["region_from"][r]
            r_to = trade_options["region_to"][r]

            # If shipping costs are made up from marginal and capital
            # total_cost = ship_c + int(
            #     float(trade_options["shipping_distance [km]"][r]) * ship_mc
            # )
            # If shipping costs are made up from marginal only
            total_cost_steel = ship_steel_mc * float(trade_options["shipping_distance [km]"][r]) 
            total_cost = total_cost_steel

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
                carrier="shipping_" + product,
                bus0=r_from,
                bus1=r_to,
                efficiency=eff,  # %, calculated above
                marginal_cost=total_cost,  # EUR/MWh or EUR/t
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
                p_nom_extendable=True,
            )
            print("shipping link made from {} to {} - eff {}".format(r_from, r_to, eff))

            # Add iron ore shipping link
            total_cost_iron_ore = ship_iron_ore_mc * float(
                trade_options["shipping_distance [km]"][r]
            ) # TODO Capital cost are not separate but included

            network.add(
                "Link",
                "iron ore shipping {}-{}".format(r_from, r_to),
                carrier="shipping_iron_ore",
                bus0=r_from + "_ore",
                bus1=r_to + "_ore",
                efficiency=1,
                marginal_cost=total_cost_iron_ore,  # EUR/t_ironore
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
                p_nom_extendable=True,
            )
            print(
                "iron ore shipping link made from {}_ore to {}_ore - eff {}".format(
                    r_from, r_to, eff
                )
            )

        # checking if the row connects with pipeline
        if (trade_options["pipeline"][r] == 1) & (product == "hydrogen"):
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
                bus0=r_from,
                bus1=r_to,
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
    df_gen.insert(3, "unit", unit)
    sol = pd.concat([sol, df_gen], ignore_index=True)
    print("added generators to sol")

    # add all links with names and flows
    df_links = solved_network.links.p_nom_opt.T.to_frame()
    df_links.reset_index(inplace=True)
    df_links = df_links.rename(columns={"Link": "variable", "p_nom_opt": "value"})
    df_links.insert(0, "type", "link")
    df_links.insert(3, "unit", unit)
    sol = pd.concat([sol, df_links], ignore_index=True)
    print("added links to sol")

    # add all bus (balance) - who is importing/exporting
    df_bus = solved_network.buses_t.p.T
    df_bus.reset_index(inplace=True)
    df_bus = df_bus.rename(columns={"Bus": "variable", "now": "value"})
    df_bus.insert(0, "type", "bus")
    df_bus.insert(3, "unit", unit + "/a")
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


def plot_trade_network(n, product="steel", alpha_supply=0.7, alpha_demand=1, output_path=None):
    """
    Plot trade network using config-driven colors and sizes.
    Calculates supply, demand, and trade according to product type.
    """
    config = snakemake.config
    plot_config = config["plot"]["world_map"][product]
    colors = config["plot"]["colors"]
    supply_color = colors.get(f"{product}_supply", "black")
    demand_color = colors.get(f"{product}_demand", "lightsteelblue")
    link_colors = colors.get(f"{product}_link", "gray")

    fig = plt.figure(figsize=(10, 5))

    # Calculate supply, demand, trade according to product
    if product == "steel":
        supply = n.statistics.supply(comps=["Link"], groupby=["bus", "carrier"]).loc[:, :, "steel"].droplevel(0)
        demand = n.loads.groupby("bus").p_set.sum()
        trade = n.links[n.links.carrier == "shipping_steel"].p_nom_opt.astype(int)
    elif product == "iron_ore":
        supply = n.statistics.supply(comps=["Generator"], groupby=["bus", "carrier"]).loc[:, :, "iron_ore"].droplevel(0)
        demand = n.statistics.withdrawal(comps=["Link"], groupby=["bus", "carrier"]).loc[:, :, "steel"].droplevel(0)
        trade = n.links[n.links.carrier == "shipping_iron_ore"].p_nom_opt.astype(int)
    else:
        raise ValueError("Unsupported product for plotting.")

    # Plot demand
    n.plot.map(
        bus_sizes=demand * plot_config["bus_size"],
        bus_colors=demand_color,
        bus_alpha=alpha_demand,
        link_widths=0,
        branch_components=["Link"],
    )

    # Plot supply
    n.plot.map(
        bus_sizes=supply * plot_config["bus_size"],
        bus_colors=supply_color,
        bus_alpha=alpha_supply,
        link_widths=trade * plot_config["link_width"],
        branch_components=["Link"],
        link_colors=link_colors,
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=link_colors, label="shipping"),
        plt.Line2D([0], [0], marker="o", color="white", label="Demand",
                   markerfacecolor=demand_color, markersize=10),
        plt.Line2D([0], [0], marker="o", color="white", label="Supply",
                   markerfacecolor=supply_color, markersize=10),
    ]
    fig.legend(
        handles=legend_elements,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.22, 0.28),
    )

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    return


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "model_trade",
            transport_cost="custom",
            cost_year="2030",
            demand=1,
            product="steel",
        )

    product = snakemake.wildcards["product"]

    print("starting up with all regions--- ")
    # making dataframes
    transport_costs = pd.read_csv(snakemake.input.transport_costs, header=0)
    trade_options = pd.read_csv(snakemake.input.trade_options, header=0)
    supply_curves = snakemake.input.supply_curves
    bus_locations = pd.read_csv(snakemake.input.bus_locations, header=0)
    iron_ore = pd.read_csv(snakemake.input.iron_ore, header=0)

    if product == "steel":

        demands = pd.read_csv(snakemake.input.steel_demand, header=0)
        demands.rename(columns={"SteelProductionMt": "demand"}, inplace=True)
        demands["demand"] = demands["demand"] * 1e6  # Mt to t
        unit = "t"
        cost_descriptor = "LCOS"

    elif product == "hydrogen":

        demands = pd.read_csv(snakemake.input.demand, header=0)
        unit = "MWh"
        cost_descriptor = "LCOH"
    else:
        raise ValueError("Product must be either 'steel' or 'hydrogen'.")

    plot_config = snakemake.config["plot"]["world_map"][product]

    print("data loaded successfully")

    # building model
    print("building model")
    network = building_model(supply_curves, demands, bus_locations, product)

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
    # Plot steel map
    plot_trade_network(network, product="steel", alpha_supply=0.7, output_path=snakemake.output.trade_plot_steel)
    # Plot iron ore map
    plot_trade_network(network, product="iron_ore", alpha_supply=0.5, output_path=snakemake.output.trade_plot_ironore)
