from typing import Any

import pypsa
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from _helpers import setup_logging

snakemake: Any = globals().get("snakemake")

logger = setup_logging(__name__, snakemake=snakemake, log_filename="model_trade.log")

plt.style.use("bmh")


def build_region_geodataframe(config):
    """
    Build a dissolved GeoDataFrame of model regions from the config country lists.

    Each model region (e.g. "Europe", "Middle_East") is formed by dissolving its
    member countries (specified as ISO 3166-1 alpha-3 codes) from the NaturalEarth
    110m dataset, so only region borders are visible in the map — not internal
    country borders.
    """

    regions = config["regions"]

    # Build ISO A3 -> region mapping directly from config
    iso_to_region = {
        code.strip(): region
        for region, countries in regions.items()
        for code in countries
    }

    # Load NaturalEarth 110m countries shapefile
    reader = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    world = gpd.read_file(reader)

    # Assign region and dissolve to remove internal country borders
    world["region"] = world["ISO_A3_EH"].map(iso_to_region)
    region_gdf = world.dropna(subset=["region"]).dissolve(by="region").reset_index()

    return region_gdf


# inputs are transportation costs, supply curves, trade options and load demand for all regions
def building_model(
    supply_curves_interone, supply_curves_intertwo, demands, bus_location, trade_chain
):
    """
    Build the PyPSA network from the trade_chain config.

    Network structure (buses, carriers, links) is derived entirely from
    trade_chain["stages"] and trade_chain["tradeable_commodities"] rather
    than from wildcards.
    """
    # --- Derive model structure from trade_chain config ---
    final = trade_chain["final_product"]
    tradeable = trade_chain["tradeable_commodities"]
    stages = trade_chain["stages"]

    # is_material_chain: iron ore is involved (steel-type chain)
    # otherwise: pure energy chain (e.g. hydrogen)
    is_material_chain = "iron_ore" in tradeable

    # interone: the first non-iron_ore tradeable intermediate (e.g. "hbi")
    interone_candidates = [c for c in tradeable if c != "iron_ore"]
    interone = interone_candidates[0] if interone_candidates else final

    # two_stage: a separate intermediate bus exists between ore and final product
    two_stage = is_material_chain and (interone != final)

    # intertwo: process label of the last production stage (e.g. "eaf-grid")
    intertwo = stages[max(stages.keys())]["process_label"]

    # --- Create network ---
    n = pypsa.Network()

    # Add carriers
    n.add("Carrier", name=final, color=snakemake.config["colors"][final])
    if is_material_chain:
        n.add("Carrier", name="iron_ore", color=snakemake.config["colors"]["iron_ore"])
        n.add("Carrier", name=interone, color=snakemake.config["colors"][interone])
        n.add(
            "Carrier",
            name="shipping_" + shipping_first,
            color=snakemake.config["colors"][shipping_first + "_shipping"],
        )
        n.add(
            "Carrier",
            name="shipping_" + shipping_second,
            color=snakemake.config["colors"][shipping_second + "_shipping"],
        )

    # --- Build buses, generators, and loads per region ---
    for r in range(len(supply_curves_interone)):
        region_file_interone = supply_curves_interone[r]
        region_file_intertwo = supply_curves_intertwo[r]
        region_data_interone = (
            pd.read_csv(region_file_interone, header=0)
            .dropna(subset=[f"{cost_descriptor} [EUR/{unit}]"])
            .reset_index(drop=True)
        )  # Filter out rows where "lcox [EUR/t]" is NaN
        region_data_intertwo = (
            pd.read_csv(region_file_intertwo, header=0)
            .dropna(subset=[f"{cost_descriptor} [EUR/{unit}]"])
            .reset_index(drop=True)
        )  # Filter out rows where "lcox [EUR/t]" is NaN

        region_name = os.path.basename(region_file_interone).rsplit(
            "_marginal_cost_", 1
        )[0]

        logger.info(f"building generators and loads for {region_name}")

        # Bus coordinates (shared by all buses in this region)
        loc = bus_location.loc[bus_location["region_name"] == region_name]
        x = float(loc["long"].values[0])
        y = float(loc["lat"].values[0])

        if is_material_chain:
            # Iron ore bus
            n.add("Bus", region_name + "_ore", carrier="iron_ore", x=x, y=y)
            # Intermediate (interone) bus
            n.add("Bus", region_name + "_" + interone, carrier=interone, x=x, y=y)

        # Final product bus (always present; also covers the interone=final single-stage case)
        if not is_material_chain or two_stage:
            n.add("Bus", region_name + "_" + final, carrier=final, x=x, y=y)

        if is_material_chain:
            # Iron ore generator
            iron_ore_limit = (
                iron_ore.loc[iron_ore["region"] == region_name][
                    "IronOreProductionMt"
                ].values[0]
                * 1e6
                * snakemake.config["iron_ore"]["potential_allowance"]
            )  # Limit in t_ore

            if regionalise == "grade-dependent":
                iron_ore_cost = iron_ore.loc[iron_ore["region"] == region_name][
                    "IronOreEur/t_ironore"
                ].values[0]
            elif regionalise == "uniform":
                iron_ore_cost = snakemake.config["iron_ore"]["marginal_cost"]
            else:
                raise ValueError(
                    "Invalid option for iron ore regionalisation. Choose 'grade-dependent' or 'uniform'."
                )

            n.add(
                "Generator",
                "{}_ore".format(region_name),
                bus=region_name + "_ore",
                carrier="iron_ore",
                p_nom_extendable=True,
                p_nom_max=iron_ore_limit,  # t_ore
                marginal_cost=iron_ore_cost,  # EUR/t_ore
                capital_cost=1 / 1000,  # to prevent optimisation shenanigans
            )

        # Demand load
        load = demands.loc[demands["region"] == region_name, "demand"].values[0]
        logger.info("Load set to 100%% of regional final energy demand.")
        n.add(
            "Load",
            region_name + "_" + final,
            bus=region_name + "_" + final,
            carrier=final,
            p_set=load,
        )

        # --- Stage 1 supply: ore → interone (material) or direct supply (energy) ---
        for s in range(len(region_data_interone)):

            if s == 0:
                p_nom_interone = float(region_data_interone[f"demand [{unit}]"][s])
            else:
                p_nom_interone = float(
                    region_data_interone[f"demand [{unit}]"][s]
                ) - float(region_data_interone[f"demand [{unit}]"][s - 1])

            m_cost = float(region_data_interone[f"{cost_descriptor} [EUR/{unit}]"][s])

            if not is_material_chain:
                # Pure energy chain (e.g. hydrogen): Generator directly on final bus
                n.add(
                    "Generator",
                    "{} supply {}_{}".format(
                        final, region_name, region_data_interone["load [t/h]"][s]
                    ),
                    bus=region_name,
                    carrier=final,
                    p_nom_extendable=True,
                    p_nom_max=p_nom_interone,  # MWh, demand = potential supply
                    marginal_cost=m_cost,  # EUR/MWh
                    capital_cost=1 / 1000,  # to prevent optimisation shenanigans
                )
            else:
                # Material chain: Link from ore bus to interone bus
                ore_ratio = snakemake.config["iron_ore"]["ore_to_steel_ratio"]
                n.add(
                    "Link",
                    "{} supply {}_{}".format(
                        interone,
                        region_name,
                        region_data_interone["load [t/h]"][s],
                    ),
                    bus0=region_name + "_ore",
                    bus1=region_name + "_" + interone,
                    carrier=interone,
                    p_nom_max=p_nom_interone
                    * ore_ratio,  # t, demand = potential supply
                    p_nom_extendable=True,
                    efficiency=1 / ore_ratio,
                    marginal_cost=m_cost / ore_ratio,  # referred to bus0
                    capital_cost=1 / 1000,  # to prevent optimisation shenanigans
                )

        # --- Stage 2 supply: interone → final (two-stage material chain only) ---
        if two_stage:
            for s in range(len(region_data_intertwo)):

                if s == 0:
                    p_nom_intertwo = float(region_data_intertwo[f"demand [{unit}]"][s])
                else:
                    p_nom_intertwo = float(
                        region_data_intertwo[f"demand [{unit}]"][s]
                    ) - float(region_data_intertwo[f"demand [{unit}]"][s - 1])

                # Override capacity for grid-connected EAF based on grid potential
                if intertwo == "eaf-grid":
                    grid_potential = pd.read_csv(
                        snakemake.input.grid_potential, header=0, index_col=0
                    )
                    p_nom_intertwo = (
                        grid_potential.loc[region_name, "potential_mt_steel"] * 1e6
                    ) / len(
                        region_data_intertwo
                    )  # split evenly across supply steps

                m_cost = float(
                    region_data_intertwo[f"{cost_descriptor} [EUR/{unit}]"][s]
                )

                n.add(
                    "Link",
                    "{} supply {}_{}".format(
                        final, region_name, region_data_intertwo["load [t/h]"][s]
                    ),
                    bus0=region_name + "_" + interone,
                    bus1=region_name + "_" + final,
                    carrier=final,
                    p_nom_max=p_nom_intertwo,  # t, demand = potential supply
                    p_nom_extendable=True,
                    efficiency=1,  # direct conversion, no ratio needed
                    marginal_cost=m_cost,  # EUR/t
                    capital_cost=1 / 1000,  # to prevent optimisation shenanigans
                )

    return n


def create_links(trade_options):

    # for in range of length of input csv with all the different links, region_from = column , region_to = column 2
    # create links with the correct corresponding costs

    shipping = snakemake.config["trade"]["shipping"]

    port_loading = shipping["port_loading"]
    nh3_cost = shipping["nh3_cost"]  # USD/t_NH3
    nh3_consumption = shipping["nh3_consumption"]  # t_NH3/nm
    charter_rate = shipping["charter_rate"]  # USD/(t*km)
    panamax_load = shipping["panamax_load"]  # t
    nm_to_km = shipping["nm_to_km"]
    eur_usd = snakemake.config["techno-economic parameters"]["eur_usd"]

    variable_cost = (
        (nh3_cost * nh3_consumption / nm_to_km) / panamax_load + charter_rate
    ) / eur_usd  # EUR/(t*km)
    fixed_cost = port_loading * 2 / eur_usd  # EUR/t

    logger.info(
        f"variable shipping cost {variable_cost:.6f} EUR/(t*km) and fixed cost {fixed_cost:.6f} EUR/t applied to shipping links"
    )

    # if there should be a link, create a link
    # do this for both shipping and pipeline
    for r in range(0, len(trade_options)):
        # checking if the row connects with shipping
        if trade_options["shipping"][r] == 1:
            r_from = trade_options["region_from"][r]
            r_to = trade_options["region_to"][r]

            shipping_cost = (
                variable_cost * float(trade_options["shipping_distance [km]"][r])
                + fixed_cost
            )  # EUR/t

            n.add(
                "Link",
                f"shipping {interone} {r_from}-{r_to}",
                carrier="shipping_" + interone,
                bus0=r_from + "_" + interone,
                bus1=r_to + "_" + interone,
                efficiency=1,
                marginal_cost=shipping_cost,
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
                p_nom_extendable=True,
            )
            logger.info(
                "shipping %s link made from %s to %s",
                interone,
                r_from,
                r_to,
            )

            n.add(
                "Link",
                "shipping iron ore {}-{}".format(r_from, r_to),
                carrier="shipping_iron_ore",
                bus0=r_from + "_ore",
                bus1=r_to + "_ore",
                efficiency=1,
                marginal_cost=shipping_cost,  # EUR/t_ironore
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
                p_nom_extendable=True,
            )
            logger.info(
                "iron ore shipping link made from %s_ore to %s_ore",
                r_from,
                r_to,
            )

        # checking if the row connects with pipeline
        if (trade_options["pipeline"][r] == 1) & (final == "hydrogen"):
            r_from = trade_options["region_from"][r]
            r_to = trade_options["region_to"][r]
            p_cost = int(float(trade_options["pipeline_distance [km]"][r]) * pipe_mc)
            filling_demand = 1.5 / 100  # DEA, energy transport datasheet, 2050, %
            losses = 1.7 / 100  # DEA, energy transport datasheet 2022, 2050, %/1000km
            eff = (1 - filling_demand) * (1 - losses) ** (
                float(trade_options["pipeline_distance [km]"][r]) / 1000
            )

            n.add(
                "Link",
                "pipeline {}-{}".format(r_from, r_to),
                bus0=r_from,
                bus1=r_to,
                efficiency=eff,  # calculated above
                marginal_cost=p_cost,  # EUR/MWh
                p_nom_extendable=True,
                capital_cost=1 / 1000,  # to prevent optimisation shenenigans
            )
            logger.info("pipeline link made from %s to %s - eff %s", r_from, r_to, eff)

    return


def save_trade_network(solved_network):

    sol = pd.DataFrame(columns=["type", "variable", "value", "unit"])
    # add objective cost
    sol.loc[sol.shape[0]] = ["objective", "cost", solved_network.objective, "EUR"]
    logger.info("added objective cost to sol")

    # add all generators with name and production value
    df_gen = solved_network.generators.p_nom_opt.T.to_frame()
    df_gen.reset_index(inplace=True)
    df_gen = df_gen.rename(columns={"Generator": "variable", "now": "value"})
    df_gen.insert(0, "type", "generator")
    df_gen.insert(3, "unit", unit)
    sol = pd.concat([sol, df_gen], ignore_index=True)
    logger.info("added generators to sol")

    # add all links with names and flows
    df_links = solved_network.links.p_nom_opt.T.to_frame()
    df_links.reset_index(inplace=True)
    df_links = df_links.rename(columns={"Link": "variable", "p_nom_opt": "value"})
    df_links.insert(0, "type", "link")
    df_links.insert(3, "unit", unit)
    sol = pd.concat([sol, df_links], ignore_index=True)
    logger.info("added links to sol")

    # add all bus (balance) - who is importing/exporting
    df_bus = solved_network.buses_t.p.T
    df_bus.reset_index(inplace=True)
    df_bus = df_bus.rename(columns={"Bus": "variable", "now": "value"})
    df_bus.insert(0, "type", "bus")
    df_bus.insert(3, "unit", unit + "/a")
    sol = pd.concat([sol, df_bus], ignore_index=True)
    logger.info("added bus_balances to sol")

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
    logger.info("added bus_capacities to sol")

    sol.to_csv(snakemake.output.trade_result)

    return


def save_network_collection(nc, output_path, optimal_network=None):
    """
    Save a NetworkCollection to netCDF format.

    Each network in the collection is saved as a separate netCDF file with
    the index/key appended to the filename. The optimal solution (without slack)
    is also saved to the base filename.

    Parameters:
    -----------
    nc : pypsa.NetworkCollection
        The NetworkCollection to save
    output_path : str
        Base output path (e.g., "network.nc"). Will become "network.nc" (optimal),
        "network_0.005.nc", "network_0.01.nc", etc.
    optimal_network : pypsa.Network, optional
        The optimal network (solved without MGA slack). If provided, saved to output_path.
    """

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split path into base and extension
    base, ext = os.path.splitext(output_path)

    logger.info(
        "Saving NetworkCollection with %s networks to %s", len(nc.networks), output_dir
    )

    # Save optimal network (without slack) to the base filename if provided
    if optimal_network is not None:
        logger.info(f"  Saving optimal network (no slack) to {output_path}")
        optimal_network.export_to_netcdf(output_path)

    # Save each network with its index/key in the filename
    for key, network in nc.networks.items():
        filename = f"{base}_{key}{ext}"
        logger.info(f"  Saving network with key '{key}' to {filename}")
        network.export_to_netcdf(filename)

    total_saved = len(nc.networks) + (1 if optimal_network is not None else 0)
    logger.info(f"Saved {total_saved} networks to {output_dir}")


def plot_trade_network(
    n,
    product="steel",
    alpha_supply=0.7,
    alpha_demand=1,
    output_path=None,
    output_path_png=None,
    region_gdf=None,
):
    """
    Plot trade network using config-driven colors and sizes.
    Calculates supply, demand, and trade according to product type.
    """
    config = snakemake.config
    plot_config = config["plot"]["world_map"][product]
    colors = config["colors"]
    supply_color = colors.get(f"{product}_supply", "black")
    demand_color = colors.get(f"{product}_demand", "lightsteelblue")
    link_colors = colors.get(f"{product}_link", "gray")

    supply_carrier = {"steel": "steel", "iron_ore": "iron_ore", "hbi": "hbi"}.get(
        product
    )
    supply_comp = {"steel": "Link", "iron_ore": "Generator", "hbi": "Link"}.get(product)
    demand_carrier = {"steel": "steel", "iron_ore": interone, "hbi": "steel"}.get(
        product
    )
    demand_comp = {"steel": "Load", "iron_ore": "Link", "hbi": "Link"}.get(
        product
    )  # Except when hbi is the final product, then the compontent for hbi must be load
    trade_carrier = {
        "steel": "shipping_steel",
        "iron_ore": "shipping_iron_ore",
        "hbi": "shipping_hbi",
    }.get(product)

    if product in [interone, intertwo, final, "iron_ore"]:
        supply = (
            n.statistics.supply(comps=[supply_comp], groupby=["bus", "carrier"])
            .loc[:, :, supply_carrier]
            .droplevel(0)
        )
        demand = (
            n.statistics.withdrawal(comps=[demand_comp], groupby=["bus", "carrier"])
            .loc[:, :, demand_carrier]
            .droplevel(0)
        )
        trade = n.links[n.links.carrier == trade_carrier].p_nom_opt.astype(int)
    else:
        supply = 0
        demand = 0
        trade = 0

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw dissolved region basemap (region borders only, no internal country borders)
    # color_geomap=False suppresses PyPSA's default NaturalEarth country background
    if region_gdf is not None:
        region_gdf.plot(
            ax=ax,
            color="lightgrey",
            edgecolor="white",
            linewidth=0.5,
        )

    # Plot demand
    n.plot.map(
        ax=ax,
        bus_sizes=demand * plot_config["bus_size"],
        bus_colors=demand_color,
        bus_alpha=alpha_demand,
        link_widths=0,
        branch_components=["Link"],
        geomap=False,
    )

    # Plot supply
    n.plot.map(
        ax=ax,
        bus_sizes=supply * plot_config["bus_size"],
        bus_colors=supply_color,
        bus_alpha=alpha_supply,
        link_widths=trade * plot_config["link_width"],
        branch_components=["Link"],
        link_colors=link_colors,
        geomap=False,
    )

    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
    # ax.set_global()

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=link_colors, label="shipping"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="Demand",
            markerfacecolor=demand_color,
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="Supply",
            markerfacecolor=supply_color,
            markersize=10,
        ),
    ]
    fig.legend(
        handles=legend_elements,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.22, 0.28),
    )

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
        fig.savefig(
            output_path_png, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300
        )

    # Close figure
    plt.close(fig)

    return


def apply_cost_penalty(n, cost_penalty):

    # Add cost pentalty to all technologies of a certain region, excluding shipping

    if cost_penalty:
        for region in cost_penalty.keys():
            n.links.loc[
                (
                    (n.links.bus1 == f"{region}_steel")
                    | (n.links.bus1 == f"{region}_hbi")
                )
                & ~n.links.carrier.str.contains("shipping"),
                "marginal_cost",
            ] *= cost_penalty[region]
    else:
        logger.info("No cost penalty applied")

    return n


def apply_hbi_diversity_constraint(n, diversity_factor, demands):
    """
    Apply HBI import diversity constraint.

    Constrains each importer region to not import more than diversity_factor
    from any single supplier.

    E.g., diversity_factor=0.5 means each region can import at most 50% of its
    steel demand from any single HBI supplier.

    Parameters:
    -----------
    n : pypsa.Network
        The network object
    diversity_factor : float
        Maximum share of HBI demand that can be supplied by a single supplier (0-1)
    demands : pd.DataFrame
        DataFrame with steel demands by region
    """

    if diversity_factor is False:
        logger.info("HBI diversity constraint disabled")
        return n

    if diversity_factor <= 0 or diversity_factor > 1:
        raise ValueError("diversity_factor must be between 0 and 1")

    # Get all HBI shipping links
    hbi_shipping_links = n.links[n.links.carrier == "shipping_hbi"]

    # Group by destination (bus1) to find all suppliers for each importer
    for destination_bus, group in hbi_shipping_links.groupby("bus1"):
        # Extract region name from bus (e.g., "Europe_hbi" -> "Europe")
        if destination_bus.endswith("_hbi"):
            region_name = destination_bus[:-4]
        else:
            region_name = destination_bus

        # Get the HBI demand for this region (equals steel demand in tonnes)
        demand_tonnes = demands.loc[demands["region"] == region_name, "demand"].values

        if len(demand_tonnes) == 0:
            logger.warning(
                "Warning: No demand found for region %s, skipping diversity constraint",
                region_name,
            )
            continue

        demand_tonnes = float(demand_tonnes[0])

        # Maximum import from single supplier = diversity_factor * demand
        max_from_single_supplier = diversity_factor * demand_tonnes

        # Set p_nom_max for all links to this destination
        for link_idx in group.index:
            n.links.loc[link_idx, "p_nom_max"] = max_from_single_supplier

        logger.info(
            "HBI diversity constraint applied to %s: max %.0f%% of %.0ft = %.0ft per supplier",
            region_name,
            diversity_factor * 100,
            demand_tonnes,
            max_from_single_supplier,
        )

    return n


def normalize_regions(regions, carrier):
    """Ensure regions are lists and suffixed with _{carrier}."""
    if regions is None:
        return None
    if isinstance(regions, str):
        regions = [regions]

    normalized = []
    for r in regions:
        if r.endswith(f"_{carrier}"):
            normalized.append(r)
        else:
            normalized.append(f"{r}_{carrier}")
    return normalized


def resolve_mga_exporters_from_indicator(mga, indicators):
    """
    Resolve MGA (Modelling to generate Alternatives) exporters based on indicator thresholds.

    If mga config specifies an 'indicator' instead of explicit 'export' regions,
    this function selects regions where the indicator is below the threshold_value.    Parameters:
    -----------
    mga : dict
        MGA configuration containing:
        - 'indicator': str, name of the indicator (e.g., 'stability')
        - 'threshold_value': float, regions with indicator below this value are selected
        - OR 'export': list of regions (manual specification takes precedence)
    indicators : dict
        Dictionary mapping indicator_name -> DataFrame with region as index

    Returns:
    --------
    dict : Updated mga config with 'export' regions resolved
    """

    # If export is manually specified, use that
    if "export" in mga and mga["export"] is not None:
        return mga

    # Otherwise, resolve from indicator
    if "indicator" not in mga:
        logger.info("No indicator or export specified in MGA config")
        return mga

    indicator_name = mga["indicator"]
    threshold = mga.get("threshold_value")

    # Chokepoint and blocks indicators are handled separately in solve_network
    if indicator_name in ("chokepoint", "blocks"):
        return mga

    if indicator_name not in indicators:
        raise ValueError(
            f"Indicator '{indicator_name}' not found in available indicators: {list(indicators.keys())}"
        )

    if threshold is None:
        raise ValueError(
            f"threshold_value must be specified when using indicator '{indicator_name}'"
        )

    indicator_data = indicators[indicator_name]

    # Select regions where indicator is below threshold
    # indicator_data is a DataFrame with region as index and a single column
    # Extract the actual values (first column)
    if isinstance(indicator_data, pd.DataFrame):
        values = indicator_data.iloc[:, 0]
    else:
        values = indicator_data

    selected_regions = values[values < threshold].index.tolist()

    logger.info(
        f"Selected regions with {indicator_name} < {threshold}: {selected_regions}"
    )
    logger.info(f"Values: {values[values < threshold].to_dict()}")

    # Set export to the selected regions
    mga["export"] = selected_regions

    return mga


def resolve_mga_links_from_chokepoints(n, mga, trade_options, interone):
    """
    Resolve MGA link indices based on chokepoint avoidance.

    Instead of filtering by exporter region (like stability), this function
    selects shipping links whose routes pass through any of the chokepoints
    listed in mga['threshold_value'].

    Parameters:
    -----------
    n : pypsa.Network
        The network containing shipping links
    mga : dict
        MGA configuration containing:
        - 'carrier': str, the carrier type (e.g., 'hbi')
        - 'threshold_value': list of str, chokepoint names to avoid
          (e.g., ['suez', 'ormuz', 'babalmandab'])
    trade_options : pd.DataFrame
        Trade options DataFrame with 'region_from', 'region_to', and 'chokepoints' columns.
        The 'chokepoints' column contains semicolon-separated passage names.
    interone : str
        The intermediate product name (e.g., 'hbi')

    Returns:
    --------
    pd.Index : Index of link names whose routes traverse listed chokepoints
    """
    carrier = mga["carrier"]
    chokepoints_to_avoid = set(mga["threshold_value"])

    # Build a lookup: (region_from, region_to) -> set of chokepoints
    route_chokepoints = {}
    for _, row in trade_options.iterrows():
        r_from = row["region_from"]
        r_to = row["region_to"]
        cp_str = row.get("chokepoints", "")
        if pd.isna(cp_str) or cp_str == "":
            cp_set = set()
        else:
            cp_set = set(str(cp_str).split(";"))
        route_chokepoints[(r_from, r_to)] = cp_set

    # Select shipping links that match the carrier and traverse any listed chokepoint
    mask = n.links.carrier == f"shipping_{carrier}"
    carrier_links = n.links[mask]

    selected_links = []
    for link_name, link_row in carrier_links.iterrows():
        # Extract region_from and region_to from link name
        # Link names: "shipping {interone} {r_from}-{r_to}" or "shipping iron ore {r_from}-{r_to}"
        if f"shipping {interone}" in link_name:
            route_part = link_name.replace(f"shipping {interone} ", "")
        elif "shipping iron ore" in link_name:
            route_part = link_name.replace("shipping iron ore ", "")
        else:
            continue

        parts = route_part.split("-")
        if len(parts) == 2:
            r_from, r_to = parts
        else:
            # Handle region names that contain hyphens (unlikely but safe)
            continue

        # Check if this route passes through any chokepoint to avoid
        route_cp = route_chokepoints.get((r_from, r_to), set())
        if route_cp & chokepoints_to_avoid:  # set intersection
            selected_links.append(link_name)
            logger.info(
                "  Chokepoint MGA: link '%s' traverses %s",
                link_name,
                route_cp & chokepoints_to_avoid,
            )

    logger.info(
        "Chokepoint MGA: selected %s/%s shipping links traversing %s",
        len(selected_links),
        len(carrier_links),
        chokepoints_to_avoid,
    )

    return pd.Index(selected_links)


def resolve_mga_links_from_blocks(n, mga):
    """
    Resolve MGA link indices for inter-block trade minimisation.

    Selects shipping links where the exporter (bus0) belongs to one block
    and the importer (bus1) belongs to a different block.  Intra-block trade
    is left unconstrained.

    Parameters
    ----------
    n : pypsa.Network
        The network containing shipping links.
    mga : dict
        MGA configuration.  ``threshold_value`` must be a dict mapping
        block names to lists of region names, e.g.
        ``{"block_a": ["Europe", ...], "block_b": ["Middle_East", ...]}``.

    Returns
    -------
    pd.Index
        Index of link names that represent inter-block shipping.
    """
    carrier = mga["carrier"]
    blocks = mga["threshold_value"]  # dict: block_name -> [regions]

    # Build region -> block mapping.
    # Each entry in the regions list may itself be a comma-separated string
    # (e.g. from YAML flow syntax), so split defensively.
    region_to_block = {}
    for block_name, regions in blocks.items():
        for entry in regions:
            for region in str(entry).split(","):
                region = region.strip()
                if region:
                    region_to_block[region] = block_name

    logger.info(
        "Blocks MGA: %s",
        ", ".join(
            f"{k}: {sum(1 for v in region_to_block.values() if v == k)} regions"
            for k in blocks
        ),
    )

    # Select shipping links for the target carrier
    mask = n.links.carrier == f"shipping_{carrier}"
    carrier_links = n.links[mask]

    selected_links = []
    for link_name, link_row in carrier_links.iterrows():
        # bus0 / bus1 are e.g. "Europe_hbi", "Middle_East_hbi"
        bus0_region = link_row["bus0"].rsplit(f"_{carrier}", 1)[0]
        bus1_region = link_row["bus1"].rsplit(f"_{carrier}", 1)[0]

        block_from = region_to_block.get(bus0_region)
        block_to = region_to_block.get(bus1_region)

        if block_from is None or block_to is None:
            # Region not assigned to any block — skip
            continue

        if block_from != block_to:
            selected_links.append(link_name)
            logger.info(
                "  Blocks MGA: link '%s' crosses %s → %s",
                link_name,
                block_from,
                block_to,
            )

    logger.info(
        "Blocks MGA: selected %s/%s inter-block shipping links",
        len(selected_links),
        len(carrier_links),
    )

    return pd.Index(selected_links)


def resolve_mga_links(n, mga, indicators=None):
    """
    Unified dispatcher that resolves which shipping links to include in the
    MGA objective, based on the indicator type.

    Supports:
    - ``"chokepoint"``: routes traversing listed chokepoints
    - ``"blocks"``: inter-block trade routes
    - ``"stability"`` / manual ``export``/``import``: region-based filtering

    Returns
    -------
    pd.Index
        Index of link names to include in the MGA weights.
    """
    indicator_name = mga.get("indicator")
    carrier = mga["carrier"]

    if indicator_name == "chokepoint" and indicators and "chokepoint" in indicators:
        return resolve_mga_links_from_chokepoints(
            n, mga, indicators["chokepoint"], interone
        )

    if indicator_name == "blocks":
        return resolve_mga_links_from_blocks(n, mga)

    # Default: region-based (stability, manual export/import)
    exports = normalize_regions(mga.get("export"), carrier)
    imports = normalize_regions(mga.get("import"), carrier)

    mask = n.links.carrier == f"shipping_{carrier}"
    if exports is not None:
        mask &= n.links.bus0.isin(exports)
    if imports is not None:
        mask &= n.links.bus1.isin(imports)

    return n.links[mask].index


def solve_network(n, mga=None, indicators=None):
    """
    Solve the network with optional MGA (Modelling to generate Alternatives).

    Always returns a tuple (optimal_network, NetworkCollection) where:
    - optimal_network: the optimal solution without MGA slack
    - NetworkCollection: contains all MGA variants (or just the optimal if no MGA)

    Parameters:
    -----------
    n : pypsa.Network
        The network to solve
    mga : dict, optional
        MGA configuration with keys: carrier, slack (list), sense,
        and optionally export, import, indicator, threshold_value
    indicators : dict, optional
        Dictionary of indicator DataFrames for indicator-based MGA

    Returns:
    --------
    tuple(pypsa.Network, pypsa.NetworkCollection)
        (optimal_network, NetworkCollection with MGA variants or just optimal)
    """

    solver_name = snakemake.config["solver"]["name"]
    options = snakemake.config["solver_options"][snakemake.config["solver"]["options"]]

    # First optimize without MGA
    n.optimize(n.snapshots, solver_name=solver_name, solver_options=options)

    # Clear solver model attached to the solved network so it can be copied.
    try:
        if hasattr(n, "model") and getattr(n.model, "solver_model", None) is not None:
            n.model.solver_model = None
    except Exception as e:
        logger.warning(f"Warning clearing solver model before copying network: {e}")

    optimal_network = n.copy()  # Store the optimal solution

    # If no MGA, return optimal as both optimal and collection
    if mga is None:
        nc = pypsa.NetworkCollection({None: optimal_network})
        return (optimal_network, nc)

    # Resolve exporters from indicator if needed
    if indicators:
        mga = resolve_mga_exporters_from_indicator(mga, indicators)

    tsc = (
        pd.concat([n.statistics.capex(), n.statistics.opex()], axis=1)
        .sum(axis=1)
        .div(1e9)
    )
    optimal_cost = tsc.sum()

    # Resolve which links to target via unified dispatcher
    idx = resolve_mga_links(n, mga, indicators)

    # Build MGA weights for all matched links.
    # If the config provides a 'weighting' dict (region -> float), links whose
    # source bus starts with that region name receive that weight.
    # All other links default to weight 1.
    region_weights = mga.get("weighting") or {}

    def _link_weight(link_name):
        bus0 = n.links.at[link_name, "bus0"]  # e.g. "North_West_Africa_hbi"
        for region, w in region_weights.items():
            if bus0.startswith(region):
                return float(w)
        return 1.0

    weights = {"Link": {"p_nom": {link: _link_weight(link) for link in idx}}}

    sense = mga["sense"]
    slack_list = mga["slack"]  # Always a list in config

    # Handle slack values (always as a list)
    logger.info(f"MGA activated with slacks: {slack_list}")
    logger.info(f"Optimal cost (no MGA): {optimal_cost:.2f} B€")

    networks = {}

    for slack_value in slack_list:
        logger.info(f"\n--- Solving with slack = {slack_value} ---")

        # Create a copy of the network for each slack
        n_copy = n.copy()

        # Run MGA optimization
        n_copy.optimize.optimize_mga(
            slack=slack_value,
            weights=weights,
            sense=sense,
            solver_name=solver_name,
            solver_options=options,
        )

        tsc = (
            pd.concat([n_copy.statistics.capex(), n_copy.statistics.opex()], axis=1)
            .sum(axis=1)
            .div(1e9)
        )
        mga_cost = tsc.sum()
        logger.info(
            f"MGA cost: {mga_cost:.2f} B€, allowed cost increase: {optimal_cost * (1 + slack_value):.2f} B€"
        )

        # Store in dictionary with slack as key
        networks[slack_value] = n_copy

    # Create NetworkCollection with optimal at None key plus all MGA variants
    networks[None] = optimal_network
    nc = pypsa.NetworkCollection(networks)
    return (optimal_network, nc)


if __name__ == "__main__":
    if snakemake is None:
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "model_trade",
            cost_year="2050",
            interone="hbi",
            intertwo="eaf",
            final="steel",
            scenario="mga-blocs",
            wacc="regional",
            chain_id="newre1206_2050",
        )

    final = snakemake.wildcards["final"]
    interone = snakemake.wildcards["interone"]
    intertwo = snakemake.wildcards["intertwo"]
    scenario = snakemake.wildcards["scenario"]
    chain_id = snakemake.wildcards["chain_id"]

    trade_chain = snakemake.config["trade_chains"]

    # Derive model structure from trade_chains config (source of truth).
    # The wildcard variables above are kept for Snakefile compatibility only.
    final = trade_chain["final_product"]
    tradeable = trade_chain["tradeable_commodities"]
    interone = next(c for c in tradeable if c != "iron_ore")
    stages = trade_chain["stages"]
    intertwo = stages[max(stages.keys())]["process_label"]

    logger.info(
        "intermediate 1 (%s) and intermediate 2 (%s) to final product %s",
        interone,
        intertwo,
        final,
    )

    shipping_first = "iron_ore"
    shipping_second = interone

    logger.info("starting up with all regions--- ")
    # making dataframes
    trade_options = pd.read_csv(snakemake.input.trade_options, header=0)
    supply_curves_interone = snakemake.input.supply_curves_interone
    supply_curves_intertwo = snakemake.input.supply_curves_intertwo
    bus_locations = pd.read_csv(snakemake.input.bus_locations, header=0)
    iron_ore = pd.read_csv(snakemake.input.iron_ore, header=0)
    political_stability_data = pd.read_csv(
        snakemake.input.political_stability, index_col=0
    )
    regions = snakemake.config["regions"]

    # Load indicators for MGA (flexible architecture for future extensions)
    indicators = {}
    indicators["stability"] = political_stability_data
    indicators["chokepoint"] = (
        trade_options  # trade_options now has 'chokepoints' column
    )

    # limit regions
    trade_options = trade_options[
        (trade_options["region_from"].isin(regions))
        & (trade_options["region_to"].isin(regions))
    ].reset_index(drop=True)
    bus_locations = bus_locations[
        bus_locations["region_name"].isin(regions)
    ].reset_index(drop=True)
    iron_ore = iron_ore[iron_ore["region"].isin(regions)].reset_index(drop=True)

    # Get iron ore cost option: grade-dependent (regional) or uniform
    regionalise = snakemake.config["iron_ore"]["regionalise"]

    if final == "steel":
        demands = pd.read_csv(snakemake.input.steel_demand, header=0)
        demands.rename(columns={"SteelDemand_DRI_Mt": "demand"}, inplace=True)
        demands["demand"] = demands["demand"] * 1e6  # Mt to t
        unit = "t"

    elif final == "hydrogen":
        demands = pd.read_csv(snakemake.input.demand, header=0)
        unit = "MWh"

    else:
        raise ValueError("Product must be either 'steel' or 'hydrogen'.")

    cost_descriptor = "lcox"

    plot_config = snakemake.config["plot"]["world_map"][final]

    logger.info("data loaded successfully")

    # building model
    logger.info("building model")
    n = building_model(
        supply_curves_interone,
        supply_curves_intertwo,
        demands,
        bus_locations,
        trade_chain,
    )

    # building transport network connecting the individual buses
    logger.info("building transportation links")
    create_links(trade_options)

    # Cost penalty
    if snakemake.config["scenario"][scenario]["modifiers"]["cost_penalty"] is None:
        cost_penalty = None
        logger.info("cost_penalty not activated")
    else:
        cost_penalty = snakemake.config["scenario"][scenario]["modifiers"][
            "cost_penalty"
        ]
        logger.info(
            "applying cost penalty scenario: %s with penalties %s",
            scenario,
            cost_penalty,
        )
        n = apply_cost_penalty(n, cost_penalty)

    # HBI diversity constraint
    diversity_factor = snakemake.config["trade"]["diversity_factor"]
    if diversity_factor is not False:
        logger.info(
            "applying HBI diversity constraint with factor %s", diversity_factor
        )
        n = apply_hbi_diversity_constraint(n, diversity_factor, demands)
    else:
        logger.info("HBI diversity constraint disabled")

    # MGA
    if "mga" not in snakemake.config["scenario"][scenario]["modifiers"].keys():
        mga = None
        logger.info("MGA not activated")
    else:
        mga = snakemake.config["scenario"][scenario]["modifiers"]["mga"]
        logger.info(f"MGA activated with slack {mga['slack']}")

    # solving model
    logger.info("solving model")
    result = solve_network(n, mga=mga, indicators=indicators if indicators else None)
    logger.info("network was solved")

    # Export result: always a tuple (optimal_network, NetworkCollection)
    logger.info("saving network to netCDF")
    optimal_net, nc = result
    save_network_collection(
        nc, snakemake.output.trade_network, optimal_network=optimal_net
    )
    n_selected = optimal_net
    logger.info(
        f"Saved optimal network and NetworkCollection with {len(nc.networks)} networks"
    )

    # saving results and calculating LCOH
    logger.info("saving results as network+csv")
    save_trade_network(n_selected)

    # Build dissolved region GeoDataFrame once for basemap (no internal country borders)
    region_gdf = build_region_geodataframe(snakemake.config)

    # Plot results: consolidate plotting for all networks
    logger.info("saving plots")

    # Define the products to plot and their settings
    plot_settings = [
        (
            "iron_ore",
            0.5,
            snakemake.output.trade_plot_ironore,
            snakemake.output.trade_plot_ironore_png,
        ),
        (
            "hbi",
            0.5,
            snakemake.output.trade_plot_hbi,
            snakemake.output.trade_plot_hbi_png,
        ),
        (
            "steel",
            0.7,
            snakemake.output.trade_plot_steel,
            snakemake.output.trade_plot_steel_png,
        ),
    ]

    # Plot for each network in the collection (including optimal at key=None)
    for slack_key, network in nc.networks.items():
        # Check if this is the optimal solution (None key)
        is_optimal = pd.isna(slack_key)

        if is_optimal:
            logger.info("\nPlotting optimal network (no slack)")
        else:
            logger.info(f"\nPlotting for slack={slack_key}")

        for product, alpha_supply, output_path, output_path_png in plot_settings:
            # Use base filenames for optimal, append slack value for MGA variants
            if is_optimal:
                final_output_path = output_path
                final_output_path_png = output_path_png
            else:
                base, ext = os.path.splitext(output_path)
                final_output_path = f"{base}_{slack_key}{ext}"
                base_png, ext_png = os.path.splitext(output_path_png)
                final_output_path_png = f"{base_png}_{slack_key}{ext_png}"

            plot_trade_network(
                network,
                product=product,
                alpha_supply=alpha_supply,
                output_path=final_output_path,
                output_path_png=final_output_path_png,
                region_gdf=region_gdf,
            )
