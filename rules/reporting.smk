"""Reporting workflow rules.

Collects final figures and presentation artifacts produced by notebooks and the
main optimization workflow.
"""

rule plot_regions:
    params:
        region_nice_names=config["region_nice_names"],
    output:
        global_map_countries = "results/figures_general/global_map_countries.pdf", #workflow/notebooks/plot_countries.ipynb
        global_map_countries_png = "results/figures_general/global_map_countries.png", #workflow/notebooks/plot_countries.ipynb
    notebook:
        str(NOTEBOOKS_DIR / "plot_countries.ipynb")


rule collect_figures:
    input:
        global_supply_curve = "../results/figures_general/{wacc}/{scenario}/global_supply_curve_{sort}_demand_{demand}_{interone}.pdf", #workflow/notebooks/analysis-coststructure.ipynb
        global_supply_curve_png = "../results/figures_general/{wacc}/{scenario}/global_supply_curve_{sort}_demand_{demand}_{interone}.png", #workflow/notebooks/analysis-coststructure.ipynb
        electricity_demand = "../results/figures_general/electricity_demand.pdf", #workflow/notebooks/analysis-electricity-demand.ipynb
        electricity_demand_png = "../results/figures_general/electricity_demand.png", #workflow/notebooks/analysis-electricity-demand.ipynb
        electricity_demand_steel = "../results/figures_general/electricity_demand_in_steel.pdf", #workflow/notebooks/analysis-electricity-demand.ipynb
        electricity_demand_steel_png = "../results/figures_general/electricity_demand_in_steel.png", #workflow/notebooks/analysis-electricity-demand.ipynb
        global_map_countries = "../results/figures_general/global_map_countries.pdf", #workflow/notebooks/plot_countries.ipynb
        global_map_countries_png = "../results/figures_general/global_map_countries.png", #workflow/notebooks/plot_countries.ipynb
        cost_comparison = "../results/figures_general/comparison/cost_comparison.pdf", #workflow/notebooks/compare-scenarios.ipynb
        cost_comparison_png = "../results/figures_general/comparison/cost_comparison.png", #workflow/notebooks/compare-scenarios.ipynb
        value_chain_comparison = "../results/figures_general/value_chain_comparison.pdf", #workflow/notebooks/analyse-steel-hbi-split.ipynb
        value_chain_comparison_png = "../results/figures_general/value_chain_comparison.png", #workflow/notebooks/analyse-steel-hbi-split.ipynb
        hourly_analysis = "../results/figures_general/hourly_analysis.pdf", #workflow/notebooks/analysis-hourly.ipynb
        hourly_analysis_png = "../results/figures_general/hourly_analysis.png", #workflow/notebooks/analysis-hourly.ipynb
        iron_ore       = "../results/figures_general/trade-today/Iron_Ore_net_flow.pdf",
        dri_hbi        = "../results/figures_general/trade-today/DRI-HBI_net_flow.pdf",
        steel_raw      = "../results/figures_general/trade-today/Steel_raw_net_flow.pdf",
        mga_plot = "../results/figures_general/mga/mga_analysis.pdf", # integrated in workflow
        mga_plot_png = "../results/figures_general/mga/mga_analysis.png", # integrated in workflow
        map_chokepoints = "../results/figures_general/chokepoints/map_chokepoints.pdf", # integrated in workflow
        map_chokepoints_png = "../results/figures_general/chokepoints/map_chokepoints.png", # integrated in workflow


rule get_figures:
    input:
        [
        "results/figures_general/comparison/cost_year~2050/wacc~uniform/lcox_comparison_East_Asia_East_East_Asia.pdf",
        "results/figures_general/comparison/cost_year~2050/wacc~uniform/lcox_comparison_South_America_Europe.pdf",
        "results/figures_general/comparison/cost_year~2050/wacc~regional/lcox_comparison_East_Asia_East_East_Asia.pdf",
        "results/figures_general/comparison/cost_year~2050/wacc~regional/lcox_comparison_South_America_Europe.pdf",
        "results/figures_general/mga/chain_id~supplyconstraint/wacc~regional/mga_analysis.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~default/map_hbi.pdf",
        "results/figures_general/global_map_countries.pdf",
        "results/figures_general/chokepoints/map_chokepoints.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-chokepoints/map_hbi_0.002.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-chokepoints/map_ironore_0.002.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-blocs/map_hbi_0.001.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-blocs/map_ironore_0.001.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~constrain-supply/map_hbi_250.0.pdf",
        "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~constrain-supply/map_ironore_250.0.pdf",
        "results/figures_general/global_supply_curve/chain_id~supplyconstraint/cost_year~2050/uniform/default/global_supply_curve_cost_average_hbi.pdf",
        "results/figures_general/global_supply_curve/chain_id~supplyconstraint/cost_year~2050/uniform/default/global_supply_curve_cost_global_hbi.pdf",
        "results/figures_general/global_supply_curve/chain_id~supplyconstraint/cost_year~2050/regional/default/global_supply_curve_cost_average_hbi.pdf",
        "results/figures_general/global_supply_curve/chain_id~supplyconstraint/cost_year~2050/regional/default/global_supply_curve_cost_global_hbi.pdf",
        "results/figures_general/global_supply_curve/chain_id~supplyconstraint/cost_year~2050/regional/default/supply_curve_details_global_hbi.pdf",
        "results/figures_general/pull/chain_id~supplyconstraint/cost_year~2050/regional/default/magnitude_pull_hbi.pdf",
        "results/figures_general/mga/map_robust.pdf",
        # Supplementary
        "results/figures_general/trade-today/Iron_Ore_net_flow.pdf",
        "results/figures_general/trade-today/DRI-HBI_net_flow.pdf",
        "results/figures_general/trade-today/Steel_net_flow.pdf",
        ]
    output:
        [
        "results/figures_streamlined/lcox-east-asia-homogenous.pdf",
        "results/figures_streamlined/lcox-south-america-homogenous.pdf",
        "results/figures_streamlined/lcox-east-asia.pdf",
        "results/figures_streamlined/lcox-south-america.pdf",
        "results/figures_streamlined/mga-analysis.pdf",
        "results/figures_streamlined/map-hbi-opti.pdf",
        "results/figures_streamlined/map-countries.pdf",
        "results/figures_streamlined/map-chokepoints.pdf",
        "results/figures_streamlined/map-hbi-chokepoints.pdf",
        "results/figures_streamlined/map-ironore-chokepoints.pdf",
        "results/figures_streamlined/map-hbi-blocs.pdf",
        "results/figures_streamlined/map-ironore-blocs.pdf",
        "results/figures_streamlined/map-hbi-supply.pdf",
        "results/figures_streamlined/map-ironore-supply.pdf",
        "results/figures_streamlined/supply-sorted-homo.pdf",
        "results/figures_streamlined/supply-unsorted-homo.pdf",
        "results/figures_streamlined/supply-sorted-hetero.pdf",
        "results/figures_streamlined/supply-unsorted-hetero.pdf",
        "results/figures_streamlined/supply-details.pdf",
        "results/figures_streamlined/magnitude-pull.pdf",
        "results/figures_streamlined/map-robust.pdf",
        "results/figures_streamlined/today-ironore.pdf",
        "results/figures_streamlined/today-dri-hbi.pdf",
        "results/figures_streamlined/today-steel.pdf",
        ]
    threads: 1
    run:
        for i in range(len(input)):
            copyfile(input[i], output[i])



rule plot_mga:
    params:
        region_nice_names=config["region_nice_names"],
    input:
        network_mga_production = "results/chain_id~{trade_chain}/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-stability-weighted/network.nc",
        network_mga_chokepoints = "results/chain_id~{trade_chain}/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-chokepoints/network.nc",
        network_mga_blocks = "results/chain_id~{trade_chain}/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-blocs/network.nc",
        network_pareto_supply = "results/chain_id~{trade_chain}/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~constrain-supply/network.nc",
        political_stability = "resources/political_stability_clustered.csv",
        trade_options_chokepoints = "resources/trade_opt_chokepoints.csv",
        steel_demand = "resources/steel_demand_clustered_2050.csv",
    output:
        mga_plot = "results/figures_general/mga/chain_id~{trade_chain}/wacc~{wacc}/mga_analysis.pdf",
        mga_plot_png = "results/figures_general/mga/chain_id~{trade_chain}/wacc~{wacc}/mga_analysis.png",
    resources:
        mem_mb=4000,
    threads: 2
    notebook:
        str(NOTEBOOKS_DIR / "plot-mga.ipynb")

rule plot_mga_all:
    input:
        expand("results/figures_general/mga/chain_id~{trade_chain}/wacc~{wacc}/mga_analysis.pdf", trade_chain=[config["trade_chains"]["id"]], wacc=[config["trade_chains"]["wacc"]], allow_missing=True) #wacc=["uniform", "regional"]


rule plot_trade_today:
    input:
        baci_folder = ancient("data/BACI_HS22_V202601"),
    output:
        iron_ore       = "results/figures_general/trade-today/Iron_Ore_net_flow.pdf",
        iron_ore_png   = "results/figures_general/trade-today/Iron_Ore_net_flow.png",
        dri_hbi        = "results/figures_general/trade-today/DRI-HBI_net_flow.pdf",
        dri_hbi_png    = "results/figures_general/trade-today/DRI-HBI_net_flow.png",
        steel      = "results/figures_general/trade-today/Steel_net_flow.pdf",
        steel_png  = "results/figures_general/trade-today/Steel_net_flow.png",
        iron_ore_csv   = "results/figures_general/trade-today/Iron_Ore_trade_iso3.csv",
        dri_hbi_csv    = "results/figures_general/trade-today/DRI-HBI_trade_iso3.csv",
        steel_csv  = "results/figures_general/trade-today/Steel_trade_iso3.csv",
    script:
        str(NOTEBOOKS_DIR / "plot_todays-trade.py")

rule plot_global_supply:
    input:
        trade_network="results/chain_id~{trade_chain}/cost_year~{cost_year}/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~{scenario}/network.nc",
        # supply = "../resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_{product}.csv",
        # supply_nodemand = "../resources/supply_curves_nodemand/cost_year~{cost_year}/wacc~{wacc}/{region}_{product}.csv",
        supply_curves_interone = expand(
            "resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{interone}.csv",
            allow_missing=True, region=config["regions"]),
        steel_demand="resources/steel_demand_clustered_{cost_year}.csv",
        population="data/owid-population/population.csv",
    output:
        network_curve="results/figures_general/global_supply_curve/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_{interone}.pdf",
        network_curve_png="results/figures_general/global_supply_curve/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_{interone}.png",
        # supply_curves
        network_curve_details="results/figures_general/global_supply_curve/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/supply_curve_details_{sort}_{interone}.pdf",
        network_curve_details_png="results/figures_general/global_supply_curve/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/supply_curve_details_{sort}_{interone}.png",
        # supply_curves
    notebook:
        str(NOTEBOOKS_DIR / "analysis-globalsupplycurve.ipynb")

rule plot_global_supply_all:
        input:
            expand("results/figures_general/global_supply_curve/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_{interone}.pdf", trade_chain=[config["trade_chains"]["id"]],  cost_year=[2050], wacc=["regional", "uniform"], interone=["hbi"], scenario=["default"], sort=["cost_average","cost_global"], allow_missing=True)

rule plot_comparison:
    input:
        default = "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~default/network.nc",
        constraint100 = "results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~constrain-supply/network_100.0.nc",
        hightrans = "results/chain_id~hightrans/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~default/network.nc",
    output:
        cost_comparison="results/figures_general/comparison/cost_comparison.pdf",
        cost_comparison_png="results/figures_general/comparison/cost_comparison.png",
    notebook:
        str(NOTEBOOKS_DIR / "compare-scenarios.ipynb")


rule plot_magnitude_pull:
    input:
        steel_production = "data/brownfield-steel/country_crude_steel_production_population_per_capita_2024.csv",
        supply_curves_interone = expand("resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{interone}.csv", allow_missing=True, region=config["regions"]),
        steel_demand="resources/steel_demand_clustered_{cost_year}.csv",
        population="data/owid-population/population.csv",
    output:
        magnitude_pull="results/figures_general/pull/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/magnitude_pull_{interone}.pdf",
        magnitude_pull_png="results/figures_general/pull/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/magnitude_pull_{interone}.png",
    notebook:
        str(NOTEBOOKS_DIR / "plot-magnitude-pull.ipynb")

rule plot_magnitude_pull_all:
        input:
            expand("results/figures_general/pull/chain_id~{trade_chain}/cost_year~{cost_year}/{wacc}/{scenario}/magnitude_pull_{interone}.pdf", trade_chain=[config["trade_chains"]["id"]],  cost_year=[2050], wacc=["regional", "uniform"], interone=["hbi"], scenario=["default"], allow_missing=True)


rule plot_robust_map:
    input:
        chokepoints="results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-chokepoints/network_0.002.nc",
        blocs="results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-blocs/network_0.001.nc",
        constrain_supply="results/chain_id~supplyconstraint/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~constrain-supply/network_250.0.nc",
    output:
        robust_map_pdf="results/figures_general/mga/map_robust.pdf",
        robust_map_png="results/figures_general/mga/map_robust.png",
    notebook:
        str(NOTEBOOKS_DIR / "plot-robust-map.ipynb")


# Variables captured by the plot_compare_lcox input lambda (avoids two-argument lambda)

# First case
_lcox_low_cost  = ["East_Asia"]
_lcox_high_cost = ["East_East_Asia"]
_comparison = ["East_East_Asia", "East_Asia"]

# Second case
# _lcox_low_cost  = ["South_America"]
# _lcox_high_cost = ["Europe"]
# _comparison = ["Europe", "South_America"]


_lcox_quantities = [1, 10, 100]

rule plot_compare_lcox:
    params:
        low_cost   = _lcox_low_cost,
        high_cost  = _lcox_high_cost,
        quantities = _lcox_quantities,
        comparison = _comparison,
    input:
        supply_networks=lambda wildcards: expand(
            "resources/lco-hbi/cost_year~{cost_year}/wacc~{wacc}/{region}_allocated_share/network_{qty}.nc",
            cost_year=wildcards.cost_year,
            wacc=wildcards.wacc,
            region=_lcox_low_cost + _lcox_high_cost,
            qty=_lcox_quantities,
        ),
        trade_result=lambda wildcards: (
            f"results/chain_id~{config['trade_chains']['id']}"
            f"/cost_year~{wildcards.cost_year}/interone~hbi/intertwo~eaf"
            f"/wacc~{wildcards.wacc}/final~steel/scenario~default/network.nc"
        ),
    output:
        lcox_comparison="results/figures_general/comparison/cost_year~{cost_year}/wacc~{wacc}/lcox_comparison_" + f"{_lcox_low_cost[0]}" + "_" + f"{_lcox_high_cost[0]}" + ".pdf",
        lcox_comparison_png="results/figures_general/comparison/cost_year~{cost_year}/wacc~{wacc}/lcox_comparison_" + f"{_lcox_low_cost[0]}" + "_" + f"{_lcox_high_cost[0]}" + ".png",
    notebook:
        str(NOTEBOOKS_DIR / "plot-compare-lcox.ipynb")

rule plot_compare_lcox_all:
    input:
        expand("results/figures_general/comparison/cost_year~{cost_year}/wacc~{wacc}/lcox_comparison_" + f"{_lcox_low_cost[0]}" + "_" + f"{_lcox_high_cost[0]}" + ".pdf", cost_year=[2050], wacc=[config["trade_chains"]["wacc"]], low_cost=_lcox_low_cost, high_cost=_lcox_high_cost, allow_missing=True)
