"""Reporting workflow rules.

Collects final figures and presentation artifacts produced by notebooks and the
main optimization workflow.
"""


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



rule plot_mga:
    input:
        network_mga_production = "results/chain_id~labour_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-stability-weighted/network.nc",
        network_mga_chokepoints = "results/chain_id~labour_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-chokepoints/network.nc",
        network_mga_blocks = "results/chain_id~labour_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~{wacc}/final~steel/scenario~mga-blocs/network.nc",
        political_stability = "resources/political_stability_clustered.csv",
        trade_options_chokepoints = "resources/trade_opt_chokepoints.csv",
    output:
        mga_plot = "results/figures_general/mga/wacc~{wacc}/mga_analysis.pdf",
        mga_plot_png = "results/figures_general/mga/wacc~{wacc}/mga_analysis.png",
    resources:
        mem_mb=4000,
    threads: 2
    notebook:
        "../workflow/notebooks/plot-mga.ipynb"

rule plot_mga_all:
    input:
        expand("results/figures_general/mga/wacc~{wacc}/mga_analysis.pdf", wacc=["regional"], allow_missing=True) #wacc=["uniform", "regional"]


rule plot_trade_today:
    input:
        baci_folder = ancient("../data/BACI_HS22_V202601"),
    output:
        iron_ore       = "../results/figures_general/trade-today/Iron_Ore_net_flow.pdf",
        iron_ore_png   = "../results/figures_general/trade-today/Iron_Ore_net_flow.png",
        dri_hbi        = "../results/figures_general/trade-today/DRI-HBI_net_flow.pdf",
        dri_hbi_png    = "../results/figures_general/trade-today/DRI-HBI_net_flow.png",
        steel_raw      = "../results/figures_general/trade-today/Steel_raw_net_flow.pdf",
        steel_raw_png  = "../results/figures_general/trade-today/Steel_raw_net_flow.png",
        iron_ore_csv   = "../results/figures_general/trade-today/Iron_Ore_trade_iso3.csv",
        dri_hbi_csv    = "../results/figures_general/trade-today/DRI-HBI_trade_iso3.csv",
        steel_raw_csv  = "../results/figures_general/trade-today/Steel_raw_trade_iso3.csv",
    script:
        "notebooks/plot_todays-trade.py"

rule plot_global_supply:
    input:
        trade_network="../results/cost_year~{cost_year}/interone~hbi/intertwo~eaf-grid/final~steel/wacc~{wacc}/scenario~{scenario}/network.nc",
        # supply = "../resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_{product}.csv",
        # supply_nodemand = "../resources/supply_curves_nodemand/cost_year~{cost_year}/wacc~{wacc}/{region}_{product}.csv",
        supply_curves_interone = expand(
            "../resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_{interone}.csv",
            allow_missing=True, region=config["regions"]),
        supply_curves_interone_nodemand = expand(
            "../resources/supply_curves_nodemand/cost_year~{cost_year}/wacc~{wacc}/{region}_{interone}.csv",
            allow_missing=True, region=config["regions"]),
    output:
        network_curve="../results/figures_general/global_supply_curve/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_demand_{demand}_{interone}.pdf",
        network_curve_png="../results/figures_general/global_supply_curve/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_demand_{demand}_{interone}.png",
        # supply_curves
    notebook:
        "notebooks/analysis-globalsupplycurve.ipynb"

rule plot_global_supply_all:
        input:
            expand("../results/figures_general/global_supply_curve/cost_year~{cost_year}/{wacc}/{scenario}/global_supply_curve_{sort}_demand_{demand}_{interone}.pdf", cost_year=[2050], wacc=["regional"], interone=["hbi"], scenario=["default"], sort=[True,False], demand=[True,False], allow_missing=True)

rule plot_comparison:
    input:
        default = "results/chain_id~labour_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~default/network.nc",
        stability = "results/chain_id~labour_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~mga-stability-weighted/network_0.02.nc",
        hightrans = "results/chain_id~hightrans_2050/cost_year~2050/interone~hbi/intertwo~eaf/wacc~regional/final~steel/scenario~default/network.nc",
    output:
        cost_comparison="../results/figures_general/comparison/cost_comparison.pdf",
        cost_comparison_png="../results/figures_general/comparison/cost_comparison.png",
    notebook:
        "notebooks/compare-scenarios.ipynb"

    
rule plot_compare_lcox:
    input:
        eu_01 = "resources/lco-hbi/cost_year~2050/wacc~regional/Europe_unreserved/network_0.1.nc",
        eu_1 = "resources/lco-hbi/cost_year~2050/wacc~regional/Europe_unreserved/network_1.nc",
        eu_10 = "resources/lco-hbi/cost_year~2050/wacc~regional/Europe_unreserved/network_10.nc",
        eu_100 = "resources/lco-hbi/cost_year~2050/wacc~regional/Europe_unreserved/network_100.nc",
    output:
        lcox_comparison="../results/figures_general/comparison/lcox_comparison.pdf",
        lcox_comparison_png="../results/figures_general/comparison/lcox_comparison.png",
    notebook:
        "notebooks/plot-compare-lcox.ipynb"