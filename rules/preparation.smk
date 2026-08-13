


rule download_labour_data:
    output:
        merged="resources/merged_labour_inputs.csv",
    threads: 1
    resources:
        mem_mb=2000,
    script:
        str(SCRIPT_DIR / "download_labour_data.py")


rule prepare_labour_cost:
    input:
        merged="resources/merged_labour_inputs.csv",
    output:
        labour_cost="resources/labour_cost_clustered.csv",
    threads: 1
    resources:
        mem_mb=2000,
    script:
        str(SCRIPT_DIR / "prepare_labour_cost.py")


rule prepare_wacc:
    input:
        wacc="data/wacc-global.csv",
        bus_locations="data/bus_locations.csv",
    output:
        wacc="resources/wacc-clustered.csv",
        wacc_latex="resources/wacc-clustered.tex",
    threads: 2
    resources:
        mem_mb=5000,
    notebook:
        # "notebooks/prepare-wacc.ipynb"
        str(NOTEBOOKS_DIR / "prepare-wacc.ipynb")


rule prepare_political_stability:
    input:
        political_stability="data/political-stability/globaleconomy.csv",  #https://www.theglobaleconomy.com/rankings/wb_political_stability/
        bus_locations="data/bus_locations.csv",
    output:
        political_stability="resources/political_stability_clustered.csv",
    threads: 2
    resources:
        mem_mb=5000,
    notebook:
        str(NOTEBOOKS_DIR / "prepare-political-stability.ipynb")


rule prepare_chokepoints:
    input:
        trade_options="data/trade_opt.csv",
        bus_locations="data/bus_locations.csv",
    output:
        trade_options_chokepoints="resources/trade_opt_chokepoints.csv",
        map_chokepoints="results/figures_general/chokepoints/map_chokepoints.pdf",
        map_chokepoints_png="results/figures_general/chokepoints/map_chokepoints.png",
    threads: 2
    resources:
        mem_mb=5000,
    params:
        shipping_routes=config["trade"]["shipping_routes"],
    notebook:
        str(NOTEBOOKS_DIR / "prepare-chokepoints.ipynb")


rule retrieve_iron_ore:
    input:
        iron_ore_production="data/owid-iron-ore/iron-ore-crude-ore-production.csv",
        iron_ore_cost="data/devlin2023-supplementary.xlsx",
        bus_locations="data/bus_locations.csv",
    output:
        iron_ore="resources/ironore-production.csv",
        iron_ore_map="results/figures_general/iron_ore_map.pdf",
    threads: 2
    resources:
        mem_mb=5000,
    notebook:
        str(NOTEBOOKS_DIR / "global-iron-ore.ipynb")


rule prepare_iron_ore:
    input:
        iron_ore="resources/ironore-production.csv",
        bus_locations="data/bus_locations.csv",
    output:
        iron_ore="resources/ironore_production_clustered.csv",
    threads: 2
    resources:
        mem_mb=5000,
    notebook:
        str(NOTEBOOKS_DIR / "prepare-iron-ore.ipynb")


rule prepare_steel_demand:
    input:
        steel_demand="data/demand/steel_demands/output_data/country_raw_steel_demand_and_dri_share.csv",
        bus_locations="data/bus_locations.csv",
    output:
        steel_demand="resources/steel_demand_clustered_{cost_year}.csv",
    threads: 2
    resources:
        mem_mb=5000,
    notebook:
        str(NOTEBOOKS_DIR / "prepare-steel-demand.ipynb")


if config["enable"].get("cluster_renewables", True):

    rule cluster_renewables:
        input:
            merged_cdf="data/renewable_profiles_global_merged.nc",
            merged_geojson="data/renewable_profiles_global_merged.geojson",
        output:
            clustered="resources/renewables_clustered.nc",
            report="resources/renewables_clustering_report.json",
        threads: 4
        resources:
            mem_mb=16000,
            time_min=60,
        script:
            str(SCRIPT_DIR / "cluster_renewables.py")
