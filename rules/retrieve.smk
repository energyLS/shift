storage:
    provider="http",
    keep_local=True,
    retries=3,

rule retrieve_data:
    input:
        data_url=storage(
            f"https://zenodo.org/records/"
        ),
    output:
        bus_locations="data/bus_locations.csv",
        ip_market_fabrication="data/demand/ip_market__fabrication.csv",
        steel_demand_dri_share="data/demand/steel_demands/output_data/country_raw_steel_demand_and_dri_share.csv",
        devlin2023_supplementary="data/devlin2023-supplementary.xlsx",
        grid_potential_custom="data/grid_potential_custom.csv",
        unido_raw="data/labour/unido-raw/data.csv",
        iron_ore_production="data/owid-iron-ore/iron-ore-crude-ore-production.csv",
        political_stability="data/political-stability/globaleconomy.csv",
        renewable_profiles_geojson="data/renewable_profiles_global_merged.geojson",
        renewable_profiles_nc="data/renewable_profiles_global_merged.nc",
    log:
        "logs/retrieve_data.log",
    retries: 2
    resources:
        mem_mb=10000,
    params:
        folder=".",
    run:
        unpack_archive(input.data_url, params.folder)