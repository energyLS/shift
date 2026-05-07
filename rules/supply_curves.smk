"""Supply-curve workflow rules.

Builds technology inputs, prepares regional PyPSA networks, solves regional LCoX
problems, and aggregates the resulting supply curves.
"""


rule retrieve_cost_data:
    output:
        costs="resources/technology_data/costs_{cost_year}.csv",
    threads: 1
    resources:
        mem_mb=500,
    params:
        version=config["costs"]["version"],
    script:
        str(SCRIPT_DIR / "tech_database.py")


rule build_steel_skeleton:
    input:
        costs="resources/technology_data/costs_{cost_year}.csv",
    output:
        skeleton="resources/steel_skeleton/steel_skeleton_{cost_year}.nc",
    threads: 1
    resources:
        mem_mb=1000,
    script:
        str(SCRIPT_DIR / "build_x_supply_chain.py")


rule prepare_regional_network:
    input:
        skeleton="resources/steel_skeleton/steel_skeleton_{cost_year}.nc",
        renewables="data/new_renewables_consolidated.nc",
        tech_costs="resources/technology_data/costs_{cost_year}.csv",
    output:
        network="resources/networks/base_{cost_year}_{region}_{product}.nc",
    log:
        "logs/prepare_regional_network_{cost_year}_{region}_{product}.log",
    threads: 1
    resources:
        mem_mb=2000,
    params:
        region="{region}",
        product="{product}",
        config=config,
    message:
        "Preparing regional network: {wildcards.region} → {wildcards.product} "
        "(cost_year={wildcards.cost_year})"
    script:
        str(SCRIPT_DIR / "prepare_regional_network.py")


if config["enable"].get("run_supply_chain", True):

    rule calculate_regional_lcox:
        input:
            base_network="resources/networks/base_{cost_year}_{region}_{product}.nc",
            local_demand="data/un_enerdata_demand_2050_final.csv",
        output:
            results="resources/lco-{product}/cost_year~{cost_year}/{region}/results_{product_demand_mt}.csv",
            network=(
                temp(
                    "resources/lco-{product}/cost_year~{cost_year}/{region}/network_{product_demand_mt}.nc"
                )
                if not config.get("outputs", {}).get(
                    "keep_optimization_networks", False
                )
                else "resources/lco-{product}/cost_year~{cost_year}/{region}/network_{product_demand_mt}.nc"
            ),
        wildcard_constraints:
            product_demand_mt=r"\d+(?:\.\d+)?",
        threads: 2
        resources:
            mem_mb=4000,
        params:
            product_demand_mt="{product_demand_mt}",
            compute_iis=config.get("solver", {}).get("compute_iis", False),
        message:
            "Calculating LCoX for {wildcards.product} in region {wildcards.region} "
            "(product_demand={wildcards.product_demand_mt} Mt/year)."
        script:
            str(SCRIPT_DIR / "calculate_lcox.py")


if config["enable"].get("run_supply_curve", True):

    rule create_supply_curve:
        input:
            lco_product_data=lambda wildcards: expand(
                f"resources/lco-{wildcards.product}/cost_year~{wildcards.cost_year}/{wildcards.region}/results_{{product_demand_mt}}.csv",
                product_demand_mt=config.get("steel_demand_levels"),
            ),
            local_demand="data/un_enerdata_demand_2050_final.csv",
            steel_demand="resources/steel_production_clustered.csv",
        output:
            supply="resources/supply_curves/cost_year~{cost_year}/{region}_{product}.csv",
            supply_nodemand=(
                "resources/supply_curves_nodemand/cost_year~{cost_year}/{region}_{product}.csv"
                if config.get("outputs", {}).get("save_supply_nodemand", True)
                else temp(
                    "resources/supply_curves_nodemand_tmp/{region}_{product}.csv"
                )
            ),
            supply_curve="resources/supply_curves/cost_year~{cost_year}/{region}_{product}.pdf",
        threads: 1
        message:
            "Combining LCo{wildcards.product[0]} results (all product demand levels) to create supply curve for {wildcards.region}."
        script:
            str(SCRIPT_DIR / "create_supply_curve.py")


rule create_all_supply_curves:
    input:
        expand(
            "resources/supply_curves/cost_year~{cost_year}/{region}_{product}.csv",
            cost_year=[2050],
            region=config["regions"],
            product=["steel"],
            allow_missing=True,
        ),
