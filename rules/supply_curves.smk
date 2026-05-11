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
        local_demand="data/un_enerdata_demand_2050_final.csv",
    output:
        network="resources/networks/base_{cost_year}_{region}_{product}_{scenario}.nc",
    log:
        "logs/prepare_regional_network_{cost_year}_{region}_{product}_{scenario}.log",
    wildcard_constraints:
        scenario="reserved|unconstrained",
    threads: 1
    resources:
        mem_mb=2000,
    params:
        region="{region}",
        product="{product}",
        config=config,
    message:
        "Preparing {wildcards.scenario} regional network: {wildcards.region} -> {wildcards.product} "
        "(cost_year={wildcards.cost_year})"
    script:
        str(SCRIPT_DIR / "prepare_regional_network.py")


if config["enable"].get("run_supply_chain", True):

    rule calculate_regional_lcox:
        input:
            base_network="resources/networks/base_{cost_year}_{region}_{product}_{scenario}.nc",
            local_demand="data/un_enerdata_demand_2050_final.csv",
        output:
            results="resources/lco-{product}/cost_year~{cost_year}/{region}_{scenario}/results_{product_demand_mt}.csv",
            network=(
                temp(
                    "resources/lco-{product}/cost_year~{cost_year}/{region}_{scenario}/network_{product_demand_mt}.nc"
                )
                if not config.get("outputs", {}).get(
                    "keep_optimization_networks", False
                )
                else "resources/lco-{product}/cost_year~{cost_year}/{region}_{scenario}/network_{product_demand_mt}.nc"
            ),
        log:
            "logs/calculate_regional_lcox_{cost_year}_{region}_{product}_{scenario}_{product_demand_mt}.log",
        wildcard_constraints:
            product_demand_mt=r"\d+(?:\.\d+)?",
            scenario="reserved|unconstrained",
        threads: 2
        resources:
            mem_mb=4000,
        params:
            product_demand_mt="{product_demand_mt}",
            compute_iis=config.get("solver", {}).get("compute_iis", False),
        message:
            "Calculating LCoX ({wildcards.scenario}) for {wildcards.product} in {wildcards.region} "
            "(demand={wildcards.product_demand_mt} Mt/year)."
        script:
            str(SCRIPT_DIR / "calculate_lcox.py")


if config["enable"].get("run_supply_curve", True):

    rule create_supply_curve:
        input:
            lco_reserved=lambda wildcards: expand(
                f"resources/lco-{wildcards.product}/cost_year~{wildcards.cost_year}/{wildcards.region}_reserved/results_{{product_demand_mt}}.csv",
                product_demand_mt=config.get("steel_demand_levels"),
            ),
            lco_unconstrained=lambda wildcards: (
                expand(
                    f"resources/lco-{wildcards.product}/cost_year~{wildcards.cost_year}/{wildcards.region}_unconstrained/results_{{product_demand_mt}}.csv",
                    product_demand_mt=config.get("steel_demand_levels"),
                )
                if config.get("supply_curve", {}).get("generate_unconstrained", False)
                else []
            ),
            local_demand="data/un_enerdata_demand_2050_final.csv",
            steel_demand="resources/steel_production_clustered.csv",
        output:
            supply="resources/supply_curves/cost_year~{cost_year}/{region}_{product}.csv",
            supply_unconstrained=(
                "resources/supply_curves_unconstrained/cost_year~{cost_year}/{region}_{product}.csv"
                if config.get("supply_curve", {}).get("generate_unconstrained", False)
                else temp(
                    "resources/supply_curves_unconstrained_tmp/cost_year~{cost_year}/{region}_{product}.csv"
                )
            ),
            supply_curve="resources/supply_curves/cost_year~{cost_year}/{region}_{product}.pdf",
        log:
            "logs/create_supply_curve_{cost_year}_{region}_{product}.log",
        threads: 1
        message:
            "Combining LCo{wildcards.product[0]} results (reserved + unconstrained scenarios) to create supply curve for {wildcards.region}."
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
