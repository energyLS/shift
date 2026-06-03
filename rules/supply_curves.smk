"""Supply-curve workflow rules.

Builds technology inputs, prepares regional PyPSA networks, solves regional LCoX
problems, and aggregates the resulting supply curves.
"""

from trade_chain_utils import route_label_for_product, get_stage_groups, get_trade_chain


# Helper: find the internal route label for a product from config trade_chains
def _process_label_for_product(product):
    return route_label_for_product(config, product)


def _product_uses_renewables(product):
    """Check if a product's stage group uses renewable_electricity.

    Products with renewable inputs should generate reserved/unreserved scenarios.
    Products with only grid electricity should skip the unreserved variant.
    """
    chain = get_trade_chain(config)
    route_label = route_label_for_product(config, product)

    # Find the stage group for this product
    for group in get_stage_groups(chain):
        if group["label"] == route_label:
            # Check if any stage in the group uses renewable_electricity
            for stage in group["stages"]:
                energy_inputs = stage.get("energy_inputs", [])
                if "renewable_electricity" in energy_inputs:
                    return True
            return False

    # Default to True if product not found (conservative)
    return True


def _all_supply_curve_targets():
    targets = []
    for region in config["regions"]:
        wacc = config["trade_chains"].get("wacc", "uniform")
        for product in SUPPLY_CURVE_PRODUCTS:
            targets.append(
                f"resources/supply_curves/cost_year~2050/wacc~{wacc}/{region}_marginal_cost_{product}.csv"
            )
    return targets


rule retrieve_cost_data:
    output:
        costs="resources/technology_data/costs_{cost_year}.csv",
    threads: 1
    resources:
        mem_mb=500,
    params:
        version=config["techno-economic parameters"]["pypsa_tech_version"],
    script:
        str(SCRIPT_DIR / "tech_database.py")


rule build_generic_model:
    input:
        costs="resources/technology_data/costs_{cost_year}.csv",
    output:
        skeleton="resources/generic_production_model/generic_model_{cost_year}.nc",
        # Also produce one skeleton per detected stage-group so Snakemake tracks them
        group_skeletons=[
            f"resources/generic_production_model/generic_model_{{cost_year}}_{(g.get('label') or 'group')}.nc"
            for g in get_stage_groups(get_trade_chain(config))
        ],
    threads: 1
    resources:
        mem_mb=1000,
    script:
        str(SCRIPT_DIR / "build_x_supply_chain.py")


rule prepare_regional_network:
    input:
        # Prefer a per-stage-group skeleton when a route_label exists for the product;
        # otherwise fall back to the legacy full skeleton.
        skeleton=lambda wildcards: (
            f"resources/generic_production_model/generic_model_{wildcards.cost_year}_{_process_label_for_product(wildcards.product)}.nc"
            if _process_label_for_product(wildcards.product)
            else f"resources/generic_production_model/generic_model_{wildcards.cost_year}.nc"
        ),
        renewables="data/renewables_clustered.nc",
        tech_costs="resources/technology_data/costs_{cost_year}.csv",
        local_demand="data/un_enerdata_demand_2050_final.csv",
        wacc="resources/wacc-clustered.csv",
        labour_cost="resources/labour_cost_clustered.csv",
    output:
        # Output keyed by product; route_label is internal to the script
        network="resources/networks/base_{cost_year}_{region}_{wacc}_{product}_{scenario}.nc",
    log:
        "logs/prepare_regional_network_{cost_year}_{region}_{wacc}_{product}_{scenario}.log",
    wildcard_constraints:
        scenario="reserved|unreserved",
        product="hbi|steel",
    threads: 1
    resources:
        mem_mb=2000,
    params:
        region="{region}",
        product="{product}",
        route_label=lambda wildcards: _process_label_for_product(wildcards.product),
        config=config,
        uniform_interest_rate=config["interest_rate"]["default"],
    message:
        "Preparing {wildcards.scenario} regional network: {wildcards.region} -> {wildcards.product} "
        "(cost_year={wildcards.cost_year})"
    script:
        str(SCRIPT_DIR / "prepare_regional_network.py")


if config["enable"].get("run_supply_chain", True):

    rule calculate_regional_lcox:
        input:
            base_network="resources/networks/base_{cost_year}_{region}_{wacc}_{product}_{scenario}.nc",
            local_demand="data/un_enerdata_demand_2050_final.csv",
        output:
            # Internal cache keyed by route_label for reuse; only products matter for supply curves
            results="resources/lco-{product}/cost_year~{cost_year}/wacc~{wacc}/{region}_{scenario}/results_{product_demand_mt}.csv",
            network=(
                temp(
                    "resources/lco-{product}/cost_year~{cost_year}/wacc~{wacc}/{region}_{scenario}/network_{product_demand_mt}.nc"
                )
                if not config.get("outputs", {}).get(
                    "keep_optimization_networks", False
                )
                else "resources/lco-{product}/cost_year~{cost_year}/wacc~{wacc}/{region}_{scenario}/network_{product_demand_mt}.nc"
            ),
        log:
            "logs/calculate_regional_lcox_{cost_year}_{region}_{wacc}_{product}_{scenario}_{product_demand_mt}.log",
        wildcard_constraints:
            product_demand_mt=r"\d+(?:\.\d+)?",
            scenario="reserved|unreserved",
            product="hbi|steel",
        threads: 2
        resources:
            mem_mb=4000,
        params:
            product_demand_mt="{product_demand_mt}",
            compute_iis=config.get("solver", {}).get("compute_iis", False),
            product="{product}",
            route_label=lambda wildcards: _process_label_for_product(wildcards.product),
        message:
            "Calculating LCoX ({wildcards.scenario}) for {wildcards.product} in {wildcards.region} "
            "(demand={wildcards.product_demand_mt} Mt/year)."
        script:
            str(SCRIPT_DIR / "calculate_lcox.py")


if config["enable"].get("run_supply_curve", True):

    rule create_supply_curve:
        input:
            lco_reserved=lambda wildcards: expand(
                f"resources/lco-{wildcards.product}/cost_year~{wildcards.cost_year}/wacc~{wildcards.wacc}/{wildcards.region}_reserved/results_{{product_demand_mt}}.csv",
                product_demand_mt=config.get("steel_demand_levels"),
            ),
            lco_unreserved=lambda wildcards: (
                expand(
                    f"resources/lco-{wildcards.product}/cost_year~{wildcards.cost_year}/wacc~{wildcards.wacc}/{wildcards.region}_unreserved/results_{{product_demand_mt}}.csv",
                    product_demand_mt=config.get("steel_demand_levels"),
                )
                if config.get("supply_curve", {}).get("generate_unreserved", False)
                and _product_uses_renewables(wildcards.product)
                else []
            ),
            skeleton="resources/generic_production_model/generic_model_{cost_year}.nc",
            steel_demand="resources/steel_demand_clustered_{cost_year}.csv",
        output:
            # Public supply-curve artifact is product-labeled; the stage label
            # is only used to locate the correct upstream LCoX runs.
            supply="resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{product}.csv",
            supply_unreserved=(
                "resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{product}__unreserved.csv"
                if config.get("supply_curve", {}).get("generate_unreserved", False)
                and _product_uses_renewables("{product}")
                else temp(
                    "resources/supply_curves_unreserved_tmp/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{product}__unreserved.csv"
                )
            ),
            supply_curve="resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{product}.pdf",
        log:
            "logs/create_supply_curve_{cost_year}_{region}_{product}_{wacc}.log",
        wildcard_constraints:
            product="hbi|steel",
            wacc="uniform|regional",
        threads: 1
        message:
            "Combining LCo results (reserved + unreserved scenarios) to create supply curve for {wildcards.region} {wildcards.product}."
        script:
            str(SCRIPT_DIR / "create_supply_curve.py")


rule create_all_supply_curves:
    input:
        _all_supply_curve_targets(),
