"""Supply-curve workflow rules.

Builds technology inputs, prepares regional PyPSA networks, solves regional LCoX
problems, and aggregates the resulting supply curves.
"""


# Helper: find the process_label for a product from config trade_chains
def _process_label_for_product(product):
    chains = config.get("trade_chains") or []
    for chain in chains:
        stages = chain.get("stages", [])
        for s in stages:
            if s.get("output_commodity") == product:
                return s.get("process_label")
    return product


# Helper: find the product that a process_label stage belongs to
def _product_for_process_label(process_label):
    """Return the product (output_commodity) for a given process_label."""
    chains = config.get("trade_chains") or []
    for chain in chains:
        stages = chain.get("stages", [])
        for s in stages:
            if s.get("process_label") == process_label:
                return s.get("output_commodity")
    # Fallback: treat process_label as product itself
    return process_label


# Helper: get all process_labels (stages) for a given product
def _process_labels_for_product(product_name):
    """Return list of process_labels that output to product_name."""
    chains = config.get("trade_chains") or []
    labels = []
    for chain in chains:
        stages = chain.get("stages", [])
        for s in stages:
            if s.get("output_commodity") == product_name:
                labels.append(s.get("process_label"))
    return (
        labels if labels else [product_name]
    )  # Fallback: if no chain, use product name


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
        network="resources/networks/base_{cost_year}_{region}_{process_label}_{scenario}.nc",
    log:
        "logs/prepare_regional_network_{cost_year}_{region}_{process_label}_{scenario}.log",
    wildcard_constraints:
        scenario="reserved|unreserved",
        process_label="hbi|steel",
    threads: 1
    resources:
        mem_mb=2000,
    params:
        region="{region}",
        # Derive product from process_label (hbi → hbi, steel → steel)
        product=lambda wildcards: wildcards.process_label,
        process_label="{process_label}",
        config=config,
    message:
        "Preparing {wildcards.scenario} regional network: {wildcards.region} -> {wildcards.process_label} "
        "(cost_year={wildcards.cost_year})"
    script:
        str(SCRIPT_DIR / "prepare_regional_network.py")


if config["enable"].get("run_supply_chain", True):

    rule calculate_regional_lcox:
        input:
            base_network="resources/networks/base_{cost_year}_{region}_{process_label}_{scenario}.nc",
            local_demand="data/un_enerdata_demand_2050_final.csv",
        output:
            results="resources/lco-{process_label}/cost_year~{cost_year}/{region}_{scenario}/results_{product_demand_mt}.csv",
            network=(
                temp(
                    "resources/lco-{process_label}/cost_year~{cost_year}/{region}_{scenario}/network_{product_demand_mt}.nc"
                )
                if not config.get("outputs", {}).get(
                    "keep_optimization_networks", False
                )
                else "resources/lco-{process_label}/cost_year~{cost_year}/{region}_{scenario}/network_{product_demand_mt}.nc"
            ),
        log:
            "logs/calculate_regional_lcox_{cost_year}_{region}_{process_label}_{scenario}_{product_demand_mt}.log",
        wildcard_constraints:
            product_demand_mt=r"\d+(?:\.\d+)?",
            scenario="reserved|unreserved",
            process_label="hbi|steel",
        threads: 2
        resources:
            mem_mb=4000,
        params:
            product_demand_mt="{product_demand_mt}",
            compute_iis=config.get("solver", {}).get("compute_iis", False),
            process_label="{process_label}",
        message:
            "Calculating LCoX ({wildcards.scenario}) for {wildcards.process_label} in {wildcards.region} "
            "(demand={wildcards.product_demand_mt} Mt/year)."
        script:
            str(SCRIPT_DIR / "calculate_lcox.py")


if config["enable"].get("run_supply_curve", True):

    rule create_supply_curve:
        input:
            lco_reserved=lambda wildcards: expand(
                f"resources/lco-{wildcards.process_label}/cost_year~{wildcards.cost_year}/{wildcards.region}_reserved/results_{{product_demand_mt}}.csv",
                product_demand_mt=config.get("steel_demand_levels"),
            ),
            lco_unreserved=lambda wildcards: (
                expand(
                    f"resources/lco-{wildcards.process_label}/cost_year~{wildcards.cost_year}/{wildcards.region}_unreserved/results_{{product_demand_mt}}.csv",
                    product_demand_mt=config.get("steel_demand_levels"),
                )
                if config.get("supply_curve", {}).get("generate_unreserved", False)
                else []
            ),
            skeleton="resources/steel_skeleton/steel_skeleton_{cost_year}.nc",
            local_demand="data/un_enerdata_demand_2050_final.csv",
            steel_demand="resources/steel_production_clustered.csv",
        output:
            # Output files now use process_label instead of product
            supply="resources/supply_curves/cost_year~{cost_year}/{region}_marginal_cost_{process_label}.csv",
            supply_unreserved=(
                "resources/supply_curves/cost_year~{cost_year}/{region}_marginal_cost_{process_label}__unreserved.csv"
                if config.get("supply_curve", {}).get("generate_unreserved", False)
                else temp(
                    "resources/supply_curves_unreserved_tmp/cost_year~{cost_year}/{region}_marginal_cost_{process_label}__unreserved.csv"
                )
            ),
            supply_curve="resources/supply_curves/cost_year~{cost_year}/{region}_marginal_cost_{process_label}.pdf",
        log:
            "logs/create_supply_curve_{cost_year}_{region}_{process_label}.log",
        wildcard_constraints:
            process_label="hbi|steel",
        threads: 1
        message:
            "Combining LCo results (reserved + unreserved scenarios) to create supply curve for {wildcards.region} {wildcards.process_label}."
        script:
            str(SCRIPT_DIR / "create_supply_curve.py")


rule create_all_supply_curves:
    input:
        lambda wildcards: expand(
            "resources/supply_curves/cost_year~{cost_year}/{region}_marginal_cost_{process_label}.csv",
            cost_year=[2050],
            region=config["regions"],
            process_label=[
                pl
                for product in SUPPLY_CURVE_PRODUCTS
                for pl in _process_labels_for_product(product)
            ],
            allow_missing=True,
        ),
