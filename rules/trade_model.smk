"""Trade-model workflow rules.

Consumes supply curves and scenario inputs to run the interregional trade model
and collect the scenario-level outputs.
"""


rule model_trade:
    input:
        supply_curves_interone=expand(
            "resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{interone}.csv",
            allow_missing=True,
            cost_year=[2050],
            region=config["regions"],
            interone=["hbi"],
        ),
        supply_curves_intertwo=expand(
            "resources/supply_curves/cost_year~{cost_year}/wacc~{wacc}/{region}_marginal_cost_{intertwo}.csv",
            allow_missing=True,
            cost_year=[2050],
            region=config["regions"],
            intertwo=["steel"],
        ),
        trade_options="resources/trade_opt_chokepoints.csv",
        bus_locations="data/bus_locations.csv",
        demand="data/un_enerdata_demand_2050_final.csv",
        steel_demand="resources/steel_demand_clustered_{cost_year}.csv",
        iron_ore="resources/ironore_production_clustered.csv",
        grid_potential="data/grid_potential_custom.csv",
        political_stability="resources/political_stability_clustered.csv",
    output:
        trade_result=f"results/{trade_scenarios.wildcard_pattern}/result.csv",
        trade_network=f"results/{trade_scenarios.wildcard_pattern}/network.nc",
        trade_plot_ironore=f"results/{trade_scenarios.wildcard_pattern}/map_ironore.pdf",
        trade_plot_ironore_png=f"results/{trade_scenarios.wildcard_pattern}/map_ironore.png",
        trade_plot_hbi=f"results/{trade_scenarios.wildcard_pattern}/map_hbi.pdf",
        trade_plot_hbi_png=f"results/{trade_scenarios.wildcard_pattern}/map_hbi.png",
        trade_plot_steel=f"results/{trade_scenarios.wildcard_pattern}/map_steel.pdf",
        trade_plot_steel_png=f"results/{trade_scenarios.wildcard_pattern}/map_steel.png",
    threads: 4
    params:
        iron_ore_potential=config["iron_ore"]["potential_allowance"],
        cost_penalty=config["design"]["cost_penalty"],
        scenarios=config["scenario"],
    script:
        str(SCRIPT_DIR / "model_trade.py")



rule model_trade_all:
    input:
        networks=expand(
            "results/{scenarios}/network.nc",
            scenarios=trade_scenarios.instance_patterns,
        ),
        results=expand(
            "results/{scenarios}/result.csv",
            scenarios=trade_scenarios.instance_patterns,
        ),
        trade_plot_ironore=expand(
            "results/{scenarios}/map_ironore.pdf",
            scenarios=trade_scenarios.instance_patterns,
        ),
        trade_plot_steel=expand(
            "results/{scenarios}/map_steel.pdf",
            scenarios=trade_scenarios.instance_patterns,
        ),
