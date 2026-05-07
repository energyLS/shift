"""Reporting workflow rules.

Collects final figures and presentation artifacts produced by notebooks and the
main optimization workflow.
"""


rule collect_figures:
    input:
        global_supply_curve="results/figures_general/{scenario}/global_supply_curve.pdf",
        global_supply_curve_png="results/figures_general/{scenario}/global_supply_curve.png",
        electricity_demand="results/figures_general/electricity_demand.pdf",
        electricity_demand_png="results/figures_general/electricity_demand.png",
        electricity_demand_steel="results/figures_general/electricity_demand_in_steel.pdf",
        electricity_demand_steel_png="results/figures_general/electricity_demand_in_steel.png",
        global_map_countries="results/figures_general/global_map_countries.pdf",
        global_map_countries_png="results/figures_general/global_map_countries.png",
        cost_comparison="results/figures_general/comparison/cost_comparison.pdf",
        cost_comparison_png="results/figures_general/comparison/cost_comparison.png",
        value_chain_comparison="results/figures_general/value_chain_comparison.pdf",
        value_chain_comparison_png="results/figures_general/value_chain_comparison.png",
        hourly_analysis="results/figures_general/hourly_analysis.pdf",
        hourly_analysis_png="results/figures_general/hourly_analysis.png",
