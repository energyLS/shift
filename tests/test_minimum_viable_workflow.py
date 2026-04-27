import pytest

from workflow.scripts.prepare_regional_network import back_propagate_electricity_need


def test_supply_stage_electricity_chain_matches_hand_calculation(
    steel_tech_costs, steel_minimal_case
):
    electricity_per_t = back_propagate_electricity_need(
        steel_tech_costs,
        "steel",
        {},
    )

    assert electricity_per_t == pytest.approx(
        steel_minimal_case["expected_steel_electricity_mwh_per_t"]
    )


def test_supply_stage_uses_1_t_per_hour_and_is_easy_to_verify(
    steel_minimal_case,
    steel_tech_costs,
):
    hourly_demand_tph = steel_minimal_case["demand_mt_per_year"] * 1e6 / 8760

    assert hourly_demand_tph == pytest.approx(1.0)
    assert steel_minimal_case["annual_demand_t"] == pytest.approx(8760.0)

    electricity_per_t = back_propagate_electricity_need(
        steel_tech_costs,
        steel_minimal_case["product"],
        {},
    )

    annual_electricity_mwh = hourly_demand_tph * 8760 * electricity_per_t

    assert annual_electricity_mwh == pytest.approx(
        steel_minimal_case["expected_annual_electricity_mwh"]
    )


def test_trade_stage_minimum_cost_route_is_obvious(toy_trade_case):
    delivered_costs = {
        origin: offer + toy_trade_case["transport_costs"][(origin, "B")]
        for origin, offer in toy_trade_case["offers"].items()
    }

    chosen_origin = min(delivered_costs, key=delivered_costs.get)

    assert chosen_origin == toy_trade_case["expected_choice"]
    assert delivered_costs[chosen_origin] == pytest.approx(
        toy_trade_case["expected_delivered_cost"]
    )
