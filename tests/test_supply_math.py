import pytest

from workflow.scripts.prepare_regional_network import (
    back_propagate_electricity_need,
    filter_by_technologies,
)


def test_supply_stage_electricity_chain_matches_hand_calculation(
    steel_tech_costs, steel_minimal_case
):
    renewable_electricity_per_t = back_propagate_electricity_need(
        steel_tech_costs,
        "steel",
        {"eaf_electricity_source": "grid"},
    )

    assert renewable_electricity_per_t == pytest.approx(
        steel_minimal_case["expected_steel_renewable_electricity_mwh_per_t"]
    )


def test_supply_stage_includes_eaf_electricity_when_requested(
    steel_tech_costs, steel_minimal_case
):
    renewable_electricity_per_t = back_propagate_electricity_need(
        steel_tech_costs,
        "steel",
        {"eaf_electricity_source": "renewable"},
    )

    assert renewable_electricity_per_t == pytest.approx(
        steel_minimal_case["expected_steel_total_electricity_mwh_per_t"]
    )


def test_supply_stage_uses_1_t_per_hour_and_is_easy_to_verify(
    steel_minimal_case,
    steel_tech_costs,
):
    hourly_demand_tph = steel_minimal_case["demand_mt_per_year"] * 1e6 / 8760

    assert hourly_demand_tph == pytest.approx(1.0)
    assert steel_minimal_case["annual_demand_t"] == pytest.approx(8760.0)

    renewable_electricity_per_t = back_propagate_electricity_need(
        steel_tech_costs,
        steel_minimal_case["product"],
        {"eaf_electricity_source": "grid"},
    )

    annual_renewable_electricity_mwh = (
        hourly_demand_tph * 8760 * renewable_electricity_per_t
    )

    assert annual_renewable_electricity_mwh == pytest.approx(
        steel_minimal_case["expected_annual_renewable_electricity_mwh"]
    )


def test_supply_stage_keeps_only_allowed_renewable_technologies(
    renewable_region_timeseries_fixture,
):
    filtered = filter_by_technologies(
        renewable_region_timeseries_fixture,
        {"renewable_technologies": ["solar", "onwind"]},
    )

    assert list(filtered.technology.values) == ["solar", "onwind"]
    assert "offwind" not in set(str(tech) for tech in filtered.technology.values)
