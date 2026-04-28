from types import SimpleNamespace

import pandas as pd

import workflow.scripts.create_supply_curve as create_supply_curve_module


def test_supply_curve_smoke(tmp_path, monkeypatch):
    lco_file_1 = tmp_path / "results_10.csv"
    lco_file_2 = tmp_path / "results_20.csv"
    local_demand_file = tmp_path / "local_demand.csv"
    steel_demand_file = tmp_path / "steel_demand.csv"
    supply_file = tmp_path / "supply.csv"
    supply_nodemand_file = tmp_path / "supply_nodemand.csv"
    supply_curve_file = tmp_path / "supply_curve.pdf"

    pd.DataFrame(
        [
            {
                "demand [t]": 10.0,
                "load [t/h]": 1.0,
                "cost [EUR]": 1000.0,
                "lcox [EUR/t]": 100.0,
            }
        ]
    ).to_csv(lco_file_1, index=False)

    pd.DataFrame(
        [
            {
                "demand [t]": 20.0,
                "load [t/h]": 2.0,
                "cost [EUR]": 2000.0,
                "lcox [EUR/t]": "infeasible",
            }
        ]
    ).to_csv(lco_file_2, index=False)

    pd.DataFrame(
        [
            {
                "region": "TST",
                "demand": 0.0,
                "unit": "MWh",
                "el_share": 0.0,
                " note": "none",
            }
        ]
    ).to_csv(local_demand_file, index=False)
    pd.DataFrame([{"region": "TST", "SteelProductionMt": 0.0}]).to_csv(
        steel_demand_file, index=False
    )

    monkeypatch.setattr(
        create_supply_curve_module,
        "snakemake",
        SimpleNamespace(
            input=SimpleNamespace(
                lco_product_data=[str(lco_file_1), str(lco_file_2)],
                local_demand=str(local_demand_file),
                steel_demand=str(steel_demand_file),
            ),
            output=SimpleNamespace(
                supply=str(supply_file),
                supply_nodemand=str(supply_nodemand_file),
                supply_curve=str(supply_curve_file),
            ),
            wildcards={"region": "TST", "product": "steel"},
            config={
                "electricity_steel_ratio": 1.0,
                "iron_ore": {"marginal_cost": 0.0, "ore_to_steel_ratio": 0.0},
            },
        ),
        raising=False,
    )

    monkeypatch.setattr(
        create_supply_curve_module,
        "product",
        "steel",
        raising=False,
    )
    monkeypatch.setattr(
        create_supply_curve_module,
        "columns",
        {
            "demand factor": "demand factor [%]",
            "demand": "demand [t]",
            "load": "load [t/h]",
            "total cost": "cost [EUR]",
            "cost per unit": "lcox [EUR/t]",
            "xlabel": "Demand in Mt",
            "product_unit": "t",
            "ylim": (0, 900),
        },
        raising=False,
    )

    create_supply_curve_module.create_supply_curve()

    assert supply_file.exists()
    assert supply_nodemand_file.exists()
    assert supply_curve_file.exists()

    output_df = pd.read_csv(supply_file)
    assert len(output_df) == 1
    assert float(output_df.iloc[0]["demand [t]"]) == 10.0
    assert float(output_df.iloc[0]["lcox [EUR/t]"]) == 100.0
