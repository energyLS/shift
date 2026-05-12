"""Root Snakemake entrypoint for the Shift workflow.

Loads shared configuration and includes the modular rule files for supply curves,
trade optimization, and reporting.
"""

from pathlib import Path

import pandas as pd
from snakemake.utils import Paramspace

WORKFLOW_DIR = Path(workflow.basedir) / "workflow"
SCRIPT_DIR = WORKFLOW_DIR / "scripts"


configfile: "config/config.yaml"


def _load_trade_scenarios():
    trade_chains = config.get("trade_chains")
    if trade_chains:
        rows = []
        for chain in trade_chains:
            stages = chain.get("stages", [])
            if len(stages) < 2:
                raise ValueError(
                    f"Trade chain '{chain.get('id', '<unnamed>')}' needs at least 2 stages"
                )
            stages_sorted = sorted(stages, key=lambda stage: int(stage.get("order", 0)))
            rows.append(
                {
                    "chain_id": str(chain["id"]),
                    "cost_year": str(chain["cost_year"]),
                    "interone": str(stages_sorted[0]["output_commodity"]),
                    "intertwo": str(
                        stages_sorted[1].get(
                            "process_label", stages_sorted[1]["output_commodity"]
                        )
                    ),
                    "final": str(chain["final_product"]),
                    "scenario": str(chain.get("scenario", "default")),
                }
            )
        return Paramspace(pd.DataFrame(rows, dtype=str))

    return Paramspace(pd.read_csv("config/trade_scenarios.csv", dtype=str))


trade_scenarios = _load_trade_scenarios()


def _derive_supply_curve_products():
    trade_chains = config.get("trade_chains")
    if not trade_chains:
        return ["steel"]

    products = set()
    for chain in trade_chains:
        for stage in chain.get("stages", []):
            output_commodity = stage.get("output_commodity")

            if output_commodity:
                products.add(str(output_commodity))

    return sorted(products) if products else ["steel"]


SUPPLY_CURVE_PRODUCTS = _derive_supply_curve_products()


wildcard_constraints:
    country="[a-zA-Z]+",
    sweep="[a-zA-Z]+",
    rule="(0|[1-9][0-9]?|100)",


include: "rules/supply_curves.smk"
include: "rules/trade_model.smk"
include: "rules/reporting.smk"
