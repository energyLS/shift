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


trade_scenarios = Paramspace(pd.read_csv("config/trade_scenarios.csv", dtype=str))


wildcard_constraints:
    country="[a-zA-Z]+",
    sweep="[a-zA-Z]+",
    rule="(0|[1-9][0-9]?|100)",


include: "rules/supply_curves.smk"
include: "rules/trade_model.smk"
include: "rules/reporting.smk"
