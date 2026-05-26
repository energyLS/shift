# SPDX-FileCopyrightText: Contributors to shift <https://github.com/shift-project/shift>
#
# SPDX-License-Identifier: MIT
"""
Technology database utilities: Download and query PyPSA cost data.

Dual-purpose module: Snakemake rule for downloading tech costs + importable utilities.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from _helpers import setup_logging

snakemake: Any = globals().get("snakemake")
logger = setup_logging(__name__, snakemake=snakemake, log_filename="tech_database.log")


def download_tech_database(
    version: str, output_path: str, disable_progress: bool = False
) -> None:
    """Download PyPSA technology-data from GitHub. Supports standard versions or custom paths."""
    from _helpers import progress_retrieve

    # Construct URL based on version format (matches retrieve_cost_data.py logic)
    if "/" in version:
        # Custom GitHub path: "owner/repo/branch" -> https://raw.githubusercontent.com/owner/repo/branch/outputs/
        baseurl = f"https://raw.githubusercontent.com/{version}/outputs/"
    else:
        # Default PyPSA path: "v0.5.0" -> https://raw.githubusercontent.com/PyPSA/technology-data/v0.5.0/outputs/
        baseurl = f"https://raw.githubusercontent.com/PyPSA/technology-data/{version}/outputs/"

    filepath = Path(output_path)
    url = baseurl + filepath.name

    logger.info(f"Downloading technology data from '{url}'.")
    progress_retrieve(url, str(filepath), disable=disable_progress)
    logger.info(f"Technology data available at {filepath}")


def load_tech_costs(path: str) -> pd.Series:
    """Load technology costs CSV into MultiIndex Series [technology, parameter]."""
    df = pd.read_csv(path, index_col=[0, 1])
    return df.iloc[:, 0]  # Return first column as Series with MultiIndex


def get_tech(tech_costs: pd.Series, tech_name: str) -> pd.Series:
    """Retrieve all parameters for a specific technology. Returns empty Series if not found."""
    try:
        return tech_costs.loc[tech_name]
    except KeyError:
        logger.warning(f"Technology '{tech_name}' not found in database")
        return pd.Series()


def get_tech_param(
    tech_params: pd.Series, param_name: str, default: float | None = None
) -> float:
    """Extract technology parameter with optional fallback default."""
    try:
        return tech_params.loc[param_name]
    except KeyError:
        if default is not None:
            logger.warning(
                f"Parameter '{param_name}' not found, using default: {default}"
            )
            return default
        raise


# Snakemake integration: Allow direct execution as rule
if __name__ == "__main__":
    if snakemake is None:
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_cost_data", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    # Download technology data using Snakemake parameters
    version = snakemake.params.version
    output_path = Path(rootpath) / snakemake.output[0]
    disable_progress = snakemake.config["run"].get("disable_progressbar", False)

    download_tech_database(version, str(output_path), disable_progress=disable_progress)
