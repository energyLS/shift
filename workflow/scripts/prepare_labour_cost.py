"""
Labour cost calculator for H-DRI-EAF value chains.

Reads the merged labour inputs produced by data_downloader.py, computes
an all-in hourly steelworker wage [EUR/h, 2020 prices] for each country,
maps countries to the model regions defined in config/config.yaml, and
saves a regional aggregated CSV.

Methodology (Nykvist et al. 2025, Section D4)
----------------------------------------------
1. Steel wage ratio (dimensionless):
       wage_ratio = (wage_per_employee_USD * (1 + employer_contrib_rate))
                    / GNI_data_year_USD
   Both numerator and denominator use the same year and the same USD
   denomination, so currency conversion cancels out.

2. Hourly wage [EUR/h, 2020]:
       hourly_wage = (GNI_2020_EUR * wage_ratio / WORKING_YEAR) * (1 + OVERHEAD)
   where  GNI_2020_EUR = GNI_2020_USD * EUR_per_USD_2020.

3. Countries with no UNIDO data have their wage ratio imputed as the
   employment-weighted mean of all countries with data.

4. Regional aggregation: employment-weighted mean of country hourly wages
   within each model region.  Regions with no data fall back to the
   global employment-weighted mean.

Output:  resources/labour_cost_clustered.csv
    region | labour_ely | labour_dri | labour_eaf   [EUR/h, 2020]
    (all three columns hold the same hourly rate; the model applies
     different labour-intensity factors per technology downstream)

Usage
-----
    python workflow/scripts/labour_cost_calculator.py
"""

import warnings
import pycountry
import pandas as pd
from pathlib import Path
from typing import Any

from _helpers import setup_logging

snakemake: Any = globals().get("snakemake")
logger = setup_logging(
    __name__, snakemake=snakemake, log_filename="prepare_labour_cost.log"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKING_YEAR = 2080  # assumed working hours per year
OVERHEAD = 0.25  # 25 % overhead on direct labour cost
TARGET_YEAR = 2020  # reference year for all monetary outputs

# Labour intensities (Nykvist Table D5, original data Devlin)
LI_ELY_PU = 2.0  # h / kW installed electrolyser
LI_DRI_PU = 0.18  # h / t DRI
LI_EAF_PU = 0.49  # h / t steel

# ---------------------------------------------------------------------------
# Country name → ISO3 helpers for config region mapping
# ---------------------------------------------------------------------------
_EXTRA_NAME_TO_ISO3 = {
    # Config-specific names that pycountry.lookup() doesn't find by default
    "Iran, Islamic Republic of": "IRN",
    "Republic of the Congo": "COG",
    "Hong Kong": "HKG",
    "Taiwan": "TWN",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Vietnam": "VNM",
    "Bolivia": "BOL",
    "Venezuela": "VEN",
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "Laos": "LAO",
    "Brunei": "BRN",
    "Palestine": "PSE",
    "Syria": "SYR",
    "Democratic Republic of the Congo": "COD",
    "Equatorial French Guiana": None,  # not a sovereign country
}


def _config_name_to_iso3(name: str):
    """Convert a config region country name to ISO3, return None if not found."""
    name = name.strip()
    if name in _EXTRA_NAME_TO_ISO3:
        return _EXTRA_NAME_TO_ISO3[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def build_iso_to_region(config_regions: dict) -> dict:
    """
    Build {iso3: region_name} from the config 'regions' dict.

    Config entries may use '+' to combine two countries in one string
    (e.g. "Togo + Algeria"); each part is split and mapped individually.
    """
    iso_to_region = {}
    unmatched = []
    for region, countries in config_regions.items():
        for entry in countries:
            for part in str(entry).split("+"):
                iso3 = _config_name_to_iso3(part.strip())
                if iso3:
                    iso_to_region[iso3] = region
                else:
                    unmatched.append((region, part.strip()))
    if unmatched:
        warnings.warn(
            f"Could not map {len(unmatched)} config country name(s) to ISO3: "
            + ", ".join(f"{r}/{n}" for r, n in unmatched[:10])
            + (" …" if len(unmatched) > 10 else "")
        )
    return iso_to_region


# ---------------------------------------------------------------------------
# Wage computation
# ---------------------------------------------------------------------------
def compute_hourly_wage(row) -> float:
    """
    Compute the all-in hourly steelworker wage [EUR/h] in TARGET_YEAR prices.

    wage_ratio = (steel_wage_usd / steel_employees) * (1 + employer_contrib_rate)
                 / gni_data_year_usd

    hourly_wage = (gni_target_year_usd * eur_per_usd_target
                   * wage_ratio / WORKING_YEAR) * (1 + OVERHEAD)
    """
    wage_ratio = (
        (row["steel_wage_usd"] / row["steel_employees"])
        * (1.0 + row["employer_contrib_rate"])
        / row["gni_data_year_usd"]
    )
    gni_target_eur = row["gni_target_year_usd"] * row["eur_per_usd_target"]
    return (gni_target_eur * wage_ratio / WORKING_YEAR) * (1.0 + OVERHEAD)


def compute_all_hourly_wages(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns [iso3, country_name, steel_employees,
    hourly_wage_eur] for every country in *merged_df*.

    Countries missing any required column get their wage imputed as the
    employment-weighted mean of all countries with complete data.
    """
    required = [
        "steel_wage_usd",
        "steel_employees",
        "gni_data_year_usd",
        "gni_target_year_usd",
        "employer_contrib_rate",
        "eur_per_usd_target",
    ]
    valid_mask = merged_df[required].notna().all(axis=1)  # type: ignore[arg-type]
    valid = merged_df[valid_mask].copy()
    valid["hourly_wage_eur"] = valid.apply(compute_hourly_wage, axis=1)

    # Employment-weighted mean for imputation
    total_emp = valid["steel_employees"].sum()
    mean_wage = (
        (valid["hourly_wage_eur"] * valid["steel_employees"]).sum() / total_emp
        if total_emp > 0
        else valid["hourly_wage_eur"].mean()
    )

    result = merged_df[["iso3", "country_name", "steel_employees"]].merge(
        valid[["iso3", "hourly_wage_eur"]], on="iso3", how="left"
    )
    n_imputed = result["hourly_wage_eur"].isna().sum()
    if n_imputed:
        warnings.warn(
            f"{n_imputed} countries have incomplete data and will be imputed "
            f"with the global mean ({mean_wage:.2f} EUR/h)."
        )
    result["hourly_wage_eur"] = result["hourly_wage_eur"].fillna(mean_wage)
    return result


# ---------------------------------------------------------------------------
# Regional aggregation
# ---------------------------------------------------------------------------
def aggregate_by_region(
    wages_df: pd.DataFrame,
    iso_to_region: dict,
    regions: list,
) -> pd.DataFrame:
    """
    Compute the employment-weighted mean hourly wage for each model region.

    Countries with no employment figure use weight = 1.
    Regions with no matching countries use the global weighted mean.
    """
    df = wages_df.copy()
    df["region"] = df["iso3"].map(iso_to_region)
    df["weight"] = df["steel_employees"].fillna(1.0)

    # Global fallback
    global_mean = (df["hourly_wage_eur"] * df["weight"]).sum() / df["weight"].sum()

    rows = []
    for region in regions:
        grp = df[df["region"] == region]
        if grp.empty:
            wage = global_mean
        else:
            w_sum = grp["weight"].sum()
            wage = (
                (grp["hourly_wage_eur"] * grp["weight"]).sum() / w_sum
                if w_sum > 0
                else global_mean
            )
        rows.append(
            {
                "region": region,
                "steelworker_wage in euro/h": round(wage, 4),
                "ely_intensity in h/kW_ely": LI_ELY_PU,
                "dri_intensity in h/t_dri": LI_DRI_PU,
                "eaf_intensity in h/t_steel": LI_EAF_PU,
            }
        )

    return pd.DataFrame(rows).set_index("region")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("prepare_labour_cost")

    # ── 1. Load merged data ──────────────────────────────────────────────
    merged_path = Path(snakemake.input.merged)
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged labour inputs not found: {merged_path}\n"
            "  Run download_labour_data first."
        )
    logger.info(f"Loading {merged_path} …")
    merged = pd.read_csv(merged_path)
    logger.info(f"  {len(merged)} countries loaded.")

    # ── 2. Load config regions ───────────────────────────────────────────
    regions_config: dict = snakemake.config["regions"]
    iso_to_region = build_iso_to_region(regions_config)
    regions = list(regions_config.keys())
    logger.info(f"  {len(regions)} model regions from config.")

    # ── 3. Compute hourly wages ──────────────────────────────────────────
    wages = compute_all_hourly_wages(merged)
    logger.info("\nCountry-level hourly wages [EUR/h, 2020]:")
    wage_table = (
        wages[["iso3", "country_name", "hourly_wage_eur"]]
        .sort_values("hourly_wage_eur", ascending=False)  # type: ignore[call-overload]
        .to_string(index=False)
    )
    logger.info(wage_table)

    # ── 4. Aggregate by region ───────────────────────────────────────────
    result = aggregate_by_region(wages, iso_to_region, regions)

    # ── 5. Save ──────────────────────────────────────────────────────────
    output_path = Path(snakemake.output.labour_cost)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    logger.info(f"\nSaved → {output_path}")
    logger.info(result.to_string())
