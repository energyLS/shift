"""
data_downloader.py
==================
Downloads and prepares all input data needed for labour_cost_calculator.py
to run on any country worldwide.

Data sources
------------
A. World Bank WDI  – GNI per capita (Atlas method, current USD)
     Automated via the `wbgapi` package.  Requires: pip install wbgapi

B. ECB             – Annual average USD/EUR spot rates
     Automated via the ECB public REST API (no credentials needed).

C. UNIDO INDSTAT   – Steel sector wage bill & employment
     Raw data files are expected in  data/labour/unido-raw/data.csv
     (downloaded from https://stat.unido.org/data/download,
      INDSTAT Rev 4, ISIC 241, variables 04+05, all countries).
     Licence: CC BY 4.0

D. Employer SSC rates – statutory employer social-security contribution rates
     Pre-compiled reference table for ~80 countries.
     Sources: OECD Taxing Wages 2023/24, ILO Social Security Inquiry 2022,
              KPMG Global Employer Tax Guide 2023.

Outputs (saved to data/labour/)
--------------------------------
  data/labour/gni_per_capita.csv
  data/labour/ecb_usd_eur.csv
  data/labour/employer_contributions.csv
  data/labour/merged_labour_inputs.csv    ← main output for labour_cost_calculator.py

Usage
-----
  python workflow/scripts/data_downloader.py [--target-year 2020] [--force]
"""

import sys
import warnings
import requests  # type: ignore
import pandas as pd  # type: ignore
import pycountry  # type: ignore
from io import StringIO
from pathlib import Path

from _helpers import setup_logging

# ── optional wbgapi ──────────────────────────────────────────────────────────
try:
    import wbgapi as wb  # type: ignore

    HAS_WBGAPI = True
except ImportError:
    HAS_WBGAPI = False

# ─────────────────────────────────────────────────────────────────────────────
# File paths  (resolved relative to this script → works from any cwd)
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "labour"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UNIDO raw data: data/labour/unido-raw/data.csv
UNIDO_RAW = DATA_DIR / "unido-raw" / "data.csv"
GNI_CSV = DATA_DIR / "gni_per_capita.csv"
ECB_CSV = DATA_DIR / "ecb_usd_eur.csv"
CONTRIB_CSV = DATA_DIR / "employer_contributions.csv"
MERGED_CSV = DATA_DIR / "merged_labour_inputs.csv"

snakemake = globals().get("snakemake")
logger = setup_logging(
    __name__, snakemake=snakemake, log_filename="download_labour_data.log"
)

# ─────────────────────────────────────────────────────────────────────────────
# OECD member list (ISO-3, as of 2024)
# ─────────────────────────────────────────────────────────────────────────────
OECD_MEMBERS = {
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHL",
    "COL",
    "CRI",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "ISL",
    "IRL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LVA",
    "LTU",
    "LUX",
    "MEX",
    "NLD",
    "NZL",
    "NOR",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "CHE",
    "TUR",
    "GBR",
    "USA",
}


# ─────────────────────────────────────────────────────────────────────────────
# A. World Bank GNI per capita
# ─────────────────────────────────────────────────────────────────────────────
def fetch_world_bank_gni(start_year=2010, end_year=2023, force=False):
    """
    Download GNI per capita (Atlas method, current USD) for all World Bank
    economies covering *start_year*–*end_year*.

    Requires the `wbgapi` package:  pip install wbgapi

    Returns a tidy DataFrame:  iso3 | year | gni_usd
    Saves result to  data/gni_per_capita.csv.
    """
    if GNI_CSV.exists() and not force:
        logger.info(f"[GNI]  Loading cached → {GNI_CSV}")
        return pd.read_csv(GNI_CSV)

    if not HAS_WBGAPI:
        sys.exit(
            "ERROR: wbgapi is required to download World Bank GNI data.\n"
            "  pip install wbgapi"
        )

    logger.info("[GNI]  Downloading from World Bank (NY.GNP.PCAP.CD) …")
    try:
        raw = wb.data.DataFrame(
            "NY.GNP.PCAP.CD",
            time=range(start_year, end_year + 1),
            skipBlanks=True,
            columns="time",
        )
        # raw: index = economy (ISO3), columns = "YR2010" … "YR2023"
        df = (
            raw.reset_index()
            .rename(columns={"economy": "iso3"})
            .melt(id_vars="iso3", var_name="year", value_name="gni_usd")
        )
        df["year"] = df["year"].str.replace("YR", "").astype(int)
        df = df.dropna(subset=["gni_usd"])
        df.to_csv(GNI_CSV, index=False)
        logger.info(f"[GNI]  {len(df)} rows → {GNI_CSV}")
        return df

    except Exception as exc:
        sys.exit(f"ERROR downloading World Bank GNI data: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# B. ECB USD/EUR annual average exchange rates
# ─────────────────────────────────────────────────────────────────────────────

# Fallback table: ECB EXR.A.USD.EUR.SP00.A annual averages (USD per 1 EUR).
# Source: European Central Bank Statistical Data Warehouse.
# Last updated: 2024.  Invert to obtain EUR per USD.
_ECB_FALLBACK_USD_PER_EUR = {
    2000: 0.9236,
    2001: 0.8956,
    2002: 0.9454,
    2003: 1.1312,
    2004: 1.2438,
    2005: 1.2441,
    2006: 1.2556,
    2007: 1.3705,
    2008: 1.4726,
    2009: 1.3948,
    2010: 1.3257,
    2011: 1.3920,
    2012: 1.2848,
    2013: 1.3281,
    2014: 1.3285,
    2015: 1.0859,
    2016: 1.1069,
    2017: 1.1297,
    2018: 1.1810,
    2019: 1.1195,
    2020: 1.1422,
    2021: 1.1827,
    2022: 1.0530,
    2023: 1.0813,
    2024: 1.0815,
}


def _parse_ecb_csv(text):
    """Parse an ECB SDMX-CSV response into a tidy DataFrame."""
    raw = pd.read_csv(StringIO(text))
    time_col = next(c for c in raw.columns if "TIME" in c.upper())
    val_col = next(c for c in raw.columns if "OBS_VALUE" in c.upper())
    df = raw[[time_col, val_col]].copy()
    df.columns = ["year", "usd_per_eur"]
    df["year"] = df["year"].astype(int)
    df["eur_per_usd"] = 1.0 / df["usd_per_eur"]
    return df


def fetch_ecb_rates(start_year=2010, end_year=2023, force=False):
    """
    Obtain annual average USD/EUR spot rates from the ECB.

    Series: EXR.A.USD.EUR.SP00.A  →  USD per 1 EUR.
    Inverted to EUR per USD (= EUR_per_USD used in the paper).

    Strategy:
      1. Load cached CSV if present (and not --force).
      2. Try ECB SDW-WSREST API (old, well-documented endpoint).
      3. Try ECB Data Portal API v1 (new endpoint).
      4. Fall back to the embedded _ECB_FALLBACK_USD_PER_EUR table.

    Returns:  year | usd_per_eur | eur_per_usd
    Saves to  data/ecb_usd_eur.csv.
    """
    if ECB_CSV.exists() and not force:
        logger.info(f"[ECB]  Loading cached → {ECB_CSV}")
        return pd.read_csv(ECB_CSV)

    ecb_attempts = [
        (
            "ECB SDW-WSREST",
            (
                "https://sdw-wsrest.ecb.europa.eu/service/data/EXR/A.USD.EUR.SP00.A"
                f"?format=csvdata&startPeriod={start_year}&endPeriod={end_year}"
            ),
            {},
        ),
        (
            "ECB Data Portal v1 (Accept: text/csv)",
            (
                "https://data.ecb.europa.eu/api/v1/data/EXR/A.USD.EUR.SP00.A"
                f"?startPeriod={start_year}&endPeriod={end_year}"
            ),
            {"Accept": "text/csv"},
        ),
        (
            "ECB Data Portal v1 (format=csvdata)",
            (
                "https://data.ecb.europa.eu/api/v1/data/EXR/A.USD.EUR.SP00.A"
                f"?format=csvdata&startPeriod={start_year}&endPeriod={end_year}"
            ),
            {},
        ),
    ]

    for label, url, headers in ecb_attempts:
        try:
            logger.info(f"[ECB]  Trying {label} …")
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            df = _parse_ecb_csv(resp.text)
            df.to_csv(ECB_CSV, index=False)
            logger.info(f"[ECB]  {len(df)} rows → {ECB_CSV}")
            return df
        except Exception as exc:
            logger.warning(f"[ECB]  {label} failed: {exc}")

    # ── fallback: embedded reference table ───────────────────────────────────
    logger.info(
        "[ECB]  All live endpoints unavailable.  Using embedded reference table (_ECB_FALLBACK_USD_PER_EUR)."
    )
    rows = [
        {"year": y, "usd_per_eur": r, "eur_per_usd": 1.0 / r}
        for y, r in _ECB_FALLBACK_USD_PER_EUR.items()
        if start_year <= y <= end_year
    ]
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(ECB_CSV, index=False)
    logger.info(f"[ECB]  {len(df)} rows (from fallback table) → {ECB_CSV}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# C. UNIDO INDSTAT – steel sector wages & employment
# ─────────────────────────────────────────────────────────────────────────────

# Common UNIDO country-name variants → ISO-3 override
# (pycountry handles most; these cover known quirks)
_UNIDO_NAME_OVERRIDES = {
    "United States of America": "USA",
    "United States": "USA",
    "Korea, Republic of": "KOR",
    "Republic of Korea": "KOR",
    "Korea (the Republic of)": "KOR",
    "Taiwan, Province of China": "TWN",
    "China, Taiwan Province": "TWN",
    "Taiwan": "TWN",
    "Iran (Islamic Republic of)": "IRN",
    "Iran, Islamic Republic of": "IRN",
    "Viet Nam": "VNM",
    "Vietnam": "VNM",
    "Bolivia (Plurinational State of)": "BOL",
    "Bolivia": "BOL",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Venezuela": "VEN",
    "Congo, Democratic Republic of the": "COD",
    "Democratic Republic of the Congo": "COD",
    "Syrian Arab Republic": "SYR",
    "Syria": "SYR",
    "Lao People's Democratic Republic": "LAO",
    "Laos": "LAO",
    "Moldova, Republic of": "MDA",
    "Republic of Moldova": "MDA",
    "Tanzania, United Republic of": "TZA",
    "Tanzania": "TZA",
    "Slovak Republic": "SVK",
    "Czechia": "CZE",
    "Czech Republic": "CZE",
    "Russian Federation": "RUS",
    "Russia": "RUS",
    "North Macedonia": "MKD",
    "Macedonia": "MKD",
    "United Kingdom": "GBR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
}


def _name_to_iso3(name):
    """Convert a country name string to ISO-3 alpha code, or return None."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name in _UNIDO_NAME_OVERRIDES:
        return _UNIDO_NAME_OVERRIDES[name]
    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except LookupError:
        return None


def load_unido_data(filepath=UNIDO_RAW):
    """
    Load and normalise the UNIDO INDSTAT CSV downloaded from the portal.

    Handles two common export layouts:
    • Long format: columns include Country/Year/Variable/Value
    • SDMX-CSV:    columns include REF_AREA / TIME_PERIOD / INDICATOR / OBS_VALUE

    Returns a tidy DataFrame:  iso3 | country_name | year | employees | wages_usd
    Returns *None* if the file does not exist (prints download instructions).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"[UNIDO] Data file not found: {filepath}")
        logger.warning(
            "  Please download INDSTAT Rev 4 (ISIC 241, variables 04+05, all countries)\n"
            "  from https://stat.unido.org/data/download and extract data.csv into\n"
            f"  {filepath.parent}/",
        )
        return None

    logger.info(f"[UNIDO] Reading {filepath} …")
    raw = pd.read_csv(filepath, low_memory=False)

    # ── Employees: VariableCode 4, count in Value ─────────────────────────
    logger.info(f"[UNIDO] {len(raw)} raw rows read from {filepath}")
    emp_mask = raw["VariableCode"].astype(str).str.strip().isin(["4", "04"])
    emp = (
        raw[emp_mask][["Year", "Country", "Value"]]
        .copy()
        .rename(
            columns={"Year": "year", "Country": "country_name", "Value": "employees"}
        )
    )
    emp["employees"] = pd.to_numeric(emp["employees"], errors="coerce")
    # Sum across activity combinations (e.g. 2410A + 2410B) for same country/year
    emp = emp.groupby(["year", "country_name"], as_index=False)["employees"].sum()

    # ── Wages: VariableCode 5, USD amount in ValueUSD ─────────────────────
    wage_mask = raw["VariableCode"].astype(str).str.strip().isin(["5", "05"])
    wages = (
        raw[wage_mask][["Year", "Country", "ValueUSD"]]
        .copy()
        .rename(
            columns={"Year": "year", "Country": "country_name", "ValueUSD": "wages_usd"}
        )
    )
    wages["wages_usd"] = pd.to_numeric(wages["wages_usd"], errors="coerce")
    wages = wages.groupby(["year", "country_name"], as_index=False)["wages_usd"].sum()

    # ── Merge and convert country names to ISO3 ───────────────────────────
    df = pd.merge(emp, wages, on=["year", "country_name"], how="inner")
    df = df.dropna(subset=["employees", "wages_usd"])

    df["iso3"] = df["country_name"].apply(_name_to_iso3)
    df = df.dropna(subset=["iso3"])
    df["iso3"] = df["iso3"].str.upper().str.strip()

    result = df[["iso3", "country_name", "year", "employees", "wages_usd"]].copy()
    logger.info(
        f"[UNIDO] {len(result)} country-year rows loaded ({result['iso3'].nunique()} countries)."
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D. Employer social-security contribution rates
# Sources: OECD Taxing Wages 2024, ILO Social Security Inquiry 2022,
#          KPMG Global Employer Tax Guide 2023.
# All rates are the statutory employer SSC as a share of gross wages.
# ─────────────────────────────────────────────────────────────────────────────
EMPLOYER_CONTRIB_TABLE = {
    # ISO3 : (rate_fraction, source_note)
    "ARG": (0.170, "ANSES employer contributions ~17%"),
    "AUS": (0.195, "Superannuation 11%, healthcare 2%, payroll levy ~5.5%"),
    "AUT": (0.228, "Pension 12.55%, accident 1.2%, housing 0.5%, others ~8.6%"),
    "BEL": (0.270, "ONSS employer total ~27%"),
    "BGD": (0.050, "Bangladesh: employer contribution estimate ~5%"),
    "BGR": (0.185, "Pension 10.82%, health 4.8%, others ~2.9%"),
    "BLR": (0.340, "Social protection fund 34%"),
    "BRA": (0.305, "INSS 22.5%, FGTS 8%"),
    "CAN": (0.077, "CPP 5.95%, EI 1.66%"),
    "CHL": (0.024, "Social security employer portion 2.4%"),
    "CHN": (
        0.282,
        "Pension 16%, medical 5.8%, unemployment 0.5%, injury 0.4%, maternity 0.8%",
    ),
    "COL": (0.275, "Pension 12%, health 8.5%, ARL ~2%, SENA 2%, ICBF 3%"),
    "CRI": (0.264, "CCSS employer contributions 26.4%"),
    "CZE": (0.248, "Pension 21.5%, health 3.3%"),
    "DEU": (
        0.212,
        "Pension 9.3%, unemployment 1.23%, health 7.3%, nursing+other ~3.3%",
    ),
    "DNK": (0.010, "Denmark: minimal statutory employer SSC"),
    "DZA": (0.260, "CNAS employer 26%"),
    "EGY": (0.260, "Social insurance employer 26%"),
    "ESP": (
        0.310,
        "Pension 23.6%, unemployment 5.5%, FOGASA 0.2%, FP 0.6%, other 1.1%",
    ),
    "EST": (0.330, "Social tax 33%"),
    "ETH": (0.110, "Private pension 11%"),
    "FIN": (0.200, "Pension ~17.39%, unemployment ~1.91%, accident/other"),
    "FRA": (0.425, "Complex French system total ~42.5%"),
    "GBR": (0.138, "NIC Class 1 employer 13.8%"),
    "GHA": (0.130, "SSNIT 13%"),
    "GRC": (0.251, "IKA-ETAM ~25.06%"),
    "HUN": (0.130, "Social contribution tax 13%"),
    "IDN": (0.092, "BPJS: pension 3.7%, healthcare 4%, accident 0.24%, death 0.3%"),
    "IND": (0.136, "EPF 12%, ESI 3.25%, LWF negligible"),
    "IRL": (0.115, "PRSI Class A employer 11.15%"),
    "IRN": (0.230, "Social security 23%"),
    "IRQ": (0.120, "Social security employer 12%"),
    "ISL": (0.065, "Iceland: employer social security 6.5%"),
    "ISR": (0.075, "National insurance 3.55%, health insurance 3.1%"),
    "ITA": (0.320, "INPS employer ~32%"),
    "JPN": (0.151, "Pension 9.15%, health 4.99%, employment 0.6%, WC ~0.5%"),
    "KAZ": (0.180, "Social contributions 3.5%, UAPF 3.5%, health 3%, other ~7%"),
    "KOR": (0.105, "National pension 4.5%, health 3.545%, employment 0.9%, WC ~1.5%"),
    "LBY": (0.115, "Social security 11.5%"),
    "LTU": (0.017, "Employer SSC 1.77% (post-2019 reform)"),
    "LUX": (0.155, "Pension 8%, health 3.05%, accident 1.1%, mutual aid/LTC ~3.3%"),
    "LVA": (0.237, "Social insurance 23.59%"),
    "MAR": (
        0.228,
        "CNSS: pension 11.89%, family 6.4%, AMO 2.26%, training 1.6%, accident",
    ),
    "MEX": (0.250, "IMSS ~17%, INFONAVIT 5%, SAR 2%"),
    "MYS": (0.143, "EPF 12%, SOCSO 1.75%, EIS 0.4%, HRDF 0.5%"),
    "NGA": (0.100, "PRA employer contributory pension 10%"),
    "NLD": (0.190, "AOW/ANW/WLZ/WIA/ZVW combined ~19%"),
    "NOR": (0.141, "Employer social security 14.1%"),
    "NZL": (0.000, "No statutory employer SSC"),
    "OMN": (0.113, "PASI employer 11.25%"),
    "PAK": (0.120, "EOBI 5%, ESSI 5%, other ~2%"),
    "PER": (0.090, "EsSalud 9%"),
    "PHL": (0.115, "SSS 8%, PhilHealth 2.5%, Pag-IBIG 2%"),
    "POL": (
        0.204,
        "Pension 9.76%, disability 6.5%, accident avg 1.67%, FP+FGSP ~2.55%",
    ),
    "PRT": (0.238, "Social security 23.75%"),
    "QAT": (0.000, "Qatar: no employer SSC for most private-sector workers"),
    "ROU": (0.023, "Work accident/occupational disease 2.25%"),
    "RUS": (0.302, "Pension 22%, medical 5.1%, social 2.9%, injury 0.2%"),
    "SAU": (0.118, "Social insurance 11.75%"),
    "SGP": (0.170, "CPF employer ~17%"),
    "SVK": (
        0.248,
        "Pension 14%, disability 3%, sickness 1.4%, unemployment 1%, other ~5%",
    ),
    "SVN": (0.163, "Pension 8.85%, health 6.56%, accident 0.53%, employment 0.06%"),
    "SWE": (0.314, "Social security fees 31.42%"),
    "THA": (0.050, "Social security employer 5%"),
    "TUN": (0.165, "CNSS employer 16.57%"),
    "TUR": (0.225, "Social security premiums 22.5%"),
    "TZA": (0.100, "NSSF employer 10%"),
    "UKR": (0.220, "Unified social contribution 22%"),
    "ARE": (0.125, "GPSSA employer 12.5% for nationals; expats typically 0%"),
    "USA": (0.076, "FICA: SS 6.2%, Medicare 1.45%, FUTA ~0.6%"),
    "UZB": (0.120, "Social insurance 12%"),
    "VNM": (0.215, "Social insurance 17.5%, health 3%, unemployment 1%"),
    "ZAF": (0.020, "Skills development levy 1%, UIF 1%"),
    "ZMB": (0.050, "NAPSA employer 5%"),
    "ZWE": (0.045, "NSSA employer 4.5%"),
    "CHE": (0.135, "AHV/IV/EO 5.3%, ALV 1.1%, FAK ~1.7%, accident/pension ~5.4%"),
    "MKD": (0.070, "North Macedonia: employer health 7.3% → total ~7%"),
    "SRB": (
        0.173,
        "Serbia: pension 11%, health 5.15%, unemployment 0.75%, other ~0.3%",
    ),
    "HRV": (0.165, "Croatia: pension pillar II 5%, health ~16.5% total"),
    "BIH": (0.105, "Bosnia: contributions ~10.5% (varies by entity)"),
    "ALB": (0.150, "Albania: employer social insurance 15%"),
    "GEO": (0.000, "Georgia: no employer-side social contribution"),
    "ARM": (0.025, "Armenia: employer social premium 2.5%"),
    "AZE": (0.220, "Azerbaijan: employer social insurance 22%"),
    "TKM": (0.200, "Turkmenistan: employer insurance ~20%"),
    "KGZ": (0.175, "Kyrgyzstan: employer social fund 17.5%"),
    "TJK": (0.250, "Tajikistan: employer contribution ~25%"),
    "MNG": (0.135, "Mongolia: employer social insurance 13.5%"),
    "MMR": (0.030, "Myanmar: SSB employer 2.5-3%"),
    "KHM": (0.031, "Cambodia: NSSF employer 3.1%"),
    "LKA": (0.120, "Sri Lanka: EPF 12%"),
    "NPL": (0.100, "Nepal: SSF employer ~10%"),
    "KEN": (0.060, "Kenya: NSSF + NHIF employer ~6%"),
    "UGA": (0.100, "Uganda: NSSF employer 10%"),
    "SEN": (0.040, "Senegal: IPM employer + family benefit ~4%"),
    "CIV": (0.065, "Côte d'Ivoire: CNPS employer ~6.5%"),
    "CMR": (0.080, "Cameroon: CNPS employer ~8%"),
    "AGO": (0.080, "Angola: employer social security 8%"),
    "MOZ": (0.040, "Mozambique: INSS employer 4%"),
    "BWA": (0.000, "Botswana: no statutory employer SSC"),
    "NAM": (0.000, "Namibia: no statutory employer SSC"),
    "MUS": (0.060, "Mauritius: NPF 6%"),
    "TTO": (0.060, "Trinidad & Tobago: NIS employer 5.85% + 0.3%"),
    "JAM": (0.030, "Jamaica: NIS employer 2.5-3%"),
    "ECU": (0.120, "Ecuador: IESS employer 12.15%"),
    "BOL": (0.165, "Bolivia: AFP 10%, health 10%, employer total ~16.5%"),
    "PRY": (0.165, "Paraguay: IPS employer 14%, others ~2.5%"),
    "URY": (0.075, "Uruguay: BPS employer 7.5%"),
    "PAN": (0.125, "Panama: CSS employer 12.25%"),
    "GTM": (0.105, "Guatemala: IGSS employer 10.5%"),
    "HND": (0.090, "Honduras: IHSS employer 5%, RAP 5% → ~9%"),
    "SLV": (0.075, "El Salvador: ISSS 7.5% employer"),
    "NIC": (0.190, "Nicaragua: INSS employer 19%"),
    "DOM": (0.075, "Dominican Republic: AFP 7.1% → total ~7.5%"),
    "CUB": (0.145, "Cuba: employer social security 14.5%"),
}


def build_employer_contributions():
    """
    Build a DataFrame from the embedded EMPLOYER_CONTRIB_TABLE.

    Returns:  iso3 | employer_contrib_rate | contrib_source | oecd_status
    Saves to  data/employer_contributions.csv.
    """
    rows = [
        {"iso3": iso, "employer_contrib_rate": rate, "contrib_source": note}
        for iso, (rate, note) in EMPLOYER_CONTRIB_TABLE.items()
    ]
    df = pd.DataFrame(rows)
    df["oecd_status"] = df["iso3"].apply(
        lambda x: "OECD" if x in OECD_MEMBERS else "non_OECD"
    )
    df.to_csv(CONTRIB_CSV, index=False)
    logger.info(f"[CONTRIB] {len(df)} countries → {CONTRIB_CSV}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# E. Merge all sources → merged_labour_inputs.csv
# ─────────────────────────────────────────────────────────────────────────────


def _pick_latest_row(group, target_year):
    """
    From a single country's time-series, return the row with the most recent
    year ≤ target_year that has both employees and wages_usd.
    Returns None if no qualifying row exists.
    """
    valid = group[group["year"] <= target_year].dropna(
        subset=["employees", "wages_usd"]
    )
    if valid.empty:
        return None
    return valid.sort_values("year").iloc[-1]


def _nearest_year(lookup_dict, iso3, preferred_year):
    """
    From a {(iso3, year): value} dict, return (year_used, value) for the
    entry closest to preferred_year.  Returns (None, None) if no data.
    """
    candidates = [(y, v) for (c, y), v in lookup_dict.items() if c == iso3]
    if not candidates:
        return None, None
    candidates.sort(key=lambda xy: abs(xy[0] - preferred_year))
    return candidates[0]


def merge_all_data(
    target_year=2020,
    unido_filepath=None,
    output_filepath=None,
    force_download=False,
):
    """
    Orchestrate all downloads, merge on country × year, and write the master
    input CSV for labour_cost_calculator.py.

    Parameters
    ----------
    target_year : int
        Reference year for GNI and EUR conversion (default 2020).
    unido_filepath : str or Path
        Path to the manually downloaded UNIDO CSV.
    output_filepath : str or Path
        Path for the output merged CSV (default: MERGED_CSV).
    force_download : bool
        If True, re-download GNI and ECB data even if cached files exist.

    Returns
    -------
    pd.DataFrame or None
        Merged dataset (None if UNIDO file is missing).
    """
    if unido_filepath is None:
        unido_filepath = UNIDO_RAW
    if output_filepath is None:
        output_filepath = MERGED_CSV
    output_filepath = Path(output_filepath)

    # 1. Fetch each source
    gni_df = fetch_world_bank_gni(force=force_download)
    ecb_df = fetch_ecb_rates(force=force_download)
    unido_df = load_unido_data(unido_filepath)
    contrib_df = build_employer_contributions()

    if unido_df is None:
        logger.warning("[MERGE] Cannot merge – UNIDO file not yet downloaded.")
        logger.warning(
            "        GNI, ECB and employer-contribution files have been saved."
        )
        return None

    # 2. Build fast lookup dicts
    # ECB: {year: eur_per_usd}
    ecb_dict = dict(zip(ecb_df["year"].astype(int), ecb_df["eur_per_usd"]))

    # GNI: {(iso3, year): gni_usd}
    gni_dict = {
        (r.iso3.strip().upper(), int(r.year)): r.gni_usd for r in gni_df.itertuples()
    }

    # 3. Pick best row per country from UNIDO
    records = []
    for iso3, grp in unido_df.groupby("iso3"):
        row = _pick_latest_row(grp, target_year)
        if row is None:
            continue

        data_year = int(row["year"])
        country_name = str(row.get("country_name", iso3))

        # ECB rate for data year
        eur_per_usd_data = ecb_dict.get(
            data_year, ecb_dict[min(ecb_dict, key=lambda y: abs(y - data_year))]
        )
        # ECB rate for target year
        eur_per_usd_target = ecb_dict.get(
            target_year, ecb_dict[min(ecb_dict, key=lambda y: abs(y - target_year))]
        )

        # GNI for data year
        gni_data_usd = gni_dict.get((iso3, data_year))
        if gni_data_usd is None:
            _, gni_data_usd = _nearest_year(gni_dict, iso3, data_year)
        if gni_data_usd is None:
            continue  # no GNI data → skip

        # GNI for target year
        gni_target_usd = gni_dict.get((iso3, target_year))
        if gni_target_usd is None:
            _, gni_target_usd = _nearest_year(gni_dict, iso3, target_year)
        if gni_target_usd is None:
            continue  # no GNI data → skip

        records.append(
            {
                "iso3": iso3,
                "country_name": country_name,
                "data_year": data_year,
                "oecd_status": "OECD" if iso3 in OECD_MEMBERS else "non_OECD",
                "steel_employees": row["employees"],
                "steel_wage_usd": row["wages_usd"],
                "gni_data_year_usd": gni_data_usd,
                "gni_target_year_usd": gni_target_usd,
                "eur_per_usd_data": eur_per_usd_data,
                "eur_per_usd_target": eur_per_usd_target,
            }
        )

    if not records:
        logger.warning("[MERGE] No valid records produced. Check UNIDO file contents.")
        return None

    merged = pd.DataFrame(records)

    # 4. Join employer contribution rates
    merged = merged.merge(
        contrib_df[["iso3", "employer_contrib_rate", "contrib_source"]],
        on="iso3",
        how="left",
    )

    # Report countries missing employer contribution data
    missing_mask = merged["employer_contrib_rate"].isna()
    if missing_mask.any():
        missing_iso = merged.loc[missing_mask, "iso3"].tolist()
        warnings.warn(
            f"{len(missing_iso)} countries have no employer contribution rate "
            f"in EMPLOYER_CONTRIB_TABLE: {missing_iso}. "
            "The caller should impute these (e.g. with the sample mean)."
        )

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_filepath, index=False)
    logger.info(f"\n[MERGE] {len(merged)} countries → {output_filepath}")
    cols_show = [
        "iso3",
        "country_name",
        "data_year",
        "steel_employees",
        "steel_wage_usd",
        "gni_target_year_usd",
        "employer_contrib_rate",
    ]
    logger.info(merged[cols_show].to_string(index=False))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("download_labour_data")

    merge_all_data(
        target_year=2020,
        unido_filepath=snakemake.input.unido_raw,
        output_filepath=snakemake.output.merged,
    )
