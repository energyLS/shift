"""
Cluster renewable generators using stratified k-medoids.

Methodology (based on Frysztacki et al. 2021, Siala & Mahfouz 2019):
1. Load merged renewable profiles (122k+ buses)
2. Filter to onwind + solar
3. Map buses to regions via ISO3
4. Stratify by cf_high_mass (top-5% hours mean) to preserve merit order & green pockets
5. Within each stratum: k-medoids on PCA(8760 profile) + lat/lon
6. Representative = medoid bus (real observed profile, no synthetic averaging)
7. Output clustered NetCDF: (region, technology, class, time)
8. Validate supply curve preservation

Usage (Snakemake rule):
    rule cluster_renewables:
        input:
            merged = "data/renewable_profiles_global_merged.nc",
        output:
            clustered = "resources/renewables_clustered.nc",
            report = "resources/renewables_clustering_report.json",
        script:
            "scripts/cluster_renewables.py"
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import kmedoids as _kmedoids
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

snakemake: Any = globals().get("snakemake")

CACHE_VERSION = "v5"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib progress callbacks into a tqdm bar."""

    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old


def extract_iso3(bus_id: str) -> Optional[str]:
    try:
        return str(bus_id).split("_")[0]
    except Exception:
        return None


def build_region_map(config: Dict) -> Dict[str, str]:
    """ISO3 -> region name lookup from config."""
    return {
        iso3: region
        for region, iso3_list in config.get("regions", {}).items()
        for iso3 in iso3_list
    }


def load_bus_coordinates(geojson_path: str) -> Dict[str, Tuple[float, float]]:
    """Load bus_id -> (lat, lon) from the merged geojson."""
    gdf = gpd.read_file(geojson_path)
    coords = {}
    for _, row in gdf.iterrows():
        bid = row["bus_id"]
        coords[bid] = (float(row["y_centroid"]), float(row["x_centroid"]))
    return coords


# ---------------------------------------------------------------------------
# Feature extraction — minimal: only what stratification + clustering needs
# ---------------------------------------------------------------------------


def compute_cf_high_mass(cf_ts: np.ndarray, tail_frac: float = 0.05) -> float:
    """Mean of the top `tail_frac` hours — captures peak resource quality."""
    cf = np.clip(np.nan_to_num(cf_ts, nan=0.0), 0.0, 1.0)
    if cf.size == 0:
        return 0.0
    k = max(1, int(len(cf) * tail_frac))
    return float(np.mean(np.sort(cf)[-k:]))


def extract_features_batch(
    bus_ids: List,
    ds: xr.Dataset,
    iso3_to_region: Dict[str, str],
    bus_coords: Dict[str, Tuple[float, float]],
) -> List[Dict]:
    """Extract lightweight features for a batch of buses.

    Only computes what the stratified workflow actually needs:
    - avg_cf, cf_high_mass (for stratification)
    - lat, lon, capacity_mw (for within-stratum clustering metadata)
    The raw 8760 profiles are read later during clustering, not stored here.
    """
    rows: List[Dict] = []
    for bus_id in bus_ids:
        iso3 = extract_iso3(str(bus_id))
        if not iso3:
            continue
        region = iso3_to_region.get(iso3)
        if not region:
            continue

        lat, lon = bus_coords.get(bus_id, (0.0, 0.0))

        for tech in ds.technology.values:
            tech = str(tech)
            try:
                cf_ts = ds["capacity_factor"].sel(bus=bus_id, technology=tech).values
            except Exception:
                continue
            if np.isnan(cf_ts).all() or len(cf_ts) == 0:
                continue

            cf_clean = np.clip(np.nan_to_num(cf_ts, nan=0.0), 0.0, 1.0)
            avg_cf = float(np.mean(cf_clean))

            try:
                capacity_mw = float(
                    ds["p_nom_max"].sel(bus=bus_id, technology=tech).values
                )
            except Exception:
                capacity_mw = 0.0
            if np.isnan(capacity_mw) or capacity_mw <= 0:
                continue

            rows.append(
                {
                    "bus_id": bus_id,
                    "technology": tech,
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "capacity_mw": capacity_mw,
                    "avg_cf": avg_cf,
                    "cf_high_mass": compute_cf_high_mass(cf_clean),
                }
            )
    return rows


def extract_features(
    ds: xr.Dataset,
    config: Dict,
    bus_coords: Dict[str, Tuple[float, float]],
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build the feature table (one row per bus-technology pair)."""

    required_cols = {
        "bus_id",
        "technology",
        "region",
        "lat",
        "lon",
        "capacity_mw",
        "avg_cf",
        "cf_high_mass",
    }

    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached features from {cache_path}")
        df = pd.read_csv(cache_path)
        if required_cols.issubset(df.columns):
            logger.info(f"  Loaded {len(df)} rows from cache")
            return df
        logger.info("  Cache schema stale — recomputing")

    iso3_to_region = build_region_map(config)
    bus_list = list(ds.bus.values)
    n_jobs = getattr(snakemake, "threads", -1)
    batch_size = max(1, len(bus_list) // max(1, abs(n_jobs)))
    batches = [
        bus_list[i : i + batch_size] for i in range(0, len(bus_list), batch_size)
    ]

    logger.info(
        f"Extracting features: {len(bus_list)} buses, {len(batches)} batches, {n_jobs} workers"
    )

    with tqdm_joblib(tqdm(total=len(batches), desc="Feature extraction", unit="batch")):
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(extract_features_batch)(
                batch, ds, iso3_to_region, bus_coords
            )
            for batch in batches
        )

    rows = [r for batch_rows in results for r in batch_rows]
    df = pd.DataFrame(rows)
    logger.info(f"  Extracted {len(df)} bus-technology pairs")

    if cache_path:
        df.to_csv(cache_path, index=False)
        logger.info(f"  Cached to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Stratification + within-stratum clustering
# ---------------------------------------------------------------------------


def _load_profiles_for_group(
    bus_ids: List[str], tech: str, ds: xr.Dataset
) -> np.ndarray:
    """Load 8760 CF profiles for a list of buses. Returns (n_buses, 8760)."""
    profiles = []
    for bid in bus_ids:
        try:
            cf = ds["capacity_factor"].sel(bus=bid, technology=tech).values
            cf = np.clip(np.nan_to_num(cf, nan=0.0), 0.0, 1.0)
            profiles.append(cf)
        except Exception:
            profiles.append(np.zeros(8760))
    return np.stack(profiles)


def _cluster_within_stratum(
    profiles: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    capacities: np.ndarray,
    n_clusters: int,
    n_pca: int = 5,
    geo_weight: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(profiles)
    if n <= n_clusters:
        return np.arange(n), np.arange(n)
    n_clusters = max(1, min(n_clusters, n))

    # Build feature matrix
    n_comp = min(n_pca, n - 1, profiles.shape[1])
    pca_features = PCA(n_components=max(1, n_comp)).fit_transform(
        StandardScaler().fit_transform(profiles)
    )
    pca_features /= np.sqrt(max(1, n_comp))

    geo = StandardScaler().fit_transform(np.column_stack([lats, lons]))
    geo /= np.sqrt(2)

    X = np.hstack([pca_features * (1 - geo_weight), geo * geo_weight])

    # Weight rows by capacity via duplication in distance matrix
    # (FasterPAM doesn't support sample_weight natively)
    cap_weights = np.sqrt(capacities / capacities.mean())  # soft weighting
    X_weighted = X * cap_weights[:, None]

    D = squareform(pdist(X_weighted, metric="euclidean"))
    result = _kmedoids.fasterpam(D, n_clusters, random_state=42)
    labels = np.array(result.labels)
    medoid_indices = np.array(result.medoids)

    return labels, medoid_indices


def _allocate_stratum_clusters(
    stratum_groups: Dict[str, pd.DataFrame],
    total_k: int,
    profiles_by_stratum: Dict[str, np.ndarray],
    tail_boost: float = 2.0,
    diversity_threshold: float = 0.05,
    min_capacity_for_split_mw: float = 500.0,
    top_n_strata_boost: int = 3,  # boost the top N strata by avg_cf
) -> Dict[str, int]:
    """Allocate clusters with boost for highest-CF strata (green pockets)."""

    # Rank strata by their avg_cf midpoint (encoded in name)
    strata_sorted = sorted(
        stratum_groups.keys(), reverse=True
    )  # lexicographic works for cf_XX_YY
    top_strata = set(strata_sorted[:top_n_strata_boost])

    splittable = {}
    fixed_at_one = {}

    for name, group in stratum_groups.items():
        profiles = profiles_by_stratum[name]
        total_cap = group["capacity_mw"].sum()
        n_buses = len(group)

        if n_buses <= 2:
            fixed_at_one[name] = 1
            continue

        bus_means = profiles.mean(axis=1)
        internal_diversity = float(np.std(bus_means))

        if (
            total_cap < min_capacity_for_split_mw
            or internal_diversity < diversity_threshold
        ):
            fixed_at_one[name] = 1
            continue

        splittable[name] = total_cap

    remaining_k = total_k - len(fixed_at_one) - len(splittable)
    remaining_k = max(0, remaining_k)

    alloc = dict(fixed_at_one)
    if splittable and remaining_k > 0:
        weights = {}
        for name, cap in splittable.items():
            w = cap
            if name in top_strata:
                w *= tail_boost
            weights[name] = w
        total_w = sum(weights.values())

        for name, w in weights.items():
            extra = int(round(w / total_w * remaining_k))
            alloc[name] = 1 + max(0, extra)
    else:
        for name in splittable:
            alloc[name] = 1

    return alloc


def _build_strata(
    df_rt: pd.DataFrame,
    bin_width: float = 0.05,
) -> pd.Series:
    """Assign strata by fixed avg_cf intervals of `bin_width`.

    E.g. bin_width=0.05 gives bins [0.00, 0.05), [0.05, 0.10), ..., [0.95, 1.00].
    Empty bins are implicitly ignored since no rows map to them.
    """
    cf_vals = df_rt["avg_cf"].values

    # Floor to nearest bin edge
    bin_idx = np.floor(cf_vals / bin_width).astype(int)

    labels = pd.Series(
        [
            f"cf_{int(b * bin_width * 100):02d}_{int((b + 1) * bin_width * 100):02d}"
            for b in bin_idx
        ],
        index=df_rt.index,
    )

    return labels


def resolve_total_clusters(df_rt: pd.DataFrame, tech: str) -> int:
    """Resolve total cluster count for a region-technology pair from config."""
    clustering_config = snakemake.config.get("clustering", {})
    policy = clustering_config.get("cluster_count_policy", {})
    base_clusters = policy.get("base_clusters", {})
    mode = policy.get("mode", "dynamic")
    base_key = "solar" if tech == "solar" else "onwind"
    base = int(base_clusters.get(base_key))

    if mode == "fixed":
        return base

    n_buses = max(1, df_rt["bus_id"].nunique())
    total_cap = float(df_rt["capacity_mw"].sum())
    ref_buses = float(policy.get("reference_buses", 1000.0))
    ref_cap = float(policy.get("reference_capacity_mw", 50000.0))
    exp = float(policy.get("scale_exponent", 0.5))
    bus_w = float(policy.get("bus_weight", 0.5))
    cap_w = float(policy.get("capacity_weight", 0.5))

    scale = (
        bus_w * (n_buses / ref_buses) ** exp
        + cap_w * (max(1.0, total_cap) / ref_cap) ** exp
    )
    target = int(round(base * scale))

    min_c = int(policy.get("min_clusters"))
    max_c = int(policy.get("max_clusters"))
    return max(min_c, min(max_c, min(target, len(df_rt))))


def cluster_region_technology(
    region: str,
    tech: str,
    df_features: pd.DataFrame,
    ds: xr.Dataset,
    clustering_config: Dict,
    cache_dir: Optional[str] = None,
) -> Tuple[str, str, pd.DataFrame]:
    """Stratified k-medoids clustering for one region-technology pair.

    Returns:
        (region, tech, result_df) where result_df has one row per cluster:
        bus_id (medoid), capacity_mw (summed), avg_cf, cf_high_mass, stratum.
    """
    tech = str(tech)

    # --- Cache check ---
    if cache_dir:
        cache_file = Path(cache_dir) / f"clusters_{region}_{tech}.parquet"
        if cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                if cached.attrs.get("cache_version", "") == CACHE_VERSION:
                    logger.info(f"  [CACHED] {region} {tech}: {len(cached)} clusters")
                    return region, tech, cached
            except Exception:
                pass

    df_rt = df_features[
        (df_features["region"] == region) & (df_features["technology"] == tech)
    ].copy()

    if len(df_rt) == 0:
        return region, tech, pd.DataFrame()

    total_k = resolve_total_clusters(df_rt, tech)
    tail_boost = float(clustering_config.get("tail_cluster_boost", 2.0))
    geo_weight = float(clustering_config.get("geo_weight", 0.3))
    n_pca = int(clustering_config.get("n_pca_components", 5))

    # --- Stratify ---
    bin_widths = clustering_config.get("strata_bin_width", {})
    bin_width = bin_widths.get(tech, 0.05)
    top_n_strata_boost = int(clustering_config.get("top_n_strata_boost", 3))
    df_rt["stratum"] = _build_strata(df_rt, bin_width=bin_width)

    n_actual_strata = df_rt["stratum"].nunique()
    logger.info(f"    {n_actual_strata} non-empty strata (bin width: {bin_width})")

    # Load profiles per stratum for diversity check
    stratum_groups = {}
    profiles_by_stratum = {}
    for stratum_name, group in df_rt.groupby("stratum"):
        stratum_groups[stratum_name] = group
        profiles_by_stratum[stratum_name] = _load_profiles_for_group(
            group["bus_id"].values.tolist(), tech, ds
        )

    min_cap_split = float(clustering_config.get("min_capacity_for_split_mw", 500.0))
    diversity_thresh = float(clustering_config.get("diversity_threshold", 0.05))

    stratum_k = _allocate_stratum_clusters(
        stratum_groups,
        total_k,
        profiles_by_stratum,
        tail_boost=tail_boost,
        diversity_threshold=diversity_thresh,
        min_capacity_for_split_mw=min_cap_split,
        top_n_strata_boost=top_n_strata_boost,
    )

    actual_total = sum(stratum_k.values())
    logger.info(
        f"    Cluster allocation: {stratum_k} (total: {actual_total}, "
        f"budget: {total_k}, {len(stratum_k) - sum(1 for v in stratum_k.values() if v > 1)} "
        f"strata kept at 1)"
    )

    # --- Cluster within each stratum ---
    cluster_rows: List[Dict] = []
    global_cluster_id = 0

    for stratum_name, group in df_rt.groupby("stratum"):
        k = min(stratum_k.get(stratum_name, 1), len(group))
        k = max(1, k)

        bus_ids = group["bus_id"].values.tolist()
        profiles = _load_profiles_for_group(bus_ids, tech, ds)
        lats = group["lat"].values
        lons = group["lon"].values
        caps = group["capacity_mw"].values

        labels, medoid_indices = _cluster_within_stratum(
            profiles, lats, lons, caps, n_clusters=k, n_pca=n_pca, geo_weight=geo_weight
        )

        for cl in range(int(labels.max()) + 1):
            mask = labels == cl
            if not mask.any():
                continue

            member_caps = caps[mask]
            member_bus_ids = np.array(bus_ids)[mask]
            total_cap = float(member_caps.sum())

            # Find medoid for this cluster
            medoid_candidates = np.where(mask)[0]
            medoid_local = None
            for mi in medoid_indices:
                if mi in medoid_candidates:
                    medoid_local = mi
                    break
            if medoid_local is None:
                # Fallback: largest capacity bus in cluster
                medoid_local = medoid_candidates[np.argmax(member_caps)]
                logger.warning(
                    f"No medoid in cluster {cl} of stratum {stratum_name}, picking largest bus: {bus_ids[medoid_local]}"
                )

            medoid_bus = bus_ids[medoid_local]

            # Capacity-weighted avg_cf for metadata
            member_avg_cfs = group.iloc[np.where(mask)[0]]["avg_cf"].values
            wavg_cf = float(np.average(member_avg_cfs, weights=member_caps))
            member_high_mass = group.iloc[np.where(mask)[0]]["cf_high_mass"].values
            wavg_high_mass = float(np.average(member_high_mass, weights=member_caps))

            cluster_rows.append(
                {
                    "cluster_id": global_cluster_id,
                    "region": region,
                    "technology": tech,
                    "stratum": stratum_name,
                    "medoid_bus_id": medoid_bus,
                    "n_buses": int(mask.sum()),
                    "capacity_mw": total_cap,
                    "avg_cf": wavg_cf,
                    "cf_high_mass": wavg_high_mass,
                    "member_bus_ids": ",".join(str(b) for b in member_bus_ids),
                }
            )
            global_cluster_id += 1

    result = pd.DataFrame(cluster_rows)

    # --- Cache ---
    if cache_dir and len(result) > 0:
        cache_file = Path(cache_dir) / f"clusters_{region}_{tech}.parquet"
        result.attrs["cache_version"] = CACHE_VERSION
        result.to_parquet(cache_file, index=False)
        logger.info(f"    Cached {len(result)} clusters to {cache_file}")

    return region, tech, result


# ---------------------------------------------------------------------------
# Aggregation: load medoid profiles, build output
# ---------------------------------------------------------------------------


def aggregate_all_clusters(
    all_results: List[Tuple[str, str, pd.DataFrame]],
    ds: xr.Dataset,
) -> Tuple[Dict, pd.DataFrame]:
    """Build clustered_data dict and metadata from clustering results.

    The representative profile is the medoid's real 8760 series — no synthetic
    averaging or rescaling. This preserves temporal correlations exactly.
    """
    clustered_data = {}
    meta_rows = []

    for region, tech, df_clusters in all_results:
        if df_clusters is None or len(df_clusters) == 0:
            continue

        for _, row in df_clusters.iterrows():
            cid = int(row["cluster_id"])
            medoid_bus = row["medoid_bus_id"]

            # Load real medoid profile
            try:
                cf_ts = (
                    ds["capacity_factor"].sel(bus=medoid_bus, technology=tech).values
                )
                cf_ts = np.clip(np.nan_to_num(cf_ts, nan=0.0), 0.0, 1.0)
            except Exception:
                cf_ts = np.zeros(8760)

            tech_label = "solar" if tech == "solar" else "onwind"
            cluster_name = f"{region}_{tech_label}_{cid}"

            clustered_data[(region, tech, cid)] = {
                "capacity": float(row["capacity_mw"]),
                "cf_ts": cf_ts,
                "avg_cf": float(row["avg_cf"]),
                "cf_high_mass": float(row["cf_high_mass"]),
                "cluster_name": cluster_name,
                "n_buses": int(row["n_buses"]),
                "representative_bus": medoid_bus,
                "stratum": row["stratum"],
            }

            meta_rows.append(
                {
                    "cluster_name": cluster_name,
                    "region": region,
                    "technology": tech,
                    "cluster_id": cid,
                    "stratum": row["stratum"],
                    "medoid_bus_id": medoid_bus,
                    "n_buses": int(row["n_buses"]),
                    "capacity_mw": float(row["capacity_mw"]),
                    "avg_cf": float(row["avg_cf"]),
                    "cf_high_mass": float(row["cf_high_mass"]),
                }
            )

    logger.info(f"  Total clusters: {len(clustered_data)}")
    return clustered_data, pd.DataFrame(meta_rows)


# ---------------------------------------------------------------------------
# NetCDF output
# ---------------------------------------------------------------------------


def write_clustered_netcdf(
    clustered_data: Dict, output_path: str, config: Dict
) -> None:
    """Write clustered dataset to NetCDF with (region, technology, class, time)."""
    logger.info(f"Writing clustered data to {output_path}")

    regions = sorted(set(k[0] for k in clustered_data))
    techs = sorted(set(str(k[1]) for k in clustered_data))
    time = np.arange(8760)

    max_classes = max(
        sum(1 for k in clustered_data if k[0] == r and str(k[1]) == t)
        for r in regions
        for t in techs
    )

    shape_3d = (len(regions), len(techs), max_classes)
    shape_4d = (*shape_3d, 8760)

    capacity_all = np.full(shape_3d, np.nan, dtype=np.float32)
    cf_all = np.full(shape_4d, np.nan, dtype=np.float32)
    avg_cf_all = np.full(shape_3d, np.nan, dtype=np.float32)
    cf_high_mass_all = np.full(shape_3d, np.nan, dtype=np.float32)

    for ri, region in enumerate(regions):
        for ti, tech in enumerate(techs):
            clusters = sorted(
                [
                    (cid, info)
                    for (r, t, cid), info in clustered_data.items()
                    if r == region and str(t) == tech
                ],
                key=lambda x: -x[1]["cf_high_mass"],  # merit order: best first
            )
            for ci, (cid, info) in enumerate(clusters):
                capacity_all[ri, ti, ci] = info["capacity"]
                avg_cf_all[ri, ti, ci] = info["avg_cf"]
                cf_high_mass_all[ri, ti, ci] = info["cf_high_mass"]
                ts = info["cf_ts"]
                if isinstance(ts, np.ndarray) and ts.ndim == 1 and len(ts) == 8760:
                    cf_all[ri, ti, ci, :] = ts.astype(np.float32)

    ds_out = xr.Dataset(
        {
            "capacity": (("region", "technology", "class"), capacity_all),
            "capacity_factor": (("region", "technology", "class", "time"), cf_all),
            "avg_cf": (("region", "technology", "class"), avg_cf_all),
            "cf_high_mass": (("region", "technology", "class"), cf_high_mass_all),
        },
        coords={
            "region": regions,
            "technology": techs,
            "time": time,
            "class": np.arange(max_classes),
        },
    )

    ds_out.attrs["clustering_method"] = "stratified_kmedoids"
    ds_out.attrs["cache_version"] = CACHE_VERSION
    ds_out.attrs["class_order"] = "descending cf_high_mass (merit order)"

    enc = {v: {"dtype": "float32"} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_path, encoding=enc)
    logger.info(f"  ✓ Wrote {output_path} — dims: {dict(ds_out.sizes)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_clustering(
    clustered_data: Dict,
    df_features: pd.DataFrame,
    ds: xr.Dataset,
    output_report: str,
) -> Dict:
    """Validate capacity preservation and supply curve shape."""
    logger.info("Validating clustering...")

    report: Dict[str, Any] = {
        "total_clusters": len(clustered_data),
        "capacity_preservation": {},
        "supply_curve_error": {},
    }

    region_tech_pairs = sorted({(k[0], str(k[1])) for k in clustered_data})

    for region, tech in region_tech_pairs:
        # --- Capacity preservation ---
        df_rt = df_features[
            (df_features["region"] == region) & (df_features["technology"] == tech)
        ]
        original_cap = float(df_rt["capacity_mw"].sum())

        clustered_cap = sum(
            info["capacity"]
            for (r, t, _), info in clustered_data.items()
            if r == region and str(t) == tech
        )

        pres_pct = 100.0 * clustered_cap / original_cap if original_cap > 0 else 100.0
        report["capacity_preservation"][f"{region}_{tech}"] = {
            "original_mw": original_cap,
            "clustered_mw": clustered_cap,
            "preservation_pct": pres_pct,
        }

        # --- Supply curve shape comparison ---
        # Original: sort buses by avg_cf descending, cumulative capacity
        orig_sorted = df_rt.sort_values("avg_cf", ascending=False)
        orig_cum_cap = np.cumsum(orig_sorted["capacity_mw"].values)
        orig_avg_cf = orig_sorted["avg_cf"].values

        # Clustered: sort clusters by avg_cf descending
        clusters = sorted(
            [
                info
                for (r, t, _), info in clustered_data.items()
                if r == region and str(t) == tech
            ],
            key=lambda x: -x["avg_cf"],
        )
        if clusters:
            clust_cum_cap = np.cumsum([c["capacity"] for c in clusters])
            clust_avg_cf = np.array([c["avg_cf"] for c in clusters])

            # Interpolate both curves at common capacity points and compute MAE
            if len(orig_cum_cap) > 1 and len(clust_cum_cap) > 1:
                max_cap = min(orig_cum_cap[-1], clust_cum_cap[-1])
                eval_points = np.linspace(0, max_cap, 50)
                orig_interp = np.interp(eval_points, orig_cum_cap, orig_avg_cf)
                clust_interp = np.interp(eval_points, clust_cum_cap, clust_avg_cf)
                mae = float(np.mean(np.abs(orig_interp - clust_interp)))

                # Error specifically in the first 10% (green pocket region)
                n10 = max(1, len(eval_points) // 10)
                mae_first10 = float(
                    np.mean(np.abs(orig_interp[:n10] - clust_interp[:n10]))
                )
            else:
                mae, mae_first10 = 0.0, 0.0

            report["supply_curve_error"][f"{region}_{tech}"] = {
                "mae_avg_cf": mae,
                "mae_first_10pct": mae_first10,
                "n_clusters": len(clusters),
            }

    avg_pres = np.mean(
        [v["preservation_pct"] for v in report["capacity_preservation"].values()]
    )
    logger.info(f"  Avg capacity preservation: {avg_pres:.1f}%")

    if report["supply_curve_error"]:
        avg_mae = np.mean(
            [v["mae_avg_cf"] for v in report["supply_curve_error"].values()]
        )
        avg_mae10 = np.mean(
            [v["mae_first_10pct"] for v in report["supply_curve_error"].values()]
        )
        logger.info(f"  Supply curve MAE (full): {avg_mae:.4f}")
        logger.info(f"  Supply curve MAE (first 10% / green pockets): {avg_mae10:.4f}")

    with open(output_report, "w") as f:
        json.dump(
            {"timestamp": pd.Timestamp.now().isoformat(), **report},
            f,
            indent=2,
        )
    logger.info(f"  ✓ Report: {output_report}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def filter_to_onwind_pv(ds: xr.Dataset) -> xr.Dataset:
    techs = [t for t in ["onwind", "solar"] if t in ds.technology.values]
    ds_filtered = ds.sel(technology=techs)
    logger.info(f"Filtered to {techs}: {len(ds_filtered.bus)} buses")
    return ds_filtered


def main():
    if snakemake is None:
        raise RuntimeError("Must run via Snakemake")

    logger.info("=" * 70)
    logger.info("STRATIFIED K-MEDOIDS RENEWABLE CLUSTERING")
    logger.info("=" * 70)

    config = snakemake.config
    clustering_config = config.get("clustering", {})

    # Load & filter
    ds = xr.open_dataset(str(snakemake.input.merged_cdf))
    ds = filter_to_onwind_pv(ds)

    bus_coords = load_bus_coordinates(str(snakemake.input.merged_geojson))
    logger.info(f"Loaded coordinates for {len(bus_coords)} buses")

    # Extract features
    cache_features = Path("resources") / f"features_cache_{CACHE_VERSION}.csv"
    df_features = extract_features(ds, config, bus_coords, str(cache_features))

    # Build region-tech pairs
    region_tech_pairs = (
        df_features.groupby(["region", "technology"])
        .size()
        .reset_index()[["region", "technology"]]
        .values.tolist()
    )
    logger.info(f"Clustering {len(region_tech_pairs)} region-technology pairs")

    # Cluster (parallel)
    cache_dir = Path("resources") / f"clustering_cache_{CACHE_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = getattr(snakemake, "threads", -1)

    with tqdm_joblib(
        tqdm(total=len(region_tech_pairs), desc="Clustering", unit="pair")
    ):
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(cluster_region_technology)(
                region, tech, df_features, ds, clustering_config, str(cache_dir)
            )
            for region, tech in region_tech_pairs
        )

    # Aggregate
    clustered_data, metadata_df = aggregate_all_clusters(results, ds)

    # Save metadata CSV alongside NetCDF
    meta_path = Path(str(snakemake.output.clustered)).with_suffix(".metadata.csv")
    metadata_df.to_csv(meta_path, index=False)
    logger.info(f"  ✓ Metadata: {meta_path}")

    # Write NetCDF
    write_clustered_netcdf(clustered_data, str(snakemake.output.clustered), config)

    # Validate
    validate_clustering(clustered_data, df_features, ds, str(snakemake.output.report))

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
