"""
Cluster renewable generators from merged profiles for optimization.

This script:
1. Loads merged renewable profiles (all 122k+ buses from merged file)
2. Filters to onwind + pvplant (excludes unreliable offshore wind)
3. Maps buses to regions using ISO3 codes and config
4. Extracts a unified capacity-factor summary plus lat/lon
5. Applies weighted K-means clustering (0.5 merit-order, 0.3 temporal, 0.2 geospatial)
6. Selects representative timeseries per cluster to preserve real patterns
7. Outputs clustered NetCDF in expected format: (region, technology, class, time)
8. Generates validation report with quality metrics

Usage (Snakemake rule):
    rule cluster_renewables:
        input:
            merged = "data/renewable_profiles_global_merged.nc",
            config = "config/config.yaml",
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

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

snakemake: Any = globals().get("snakemake")


CACHE_VERSION = "v3_cache_versioned_1d_cf_ts"
CF_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
CF_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


def summarize_capacity_factor(cf_ts: np.ndarray) -> Dict[str, float]:
    """Summarize a capacity-factor profile for clustering and validation.

    Args:
        cf_ts: One-dimensional capacity-factor time series. Missing values and
            out-of-range values are cleaned before summary statistics are
            computed.

    Returns:
        A dictionary of robust profile descriptors, including the core
        temporal moments used by clustering (`avg_cf`, `temporal_std`, `cv`,
        `autocorr_24h`) plus quantiles, exceedance fractions at key
        thresholds, and low/high tail mass statistics. These values are used
        to preserve merit-order shape when clustering and when selecting
        representative buses.
    """
    cf_clean = np.clip(np.nan_to_num(cf_ts, nan=0.0), 0.0, 1.0)
    if cf_clean.size == 0:
        cf_clean = np.zeros(1, dtype=float)
    avg_cf = float(np.nanmean(cf_clean))
    temporal_std = float(np.nanstd(cf_clean))
    cv = temporal_std / avg_cf if avg_cf > 0 else 0.0
    if len(cf_clean) > 24:
        cf_detrended = cf_clean - np.nanmean(cf_clean)
        if np.std(cf_detrended) > 1e-10 and not np.isnan(cf_detrended[:-24]).all():
            autocorr_24h = float(
                np.corrcoef(cf_detrended[:-24], cf_detrended[24:])[0, 1]
            )
        else:
            autocorr_24h = 0.0
    else:
        autocorr_24h = 0.0
    summary = {
        f"cf_q{int(q * 100):02d}": float(np.nanquantile(cf_clean, q))
        for q in CF_QUANTILES
    }
    for threshold in CF_THRESHOLDS:
        summary[f"cf_exceed_{int(threshold * 100):03d}"] = float(
            np.mean(cf_clean >= threshold)
        )
    tail = max(1, int(len(cf_clean) * 0.05))
    summary["avg_cf"] = avg_cf
    summary["temporal_std"] = temporal_std
    summary["cv"] = cv
    summary["autocorr_24h"] = autocorr_24h
    summary["cf_low_mass"] = float(np.mean(np.sort(cf_clean)[:tail]))
    summary["cf_high_mass"] = float(np.mean(np.sort(cf_clean)[-tail:]))
    return summary


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize clustering weights into the three feature families.

    Args:
        weights: Raw weights from config. Missing categories are treated as 0.

    Returns:
        A normalized dictionary with `merit_order`, `temporal`, and
        `geospatial` keys that sum to 1.0. If the provided weights are empty or
        invalid, the default policy of 0.5/0.3/0.2 is returned.
    """
    # Expect keys: merit_order, temporal, geospatial.
    w = dict(weights or {})
    for k in ("merit_order", "temporal", "geospatial"):
        w.setdefault(k, 0.0)
    s = sum(w.values())
    if s <= 0:
        return {"merit_order": 0.5, "temporal": 0.3, "geospatial": 0.2}
    return {k: float(v / s) for k, v in w.items()}


def rescale_timeseries_to_mean(cf_ts: np.ndarray, target_mean: float) -> np.ndarray:
    """Rescale a representative profile so its mean matches the cluster mean.

    Args:
        cf_ts: Representative capacity-factor series selected from observed
            buses.
        target_mean: Mean capacity factor that the cluster should preserve.

    Returns:
        A clipped capacity-factor series with the same shape as the input and a
        mean close to `target_mean`. The result stays within the physical bounds
        of [0, 1].
    """
    ts = np.clip(np.nan_to_num(cf_ts, nan=0.0).astype(float), 0.0, 1.0)
    if ts.size == 0:
        return ts
    current = float(np.mean(ts))
    if current <= 0:
        return np.full_like(ts, np.clip(target_mean, 0.0, 1.0))
    scaled = ts * (target_mean / current)
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled


def resolve_cluster_count(df_subset: pd.DataFrame, tech: str) -> int:
    """Resolve the cluster count for one region-technology subset.

    Args:
        df_subset: Feature rows for a single region and technology.
        tech: Technology label for the subset, typically `solar` or `onwind`.

    Returns:
        An integer cluster count derived from the configured policy. In dynamic
        mode, the count scales with bus count and installed capacity while being
        clamped to the configured min/max bounds.
    """
    clustering_config = snakemake.config.get("clustering", {})
    policy = (
        clustering_config.get("cluster_count_policy", {})
        if isinstance(clustering_config, dict)
        else {}
    )
    base_clusters = policy.get("base_clusters", {}) if isinstance(policy, dict) else {}
    mode = policy.get("mode", "dynamic")
    base_key = "solar" if tech == "solar" else "onwind"
    base = int(base_clusters.get(base_key, 40))
    if mode == "fixed":
        return base

    n_buses = max(1, int(df_subset["bus_id"].nunique()))
    total_capacity = (
        float(df_subset.get("capacity_mw", pd.Series([0])).sum())
        if "capacity_mw" in df_subset
        else 0.0
    )
    ref_buses = float(policy.get("reference_buses", 1000.0))
    ref_cap = float(policy.get("reference_capacity_mw", 50000.0))
    bus_scale = (n_buses / ref_buses) ** float(policy.get("scale_exponent", 0.5))
    cap_scale = (max(1.0, total_capacity) / ref_cap) ** float(
        policy.get("scale_exponent", 0.5)
    )
    bus_w = float(policy.get("bus_weight", 0.5))
    cap_w = float(policy.get("capacity_weight", 0.5))
    target = int(round(base * (bus_w * bus_scale + cap_w * cap_scale)))
    min_c = int(policy.get("min_clusters", 3))
    max_c = int(policy.get("max_clusters", 120))
    return max(min_c, min(max_c, max(1, min(target, len(df_subset)))))


def score_representative_candidate(
    cf_ts: np.ndarray,
    cf_target: np.ndarray,
    target_summary: Dict[str, float],
    candidate_summary: Dict[str, float],
) -> float:
    """Score how well a candidate bus matches the cluster target shape.

    Args:
        cf_ts: Candidate capacity-factor time series from an observed bus.
        cf_target: Capacity-factor series representing the cluster-average
            target.
        target_summary: Summary statistics for `cf_target`.
        candidate_summary: Summary statistics for the candidate series.

    Returns:
        A similarity score where higher values indicate a better match. The
        score combines correlation with a summary-statistic distance penalty so
        the chosen representative preserves both shape and distributional tail
        behavior.
    """
    cf_ts = np.clip(np.nan_to_num(cf_ts, nan=0.0), 0.0, 1.0)
    cf_target = np.clip(np.nan_to_num(cf_target, nan=0.0), 0.0, 1.0)
    corr = 0.0
    try:
        if np.std(cf_ts) > 1e-12 and np.std(cf_target) > 1e-12:
            corr = float(np.corrcoef(cf_ts, cf_target)[0, 1])
    except Exception:
        corr = 0.0
    keys = list(target_summary.keys())
    dist = 0.0
    for k in keys:
        dist += abs(candidate_summary.get(k, 0.0) - target_summary.get(k, 0.0))
    dist = dist / max(1, len(keys))
    return 0.7 * corr + 0.3 * (1.0 - dist)


@contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib progress callbacks into a tqdm progress bar.

    Args:
        tqdm_object: An active tqdm progress bar to update as joblib batches
            complete.

    Yields:
        The same tqdm object, allowing the caller to use the context manager as
        a transparent progress-wrapper around `joblib.Parallel` work.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


def build_region_map(config: Dict, iso3_to_region: Dict[str, str]) -> Dict[str, str]:
    """Build the ISO3-to-region lookup used to assign buses to regions.

    Args:
        config: Global configuration containing the region definitions.
        iso3_to_region: Existing lookup map. The parameter is accepted for
            interface compatibility, but the mapping is rebuilt from config.

    Returns:
        A dictionary mapping ISO3 country codes to region names.
    """
    iso3_to_region_local = {}
    for region_name, iso3_list in config.get("regions", {}).items():
        for iso3 in iso3_list:
            iso3_to_region_local[iso3] = region_name
    return iso3_to_region_local


def extract_iso3_from_bus_id(bus_id: str) -> Optional[str]:
    """Extract the ISO3 code embedded in a bus identifier.

    Args:
        bus_id: Renewable bus identifier, usually prefixed by an ISO3 code.

    Returns:
        The ISO3 prefix if it can be parsed, otherwise `None`.
    """
    try:
        return bus_id.split("_")[0]
    except Exception:
        logger.warning(f"Could not extract ISO3 from bus_id: {bus_id}")
        return None


def load_merged_data(merged_path: str) -> xr.Dataset:
    """Load the merged renewable-profiles dataset from disk.

    Args:
        merged_path: Path to the consolidated NetCDF input produced upstream.

    Returns:
        An opened xarray dataset containing bus-level renewable profiles and
        metadata needed by the clustering workflow.
    """
    logger.info(f"Loading merged data from {merged_path}")
    ds = xr.open_dataset(merged_path)
    logger.info(f"  Merged data shape: {dict(ds.sizes)}")
    logger.info(f"  Variables: {list(ds.data_vars)}")
    return ds


def filter_to_onwind_pv(ds: xr.Dataset) -> xr.Dataset:
    """Keep only the technologies that are clustered by this workflow.

    Args:
        ds: Full merged renewable dataset with multiple technologies.

    Returns:
        A reduced dataset containing only `onwind` and `solar` technology
        slices. Offshore wind is intentionally excluded because this workflow
        is tuned to onshore wind and solar pocket preservation.
    """
    logger.info("Filtering to onwind + solar (excluding offwind-ac)")

    _techs_present = list(ds.coords["technology"].values)
    # The merged file includes offshore wind, which this workflow intentionally skips.
    techs_to_keep = ["onwind", "solar"]

    ds_filtered = ds.sel(technology=techs_to_keep)
    logger.info(f"  Kept technologies: {techs_to_keep}")
    logger.info(f"  Buses remaining: {len(ds_filtered.bus)}")

    return ds_filtered


def extract_bus_features_single(
    bus_id, ds: xr.Dataset, iso3_to_region: Dict[str, str]
) -> List[Dict]:
    """Extract clustering features for one bus across all technologies.

    Args:
        bus_id: Bus identifier from the merged renewable dataset.
        ds: Filtered xarray dataset containing capacity factors and metadata.
        iso3_to_region: Lookup used to map ISO3 prefixes to analysis regions.

    Returns:
        A list of feature dictionaries, one per available bus-technology pair.
        Each row contains temporal moments, geospatial coordinates, capacity,
        and the tail-shape descriptors used by the clustering model.
    """
    features = []
    iso3 = extract_iso3_from_bus_id(str(bus_id))
    if not iso3:
        return features
    region = iso3_to_region.get(iso3, None)
    if not region:
        return features

    try:
        x_centroid = float(ds["x_centroid"].sel(bus=bus_id).values)
        y_centroid = float(ds["y_centroid"].sel(bus=bus_id).values)
    except Exception:
        x_centroid, y_centroid = 0.0, 0.0

    for tech in ds.technology.values:
        tech = str(tech)  # Convert numpy.str_ to Python str
        try:
            cf_ts = ds["capacity_factor"].sel(bus=bus_id, technology=tech).values
        except Exception:
            continue
        if np.isnan(cf_ts).all() or len(cf_ts) == 0:
            continue

        # Capture the tail of the capacity-factor distribution so green pockets survive.
        cf_summary = summarize_capacity_factor(cf_ts)

        try:
            capacity_mw = float(ds["p_nom_max"].sel(bus=bus_id, technology=tech).values)
        except Exception:
            capacity_mw = np.nan

        features.append(
            {
                "lat": y_centroid,
                "lon": x_centroid,
                "capacity_mw": capacity_mw,
                **cf_summary,
                "region": region,
                "technology": tech,
                "bus_id": bus_id,
            }
        )
    return features


def extract_bus_features_batch(
    bus_ids: List, ds: xr.Dataset, iso3_to_region: Dict[str, str]
) -> List[Dict]:
    """Extract features for a batch of buses.

    Args:
        bus_ids: Batch of bus identifiers assigned to one joblib worker.
        ds: Filtered renewable dataset.
        iso3_to_region: Lookup for region assignment.

    Returns:
        Flattened feature rows for all buses in the batch.
    """
    all_features = []
    for bus_id in bus_ids:
        all_features.extend(extract_bus_features_single(bus_id, ds, iso3_to_region))
    return all_features


def extract_features(
    ds: xr.Dataset, config: Dict, cache_features_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build the full feature table used by the clustering workflow.

    Args:
        ds: Filtered renewable dataset containing the buses to cluster.
        config: Workflow configuration used to derive the region mapping and
            caching behavior.
        cache_features_path: Optional CSV cache path for reusing extracted
            features on subsequent runs.

    Returns:
        A tuple of `(feature_table, region_buses)` where `feature_table` holds
        one row per bus-technology pair and `region_buses` maps each region to
        the buses assigned to it. The cache is reused when it matches the
        expected feature schema.
    """
    required_cache_cols = {
        "avg_cf",
        "temporal_std",
        "cv",
        "autocorr_24h",
        "lat",
        "lon",
        "capacity_mw",
        "region",
        "technology",
        "bus_id",
        "cf_q10",
        "cf_q25",
        "cf_q50",
        "cf_q75",
        "cf_q90",
        "cf_exceed_020",
        "cf_exceed_040",
        "cf_exceed_060",
        "cf_exceed_080",
        "cf_low_mass",
        "cf_high_mass",
    }

    # Try to load from cache if it exists
    if cache_features_path and Path(cache_features_path).exists():
        logger.info(f"Loading cached features from {cache_features_path}")
        df_features = pd.read_csv(cache_features_path)
        if required_cache_cols.issubset(set(df_features.columns)):
            region_buses = {}
            for _, row in df_features.iterrows():
                region = row["region"]
                bus_id = row["bus_id"]
                region_buses.setdefault(region, [])
                if bus_id not in region_buses[region]:
                    region_buses[region].append(bus_id)
            logger.info(f"  Loaded {len(df_features)} features from cache")
            return df_features, region_buses
        logger.info("  Cached feature schema is stale; recomputing")

    logger.info("Extracting temporal features (batched parallelization)...")

    iso3_to_region = build_region_map(config, {})

    # Batch buses so each worker gets a non-trivial chunk of work.
    n_jobs = snakemake.threads if hasattr(snakemake, "threads") else -1

    bus_list = list(ds.bus.values)
    num_batches = abs(n_jobs) if n_jobs != -1 else 1
    batch_size = (len(bus_list) + num_batches - 1) // num_batches
    bus_batches = [
        bus_list[i : i + batch_size] for i in range(0, len(bus_list), batch_size)
    ]

    logger.info(
        f"  Using {n_jobs} threads, batch size: {batch_size} buses, {len(bus_batches)} batches (1 per worker)"
    )

    with tqdm_joblib(
        tqdm(total=len(bus_batches), desc="Extracting bus features", unit="batch")
    ):
        batch_results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(extract_bus_features_batch)(batch, ds, iso3_to_region)
            for batch in bus_batches
        )

    # Flatten batch results
    features_list = []
    region_buses = {}

    for bus_features_list in batch_results:
        for feat_dict in bus_features_list:
            features_list.append(feat_dict)
            region = feat_dict["region"]
            if region not in region_buses:
                region_buses[region] = []
            if feat_dict["bus_id"] not in region_buses[region]:
                region_buses[region].append(feat_dict["bus_id"])

    df_features = pd.DataFrame(features_list)
    logger.info(
        f"  Extracted {len(df_features)} bus-technology pairs across {len(region_buses)} regions"
    )
    logger.info(
        f"  Region distribution: {dict((k, len(set(v))) for k, v in region_buses.items())}"
    )

    # Cache features to disk for recovery if later steps fail
    if cache_features_path:
        df_features.to_csv(cache_features_path, index=False)
        logger.info(f"  ✓ Cached features to {cache_features_path}")

    return df_features, region_buses


def cluster_region_technology(
    df_subset: pd.DataFrame, n_clusters: int, feature_weights: Dict[str, float]
) -> Tuple[np.ndarray, KMeans]:
    """Cluster one region-technology subset with weighted K-means.

    Args:
        df_subset: Feature rows for a single region and technology.
        n_clusters: Requested number of clusters before any size safeguards.
        feature_weights: Normalized weights for merit-order, temporal, and
            geospatial feature families.

    Returns:
        The fitted cluster labels for each row in `df_subset` and the trained
        `KMeans` estimator. The feature matrix is built from separately
        normalized feature families so the configured weights act at the family
        level rather than on raw columns.
    """
    if len(df_subset) < n_clusters:
        logger.warning(
            f"  Subset has {len(df_subset)} buses < {n_clusters} clusters, adjusting k"
        )
        n_clusters = max(1, len(df_subset) // 2)

    # The merit-order block keeps the upper tail visible to clustering.
    merit_order_cols = [
        "avg_cf",
        "cf_q10",
        "cf_q25",
        "cf_q50",
        "cf_q75",
        "cf_q90",
        "cf_exceed_020",
        "cf_exceed_040",
        "cf_exceed_060",
        "cf_exceed_080",
        "cf_low_mass",
        "cf_high_mass",
    ]

    # Extract feature subsets
    merit_order_vals = df_subset[merit_order_cols].values
    temporal_vals = df_subset[["temporal_std", "cv", "autocorr_24h"]].values  # (n, 3)
    geospatial_vals = df_subset[["lat", "lon"]].values  # (n, 2)

    # Normalize each category independently
    scaler_cf = StandardScaler()
    scaler_temporal = StandardScaler()
    scaler_geo = StandardScaler()

    merit_order_norm = scaler_cf.fit_transform(merit_order_vals)
    temporal_norm = scaler_temporal.fit_transform(temporal_vals)  # (n, 3)
    geospatial_norm = scaler_geo.fit_transform(geospatial_vals)  # (n, 2)

    # Apply the three feature-family weights after separate normalization.
    w_cf = feature_weights["merit_order"]
    w_temporal = feature_weights["temporal"]
    w_geo = feature_weights["geospatial"]

    avg_cf_weighted = merit_order_norm * w_cf
    temporal_weighted = temporal_norm * w_temporal  # (n, 3)
    geospatial_weighted = geospatial_norm * w_geo  # (n, 2)

    # Concatenate all weighted features
    X = np.hstack([avg_cf_weighted, temporal_weighted, geospatial_weighted])  # (n, 6)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)

    return clusters, kmeans


def cluster_region_technology_pair(
    region: str,
    tech: str,
    df_features: pd.DataFrame,
    clustering_config: Dict,
    feature_weights: Dict[str, float],
    cache_dir: Optional[str] = None,
) -> Tuple[Tuple[str, str], np.ndarray]:
    """Cluster one region-technology pair and persist the assignments.

    Args:
        region: Region name selected from the configuration mapping.
        tech: Technology label for the subset.
        df_features: Full extracted feature table.
        clustering_config: Clustering-specific configuration section.
        feature_weights: Normalized family weights used by the K-means model.
        cache_dir: Optional directory for storing per-pair cluster assignments.

    Returns:
        A `(region, tech)` key paired with the cluster-label vector for the
        corresponding rows in `df_features`. Cached results are reused when
        present and compatible with the current cache version.
    """
    tech = str(tech)

    # Check cache first
    if cache_dir:
        cache_file = Path(cache_dir) / f"clustering_cache_{region}_{str(tech)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    if cached.get("cache_version") != CACHE_VERSION:
                        logger.info(
                            f"  [STALE CACHE] {region} {tech}: cache version {cached.get('cache_version')} != {CACHE_VERSION}"
                        )
                        raise ValueError("stale cache version")
                    clusters_array = np.array(cached["clusters"])
                    logger.info(
                        f"  [CACHED] {region} {str(tech)}: loaded {len(cached['clusters'])} assignments"
                    )
                    return (region, tech), clusters_array
            except Exception as e:
                logger.debug(f"Could not load cache for {region} {tech}: {e}")

    df_subset = df_features[
        (df_features["region"] == region) & (df_features["technology"] == tech)
    ]

    if len(df_subset) == 0:
        return (region, tech), np.array([])

    n_clusters = resolve_cluster_count(df_subset, tech)
    tech_name = "solar" if tech == "solar" else "onwind"

    logger.info(f"Clustering {region} {tech_name} with k={n_clusters}")

    clusters, kmeans = cluster_region_technology(df_subset, n_clusters, feature_weights)

    # Cache this region-tech pair immediately
    if cache_dir:
        cache_file = Path(cache_dir) / f"clustering_cache_{region}_{str(tech)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "cache_version": CACHE_VERSION,
                        "region": region,
                        "technology": str(tech),
                        "n_clusters": len(np.unique(clusters)),
                        "clusters": clusters.tolist(),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    },
                    f,
                )
            logger.info(f"  [CACHED] {region} {str(tech)}: saved clustering result")
        except Exception as e:
            logger.warning(f"Could not cache {region} {tech}: {e}")

    return (region, tech), clusters


def select_representative_bus(
    df_cluster: pd.DataFrame, ds: xr.Dataset, tech: str, cf_weighted_ts: np.ndarray
) -> Tuple[str, np.ndarray, float]:
    """Select the best observed bus for a cluster and re-center its mean.

    Args:
        df_cluster: Rows belonging to one cluster within a region-technology
            subset.
        ds: Filtered renewable dataset used to retrieve candidate time series.
        tech: Technology label for the cluster.
        cf_weighted_ts: Capacity-weighted cluster-average profile.

    Returns:
        A tuple containing the chosen bus identifier, the representative time
        series rescaled to the exact cluster mean, and the cluster mean itself.
        The representative is chosen to preserve both the central tendency and
        the tail shape of the cluster.
    """
    tech = str(tech)  # Convert numpy.str_ to Python str
    target_summary = summarize_capacity_factor(cf_weighted_ts)
    best_score = -np.inf
    best_bus_id = df_cluster.iloc[0]["bus_id"]
    best_cf_ts = None

    for _, row in df_cluster.iterrows():
        bus_id = row["bus_id"]

        try:
            cf_ts = ds["capacity_factor"].sel(bus=bus_id, technology=tech).values
            cf_ts = np.nan_to_num(cf_ts, nan=0.0)
            candidate_summary = summarize_capacity_factor(cf_ts)
            score = score_representative_candidate(
                cf_ts,
                cf_weighted_ts,
                target_summary,
                candidate_summary,
            )
            if score > best_score:
                best_score = score
                best_bus_id = bus_id
                best_cf_ts = cf_ts
        except Exception as e:
            logger.debug(f"Error computing correlation for {bus_id}: {e}")
            continue

    # Prefer a real series; the rescaling step keeps the cluster mean exact.
    if best_cf_ts is None:
        best_cf_ts = ds["capacity_factor"].sel(bus=best_bus_id, technology=tech).values
    cf_representative = np.nan_to_num(best_cf_ts, nan=0.0)

    avg_cf_cluster = float(np.mean(cf_weighted_ts))
    cf_cluster_ts = rescale_timeseries_to_mean(cf_representative, avg_cf_cluster)

    return best_bus_id, cf_cluster_ts, avg_cf_cluster


def aggregate_clusters(
    df_features: pd.DataFrame,
    clusters_dict: Dict[Tuple[str, str], np.ndarray],
    ds: xr.Dataset,
    config: Dict,
    cache_clusters_path: Optional[str] = None,
    clustering_cache_dir: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Aggregate cluster assignments into pseudo-bus outputs and metadata.

    Args:
        df_features: Full feature table used to recover cluster membership.
        clusters_dict: Mapping from `(region, tech)` to the cluster labels for
            the corresponding feature rows.
        ds: Filtered renewable dataset used to sum capacities and profiles.
        config: Global workflow configuration.
        cache_clusters_path: Optional joblib cache for the aggregated payload.
        clustering_cache_dir: Optional directory containing per-pair caches of
            cluster assignments.

    Returns:
        A tuple of `(clustered_data, cluster_metadata)`. `clustered_data` holds
        the pseudo-bus output for each cluster, while `cluster_metadata`
        records the summary fields that downstream steps can inspect.
    """
    logger.info("Aggregating clusters...")

    # Fast path: load the fully aggregated payload if it already exists.
    if cache_clusters_path and Path(cache_clusters_path).exists():
        try:
            cached_payload = joblib.load(cache_clusters_path)
            if (
                cached_payload.get("cache_version") == CACHE_VERSION
                and "clustered_data" in cached_payload
                and "cluster_metadata" in cached_payload
            ):
                logger.info(
                    f"Loading cached aggregated clusters from {cache_clusters_path}"
                )
                return cached_payload["clustered_data"], cached_payload[
                    "cluster_metadata"
                ]
            logger.info(
                f"Discarding stale aggregated cluster cache at {cache_clusters_path}"
            )
            Path(cache_clusters_path).unlink(missing_ok=True)
        except Exception as e:
            logger.info(
                f"Could not load cached aggregated clusters from {cache_clusters_path}: {e}"
            )

    # Recover per-region-tech assignments from cache if we have them.
    if clustering_cache_dir:
        clustering_cache_path = Path(clustering_cache_dir)
        cache_files = list(clustering_cache_path.glob("clustering_cache_*.json"))
        if cache_files:
            logger.info(
                f"Found {len(cache_files)} cached region-tech clustering results"
            )
            for cache_file in cache_files:
                try:
                    with open(cache_file, "r") as f:
                        cached = json.load(f)
                        if cached.get("cache_version") != CACHE_VERSION:
                            logger.info(
                                f"  [STALE] {cache_file.name}: cache version {cached.get('cache_version')} != {CACHE_VERSION}"
                            )
                            cache_file.unlink(missing_ok=True)
                            continue
                        region = cached["region"]
                        tech = cached["technology"]
                        clusters_array = np.array(cached["clusters"])

                        # Only add to clusters_dict if not already present
                        if (region, tech) not in clusters_dict or len(
                            clusters_dict[(region, tech)]
                        ) == 0:
                            clusters_dict[(region, tech)] = clusters_array
                            logger.info(
                                f"  [RECOVERED] {region} {tech}: loaded {len(clusters_array)} cluster assignments from cache"
                            )
                except Exception as e:
                    logger.debug(f"Could not load {cache_file}: {e}")

    clustered_data = {}  # {(region, tech, cluster_id): {capacity, cf_ts, avg_cf}}
    cluster_metadata = {}

    for (region, tech), cluster_assignments in tqdm(
        clusters_dict.items(), desc="Aggregating region-tech pairs", unit="pair"
    ):
        tech = str(tech)  # Convert numpy.str_ to Python str
        df_subset = df_features[
            (df_features["region"] == region) & (df_features["technology"] == tech)
        ]

        n_clusters = len(np.unique(cluster_assignments))
        logger.info(
            f"  {region} {tech}: {len(df_subset)} buses -> {n_clusters} clusters"
        )

        for cluster_id in tqdm(
            range(n_clusters),
            desc=f"  {region}-{tech} clusters",
            unit="cluster",
            leave=False,
        ):
            mask = cluster_assignments == cluster_id
            df_cluster = df_subset[mask].reset_index(drop=True)

            if len(df_cluster) == 0:
                continue

            # Get buses in cluster
            bus_ids_in_cluster = list(df_cluster["bus_id"].values)

            # Sum capacities within the cluster to preserve total installed potential.
            capacities = []
            for bus_id in bus_ids_in_cluster:
                try:
                    cap = float(
                        ds["p_nom_max"].sel(bus=bus_id, technology=str(tech)).values
                    )
                    if not np.isnan(cap):
                        capacities.append(cap)
                except Exception:
                    pass

            p_nom_max_cluster = float(np.sum(capacities))

            if p_nom_max_cluster <= 0:
                logger.debug(
                    f"  Skipping cluster {region}/{tech}/{cluster_id}: no valid capacity"
                )
                continue

            # Compute the cluster-average profile before picking a representative bus.
            cf_weighted_parts = []
            for bus_id in bus_ids_in_cluster:
                try:
                    cap = float(
                        ds["p_nom_max"].sel(bus=bus_id, technology=str(tech)).values
                    )
                    cf = (
                        ds["capacity_factor"]
                        .sel(bus=bus_id, technology=str(tech))
                        .values
                    )
                    if not np.isnan(cap) and cap > 0:
                        cf = np.nan_to_num(cf, nan=0.0)
                        cf_weighted_parts.append(cap * cf)
                except Exception:
                    pass

            if cf_weighted_parts:
                cf_weighted_ts = np.sum(cf_weighted_parts, axis=0) / p_nom_max_cluster
            else:
                cf_weighted_ts = np.zeros(8760)

            # Select representative bus and then rescale to the exact cluster mean CF.
            representative_id, cf_cluster_ts, avg_cf_cluster = (
                select_representative_bus(df_cluster, ds, str(tech), cf_weighted_ts)
            )

            # Cluster name: Regionname_tech_number
            tech_name = "solar" if tech == "solar" else "onwind"
            cluster_name = f"{region}_{tech_name}_{cluster_id}"

            clustered_data[(region, tech, cluster_id)] = {
                "capacity": p_nom_max_cluster,
                "cf_ts": cf_cluster_ts,
                "avg_cf": avg_cf_cluster,
                "cluster_name": cluster_name,
                "n_buses": len(df_cluster),
                "representative_bus": representative_id,
            }

            cluster_metadata[cluster_name] = {
                "region": region,
                "technology": tech,
                "cluster_id": cluster_id,
                "n_buses_consolidated": len(df_cluster),
                "representative_bus_id": representative_id,
                "total_capacity_mw": p_nom_max_cluster,
                "avg_cf": float(avg_cf_cluster),
            }

    logger.info(f"  Total clusters created: {len(clustered_data)}")

    # Cache aggregated clusters to disk for recovery
    if cache_clusters_path:
        logger.info(f"Caching aggregated clusters to {cache_clusters_path}")
        joblib.dump(
            {
                "cache_version": CACHE_VERSION,
                "clustered_data": clustered_data,
                "cluster_metadata": cluster_metadata,
                "timestamp": pd.Timestamp.now().isoformat(),
            },
            cache_clusters_path,
            compress=3,
        )
        logger.info("  ✓ Cached full clustered payload")

    return clustered_data, cluster_metadata


def write_clustered_netcdf(
    clustered_data: Dict, output_path: str, config: Dict
) -> None:
    """Write the clustered pseudo-bus dataset to NetCDF.

    Args:
        clustered_data: Aggregated cluster payload produced by
            `aggregate_clusters`.
        output_path: Destination NetCDF path.
        config: Global configuration used to stamp metadata and preserve the
            workflow context.

    Returns:
        None. The clustered dataset is written to `output_path` with padded
        region/technology/class dimensions so downstream consumers can load it
        as a regular xarray dataset.
    """
    logger.info(f"Writing clustered data to {output_path}")

    # Organize data by region and technology so we can emit ragged class arrays.
    regions = list(set(k[0] for k in clustered_data.keys()))
    techs = list(set(str(k[1]) for k in clustered_data.keys()))  # Convert to string

    logger.info(f"  Regions: {regions}")
    logger.info(f"  Technologies: {techs}")

    # Build xarray dataset
    time = np.arange(8760)

    # Initialize data variables
    capacity_data = {}
    cf_data = {}
    avg_cf_data = {}

    for region in regions:
        capacity_data[region] = {}
        cf_data[region] = {}
        avg_cf_data[region] = {}

        for tech in techs:
            # Collect all clusters for this region-tech
            clusters_for_rt = [
                (cluster_id, clustered_data[(region, tech, cluster_id)])
                for (r, t, cluster_id) in clustered_data.keys()
                if r == region and t == tech
            ]

            n_clusters = len(clusters_for_rt)
            if n_clusters == 0:
                continue

            capacity_data[region][tech] = np.zeros(n_clusters)
            cf_data[region][tech] = np.zeros((n_clusters, 8760))
            avg_cf_data[region][tech] = np.zeros(n_clusters)

            for idx, (cluster_id, cluster_info) in enumerate(clusters_for_rt):
                capacity_data[region][tech][idx] = cluster_info["capacity"]
                cf_ts = cluster_info["cf_ts"]
                if isinstance(cf_ts, np.ndarray):
                    if cf_ts.ndim != 1:
                        logger.warning(
                            f"  cf_ts has wrong shape {cf_ts.shape}, taking first row"
                        )
                        cf_ts = cf_ts[0] if cf_ts.ndim > 1 else cf_ts
                    if len(cf_ts) == 8760:
                        cf_data[region][tech][idx, :] = cf_ts.astype(np.float32)
                avg_cf_data[region][tech][idx] = cluster_info["avg_cf"]

    # Create xarray dataset with aligned dimensions
    ds_out = xr.Dataset(
        data_vars={},
        coords={
            "region": regions,
            "technology": techs,
            "time": time,
        },
    )

    # Add data variables with heterogeneous class dimension per region-tech
    # xarray does not support ragged dimensions natively, so we pad to max_classes.
    max_classes = max(
        len([k for k in clustered_data.keys() if k[0] == r and str(k[1]) == t])
        for r in regions
        for t in techs
    )

    capacity_all = np.full(
        (len(regions), len(techs), max_classes), np.nan, dtype=np.float32
    )
    cf_all = np.full(
        (len(regions), len(techs), max_classes, 8760), np.nan, dtype=np.float32
    )
    avg_cf_all = np.full(
        (len(regions), len(techs), max_classes), np.nan, dtype=np.float32
    )

    for region_idx, region in enumerate(regions):
        for tech_idx, tech in enumerate(techs):
            clusters_for_rt = [
                (cluster_id, clustered_data[(region, tech_orig, cluster_id)])
                for (r, tech_orig, cluster_id) in clustered_data.keys()
                if r == region and str(tech_orig) == tech
            ]

            for class_idx, (cluster_id, cluster_info) in enumerate(clusters_for_rt):
                cf_ts = cluster_info["cf_ts"]
                if isinstance(cf_ts, np.ndarray):
                    # Ensure cf_ts is 1D
                    if cf_ts.ndim != 1:
                        logger.warning(
                            f"  cf_ts has wrong shape {cf_ts.shape}, taking first row"
                        )
                        cf_ts = cf_ts[0] if cf_ts.ndim > 1 else cf_ts
                    if len(cf_ts) == 8760:
                        cf_all[region_idx, tech_idx, class_idx, :] = cf_ts.astype(
                            np.float32
                        )

                capacity_all[region_idx, tech_idx, class_idx] = float(
                    cluster_info["capacity"]
                )
                avg_cf_all[region_idx, tech_idx, class_idx] = float(
                    cluster_info["avg_cf"]
                )

    ds_out["capacity"] = (("region", "technology", "class"), capacity_all)
    ds_out["capacity_factor"] = (("region", "technology", "class", "time"), cf_all)
    ds_out["avg_cf"] = (("region", "technology", "class"), avg_cf_all)

    # Add metadata attributes
    ds_out.attrs["clustering_algorithm"] = "weighted_kmeans"
    ds_out.attrs["temporal_aware"] = "True"
    ds_out.attrs["feature_weights"] = json.dumps(
        {"avg_cf": 0.5, "temporal": 0.3, "geospatial": 0.2}
    )

    ds_out.to_netcdf(
        output_path,
        encoding={
            "capacity": {"dtype": "float32"},
            "capacity_factor": {"dtype": "float32"},
            "avg_cf": {"dtype": "float32"},
        },
    )

    logger.info(f"  ✓ Wrote {output_path}")
    logger.info(f"    Dimensions: {dict(ds_out.sizes)}")
    logger.info(f"    Variables: {list(ds_out.data_vars)}")


def validate_clustering(
    clustered_data: Dict,
    df_features: pd.DataFrame,
    ds_merged: xr.Dataset,
    output_report: str,
) -> Dict:
    """Validate that clustering preserves capacity and profile structure.

    Args:
        clustered_data: Aggregated cluster payload returned by the workflow.
        df_features: Source feature table used to compute baseline statistics.
        ds_merged: Filtered merged dataset used to compare the original input
            capacity totals.
        output_report: Destination path for the JSON validation report.

    Returns:
        A nested dictionary of validation metrics covering capacity
        preservation, mean capacity-factor consistency, merit-order retention,
        and threshold-based capacity retention.
    """
    logger.info("Validating clustering...")

    validation_results = {
        "total_clusters": len(clustered_data),
        "capacity_preservation": {},
        "avg_cf_consistency": {},
        "merit_order_preservation": {},
        "temporal_quality": {},
    }

    region_tech_pairs = sorted(
        {(region, tech) for region, tech, _ in clustered_data.keys()}
    )

    for region, tech in region_tech_pairs:
        df_subset = df_features[
            (df_features["region"] == region) & (df_features["technology"] == str(tech))
        ]

        # Total capacity check remains the basic integrity guard.
        original_cap = []
        for bus_id in df_subset["bus_id"].values:
            try:
                cap = float(
                    ds_merged["p_nom_max"].sel(bus=bus_id, technology=str(tech)).values
                )
                if not np.isnan(cap):
                    original_cap.append(cap)
            except Exception:
                pass

        original_total = np.sum(original_cap)
        clustered_total = sum(
            cluster_info.get("capacity", 0)
            for (r, t, _), cluster_info in clustered_data.items()
            if r == region and str(t) == str(tech)
        )

        if original_total > 0:
            preservation_pct = (clustered_total / original_total) * 100
        else:
            preservation_pct = 100.0

        validation_results["capacity_preservation"][f"{region}_{tech}"] = {
            "original_mw": float(original_total),
            "clustered_mw": float(clustered_total),
            "preservation_pct": float(preservation_pct),
        }

        cluster_rows = [
            cluster_info
            for (r, t, _), cluster_info in clustered_data.items()
            if r == region and str(t) == str(tech)
        ]
        if cluster_rows:
            cluster_avg_cf = np.array(
                [row["avg_cf"] for row in cluster_rows], dtype=float
            )
            cluster_capacity = np.array(
                [row["capacity"] for row in cluster_rows], dtype=float
            )
            merit_corr = float(
                pd.Series(cluster_avg_cf).corr(pd.Series(cluster_capacity))
            )

            thresholds = {}
            for threshold in CF_THRESHOLDS:
                retained_capacity = float(
                    np.sum(cluster_capacity[cluster_avg_cf >= threshold])
                )
                total_capacity = float(np.sum(cluster_capacity))
                # This metric answers whether the expensive/high-quality pocket survives.
                thresholds[f"{threshold}"] = {
                    "retained_capacity_mw": retained_capacity,
                    "retained_pct": float(100.0 * retained_capacity / total_capacity)
                    if total_capacity > 0
                    else 100.0,
                }

            validation_results["avg_cf_consistency"][f"{region}_{tech}"] = {
                "avg_cf_mean": float(np.mean(df_subset["avg_cf"].values))
                if len(df_subset)
                else None,
            }
            validation_results["merit_order_preservation"][f"{region}_{tech}"] = {
                "cluster_capacity_avg_cf_corr": merit_corr,
                "threshold_retention": thresholds,
            }

    logger.info(
        f"  Capacity preservation: {np.mean([v['preservation_pct'] for v in validation_results['capacity_preservation'].values()]):.1f}%"
    )

    # Write validation report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "validation_results": validation_results,
        "total_clusters": len(clustered_data),
    }

    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  ✓ Wrote validation report to {output_report}")

    return validation_results


def main():
    """Run the full renewable clustering workflow end to end.

    The entrypoint loads the merged renewable profiles, filters the supported
    technologies, extracts and caches features, clusters each region-technology
    subset, aggregates pseudo-bus outputs, writes the clustered NetCDF, and
    finally emits a validation report.
    """
    if snakemake is None:
        raise RuntimeError(
            "This script must be executed through Snakemake so the injected `snakemake` object is available."
        )

    logger.info("=" * 70)
    logger.info("CLUSTERING RENEWABLE PROFILES FOR OPTIMIZATION")
    logger.info("=" * 70)

    # Load config and normalize the feature weights before any clustering starts.
    config = snakemake.config
    logger.info(f"Clustering config: {config.get('clustering', {})}")

    clustering_config = config.get("clustering", {})
    feature_weights = clustering_config.get(
        "feature_weights",
        {
            "merit_order": 0.5,
            "temporal": 0.3,
            "geospatial": 0.2,
        },
    )

    # Normalize weights
    feature_weights = normalize_weights(feature_weights)
    logger.info(f"Normalized feature weights: {feature_weights}")

    # Load and filter the merged renewable profiles.
    ds = load_merged_data(str(snakemake.input.merged))
    ds_filtered = filter_to_onwind_pv(ds)

    # Extract features (with caching).
    cache_features = Path("resources") / "features_cache.csv"
    df_features, region_buses = extract_features(
        ds_filtered, config, str(cache_features)
    )

    # Cluster each region-technology group in parallel.
    logger.info(f"Starting parallel clustering with {snakemake.threads} threads")

    # Build list of region-tech pairs that actually exist in the filtered data.
    region_tech_pairs = [
        (region, str(tech))
        for region in region_buses.keys()
        for tech in ds_filtered.technology.values
        if len(
            df_features[
                (df_features["region"] == region)
                & (df_features["technology"] == str(tech))
            ]
        )
        > 0
    ]

    logger.info(f"  Total region-technology pairs: {len(region_tech_pairs)}")

    # Cache region-tech assignments so repeated runs can recover quickly.
    clustering_cache_dir = Path("resources") / "clustering_cache"
    clustering_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Clustering cache directory: {clustering_cache_dir}")

    # Parallelize clustering across all region-tech pairs with tqdm progress.
    with tqdm_joblib(
        tqdm(
            total=len(region_tech_pairs),
            desc="Clustering region-tech pairs",
            unit="pair",
        )
    ):
        results = joblib.Parallel(n_jobs=snakemake.threads)(
            joblib.delayed(cluster_region_technology_pair)(
                region,
                tech,
                df_features,
                clustering_config,
                feature_weights,
                str(clustering_cache_dir),
            )
            for region, tech in tqdm(
                region_tech_pairs,
                desc="Queuing clustering tasks",
                unit="pair",
                leave=False,
            )
        )

    # Convert results to dict
    clusters_dict = dict(results)

    # Aggregate clusters into pseudo-buses and write the output NetCDF.
    cache_clusters = Path("resources") / "clusters_cache.joblib"
    clustered_data, cluster_metadata = aggregate_clusters(
        df_features,
        clusters_dict,
        ds_filtered,
        config,
        str(cache_clusters),
        str(clustering_cache_dir),
    )

    # Write output
    write_clustered_netcdf(clustered_data, str(snakemake.output.clustered), config)

    # Validation report for capacity and merit-order preservation.
    validate_clustering(
        clustered_data, df_features, ds_filtered, str(snakemake.output.report)
    )

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
