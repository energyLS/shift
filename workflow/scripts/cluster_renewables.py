"""
Cluster renewable generators from merged profiles for optimization.

This script:
1. Loads merged renewable profiles (all 122k+ buses from merged file)
2. Filters to onwind + pvplant (excludes unreliable offshore wind)
3. Maps buses to regions using ISO3 codes and config
4. Extracts 6D temporal features: avg_cf, temporal_std, cv, autocorr_24h, lat, lon
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
from typing import Dict, Tuple, List
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


@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to integrate tqdm progress bar with joblib Parallel.

    Usage:
        with tqdm_joblib(tqdm(total=n_tasks, desc="Processing")) as pbar:
            results = Parallel(n_jobs=4)(delayed(func)(i) for i in range(n_tasks))
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
    """Build mapping from ISO3 code to region name."""
    iso3_to_region_local = {}
    for region_name, iso3_list in config.get("regions", {}).items():
        for iso3 in iso3_list:
            iso3_to_region_local[iso3] = region_name
    return iso3_to_region_local


def extract_iso3_from_bus_id(bus_id: str) -> str:
    """Extract ISO3 code from bus_id format: {ISO3}_{ON/OFF}_{geohash}."""
    try:
        return bus_id.split("_")[0]
    except Exception:
        logger.warning(f"Could not extract ISO3 from bus_id: {bus_id}")
        return None


def load_merged_data(merged_path: str) -> xr.Dataset:
    """Load merged renewable profiles."""
    logger.info(f"Loading merged data from {merged_path}")
    ds = xr.open_dataset(merged_path)
    logger.info(f"  Merged data shape: {dict(ds.sizes)}")
    logger.info(f"  Variables: {list(ds.data_vars)}")
    return ds


def filter_to_onwind_pv(ds: xr.Dataset) -> xr.Dataset:
    """Filter merged data to onwind + solar only (exclude offshore)."""
    logger.info("Filtering to onwind + solar (excluding offwind-ac)")

    techs_present = list(ds.coords["technology"].values)
    # Actual technology names in merged file: 'offwind-ac', 'onwind', 'solar'
    techs_to_keep = ["onwind", "solar"]

    ds_filtered = ds.sel(technology=techs_to_keep)
    logger.info(f"  Kept technologies: {techs_to_keep}")
    logger.info(f"  Buses remaining: {len(ds_filtered.bus)}")

    return ds_filtered


def extract_bus_features_single(
    bus_id, ds: xr.Dataset, iso3_to_region: Dict[str, str]
) -> List[Dict]:
    """Extract features for a single bus across all technologies (internal)."""
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

        avg_cf = float(np.nanmean(cf_ts))
        temporal_std = float(np.nanstd(cf_ts))
        cv = temporal_std / avg_cf if avg_cf > 0 else 0.0

        if len(cf_ts) > 24:
            cf_detrended = cf_ts - np.nanmean(cf_ts)
            if np.std(cf_detrended) > 1e-10 and not np.isnan(cf_detrended[:-24]).all():
                autocorr_24h = float(
                    np.corrcoef(cf_detrended[:-24], cf_detrended[24:])[0, 1]
                )
            else:
                autocorr_24h = 0.0
        else:
            autocorr_24h = 0.0

        features.append(
            {
                "avg_cf": avg_cf,
                "temporal_std": temporal_std,
                "cv": cv,
                "autocorr_24h": autocorr_24h,
                "lat": y_centroid,
                "lon": x_centroid,
                "region": region,
                "technology": tech,
                "bus_id": bus_id,
            }
        )
    return features


def extract_bus_features_batch(
    bus_ids: List, ds: xr.Dataset, iso3_to_region: Dict[str, str]
) -> List[Dict]:
    """
    Extract features for a batch of buses (parallelizable task).

    Batching reduces scheduler overhead and improves cache locality.
    Returns flattened list of all feature dicts from all buses in batch.
    """
    all_features = []
    for bus_id in bus_ids:
        all_features.extend(extract_bus_features_single(bus_id, ds, iso3_to_region))
    return all_features


def extract_features(
    ds: xr.Dataset, config: Dict, cache_features_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Extract 6D features per bus: avg_cf, temporal_std, cv, autocorr_24h, lat, lon.

    Parallelized by batching buses to reduce scheduler overhead.

    Returns:
        DataFrame with columns: [avg_cf, temporal_std, cv, autocorr_24h, lat, lon, region, tech, bus_id]
        Dict: mapping region_name -> list of buses in that region
    """
    # Try to load from cache if it exists
    if cache_features_path and Path(cache_features_path).exists():
        logger.info(f"Loading cached features from {cache_features_path}")
        df_features = pd.read_csv(cache_features_path)
        region_buses = {}
        for _, row in df_features.iterrows():
            region = row["region"]
            bus_id = row["bus_id"]
            if region not in region_buses:
                region_buses[region] = []
            if bus_id not in region_buses[region]:
                region_buses[region].append(bus_id)
        logger.info(f"  Loaded {len(df_features)} features from cache")
        return df_features, region_buses

    logger.info("Extracting temporal features (batched parallelization)...")

    iso3_to_region = build_region_map(config, {})

    # Batch buses: 1 batch per worker for minimal overhead
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
    ) as pbar:
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
    """
    Cluster a single region-technology group using weighted K-means.

    Feature categories:
    - avg_cf: 1 feature (0.5 weight)
    - temporal: 3 features [temporal_std, cv, autocorr_24h] (0.3 weight)
    - geospatial: 2 features [lat, lon] (0.2 weight)
    """
    if len(df_subset) < n_clusters:
        logger.warning(
            f"  Subset has {len(df_subset)} buses < {n_clusters} clusters, adjusting k"
        )
        n_clusters = max(1, len(df_subset) // 2)

    # Extract feature subsets
    avg_cf_vals = df_subset[["avg_cf"]].values  # (n, 1)
    temporal_vals = df_subset[["temporal_std", "cv", "autocorr_24h"]].values  # (n, 3)
    geospatial_vals = df_subset[["lat", "lon"]].values  # (n, 2)

    # Normalize each category independently
    scaler_cf = StandardScaler()
    scaler_temporal = StandardScaler()
    scaler_geo = StandardScaler()

    avg_cf_norm = scaler_cf.fit_transform(avg_cf_vals)  # (n, 1)
    temporal_norm = scaler_temporal.fit_transform(temporal_vals)  # (n, 3)
    geospatial_norm = scaler_geo.fit_transform(geospatial_vals)  # (n, 2)

    # Apply weights
    w_cf = feature_weights["avg_cf"]
    w_temporal = feature_weights["temporal"]
    w_geo = feature_weights["geospatial"]

    avg_cf_weighted = avg_cf_norm * w_cf  # (n, 1)
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
    n_clusters_onwind: int,
    n_clusters_pvplant: int,
    feature_weights: Dict[str, float],
    cache_dir: str = None,
) -> Tuple[Tuple[str, str], np.ndarray]:
    """
    Cluster a single region-technology pair (parallelizable wrapper).

    Returns:
        ((region, tech), clusters) tuple for dict conversion
    """
    # Check cache first
    if cache_dir:
        cache_file = Path(cache_dir) / f"clustering_cache_{region}_{str(tech)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
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

    # Get n_clusters from tech type
    if tech == "solar":
        n_clusters = n_clusters_pvplant
        tech_name = "solar"
    else:
        n_clusters = n_clusters_onwind
        tech_name = "onwind"

    logger.info(f"Clustering {region} {tech_name} with k={n_clusters}")

    clusters, kmeans = cluster_region_technology(df_subset, n_clusters, feature_weights)

    # Cache this region-tech pair immediately
    if cache_dir:
        cache_file = Path(cache_dir) / f"clustering_cache_{region}_{str(tech)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
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
    """
    Select representative bus for cluster based on highest correlation to weighted-avg timeseries.

    Returns:
        representative_bus_id, scaled_cf_ts, avg_cf_cluster
    """
    tech = str(tech)  # Convert numpy.str_ to Python str
    max_corr = -2.0
    best_idx = 0
    best_bus_id = df_cluster.iloc[0]["bus_id"]

    for idx, (_, row) in enumerate(df_cluster.iterrows()):
        bus_id = row["bus_id"]

        try:
            cf_ts = ds["capacity_factor"].sel(bus=bus_id, technology=tech).values
            cf_ts = np.nan_to_num(cf_ts, nan=0.0)

            # Correlation to weighted-average
            corr = float(np.corrcoef(cf_ts, cf_weighted_ts)[0, 1])
            if corr > max_corr:
                max_corr = corr
                best_idx = idx
                best_bus_id = bus_id
        except Exception as e:
            logger.debug(f"Error computing correlation for {bus_id}: {e}")
            continue

    # Get representative timeseries
    cf_representative = (
        ds["capacity_factor"].sel(bus=best_bus_id, technology=tech).values
    )
    cf_representative = np.nan_to_num(cf_representative, nan=0.0)

    # Scale to match weighted-average merit-order
    avg_cf_cluster = float(np.mean(cf_weighted_ts))
    avg_cf_representative = float(np.mean(cf_representative))

    if avg_cf_representative > 0:
        scale_factor = avg_cf_cluster / avg_cf_representative
    else:
        scale_factor = 1.0

    cf_cluster_ts = cf_representative * scale_factor
    cf_cluster_ts = np.clip(cf_cluster_ts, 0, 1)

    return best_bus_id, cf_cluster_ts, avg_cf_cluster


def aggregate_clusters(
    df_features: pd.DataFrame,
    clusters_dict: Dict[Tuple[str, str], np.ndarray],
    ds: xr.Dataset,
    config: Dict,
    cache_clusters_path: str = None,
    clustering_cache_dir: str = None,
) -> Dict:
    """Aggregate clusters and generate outputs."""
    logger.info("Aggregating clusters...")

    # Fast path: load the fully aggregated payload if it already exists.
    if cache_clusters_path and Path(cache_clusters_path).exists():
        try:
            cached_payload = joblib.load(cache_clusters_path)
            if (
                "clustered_data" in cached_payload
                and "cluster_metadata" in cached_payload
            ):
                logger.info(
                    f"Loading cached aggregated clusters from {cache_clusters_path}"
                )
                return cached_payload["clustered_data"], cached_payload[
                    "cluster_metadata"
                ]
        except Exception as e:
            logger.info(
                f"Could not load cached aggregated clusters from {cache_clusters_path}: {e}"
            )

    # Try to load pre-computed cluster results from per-region-tech cache
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

            # Sum capacities
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

            # Compute weighted-average timeseries
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

            # Select representative bus
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
    """Write clustered data to NetCDF with (region, technology, class, time) dims."""
    logger.info(f"Writing clustered data to {output_path}")

    # Organize data by region and technology
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
    # Note: xarray doesn't support ragged dimensions natively, so we'll use max_classes
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
    logger.info(f"    Dimensions: {dict(ds_out.dims)}")
    logger.info(f"    Variables: {list(ds_out.data_vars)}")


def validate_clustering(
    clustered_data: Dict, ds_merged: xr.Dataset, output_report: str
) -> Dict:
    """Validate clustering integrity."""
    logger.info("Validating clustering...")

    validation_results = {
        "total_clusters": len(clustered_data),
        "capacity_preservation": {},
        "avg_cf_consistency": {},
        "temporal_quality": {},
    }

    region_tech_pairs = sorted(
        {(region, tech) for region, tech, _ in clustered_data.keys()}
    )

    for region, tech in region_tech_pairs:
        # Total capacity check
        original_cap = []
        for bus_id in ds_merged.bus.values:
            try:
                cap = float(
                    ds_merged["p_nom_max"].sel(bus=bus_id, technology=tech).values
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
    """Main clustering pipeline."""
    logger.info("=" * 70)
    logger.info("CLUSTERING RENEWABLE PROFILES FOR OPTIMIZATION")
    logger.info("=" * 70)

    # Load config
    config = snakemake.config
    logger.info(f"Clustering config: {config.get('clustering', {})}")

    clustering_config = config.get("clustering", {})
    n_clusters_onwind = clustering_config.get("n_clusters_onwind", 40)
    n_clusters_pvplant = clustering_config.get("n_clusters_pvplant", 40)
    feature_weights = clustering_config.get(
        "feature_weights",
        {
            "avg_cf": 0.5,
            "temporal": 0.3,
            "geospatial": 0.2,
        },
    )

    # Normalize weights
    weight_sum = sum(feature_weights.values())
    feature_weights = {k: v / weight_sum for k, v in feature_weights.items()}
    logger.info(f"Normalized feature weights: {feature_weights}")

    # Load and filter data
    ds = load_merged_data(str(snakemake.input.merged))
    ds_filtered = filter_to_onwind_pv(ds)

    # Extract features (with caching)
    cache_features = Path("resources") / "features_cache.csv"
    df_features, region_buses = extract_features(
        ds_filtered, config, str(cache_features)
    )

    # Cluster each region-technology group in parallel
    logger.info(f"Starting parallel clustering with {snakemake.threads} threads")

    # Build list of (region, tech) pairs to cluster
    region_tech_pairs = [
        (region, tech)
        for region in region_buses.keys()
        for tech in ds_filtered.technology.values
        if len(
            df_features[
                (df_features["region"] == region) & (df_features["technology"] == tech)
            ]
        )
        > 0
    ]

    logger.info(f"  Total region-technology pairs: {len(region_tech_pairs)}")

    # Create clustering cache directory
    clustering_cache_dir = Path("resources") / "clustering_cache"
    clustering_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Clustering cache directory: {clustering_cache_dir}")

    # Parallelize clustering across all region-tech pairs with tqdm progress
    with tqdm_joblib(
        tqdm(
            total=len(region_tech_pairs),
            desc="Clustering region-tech pairs",
            unit="pair",
        )
    ) as pbar:
        results = joblib.Parallel(n_jobs=snakemake.threads)(
            joblib.delayed(cluster_region_technology_pair)(
                region,
                tech,
                df_features,
                n_clusters_onwind,
                n_clusters_pvplant,
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

    # Aggregate clusters (with caching)
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

    # Validation
    validate_clustering(clustered_data, ds_filtered, str(snakemake.output.report))

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
