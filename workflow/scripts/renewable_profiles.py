"""
Renewable Profiles Preprocessing Script

Builds per-bus renewable energy profiles from PyPSA-Earth data.
Pure functions — no hardcoded globals, fully parameterized for notebook/Snakemake use.

Public API:
    build_profiles()             — Main orchestrator (7-stage pipeline)
    load_pypsa_earth_profiles()  — Load raw PyPSA-Earth profile data
    load_region_boundaries()     — Load GeoJSON region boundaries
    save_profiles()              — Write NetCDF + GeoJSON + metadata.json
    load_profiles()              — Load saved files with version check
    audit_profiles_against_raw() — Compare processed vs raw stats (min/max/mean/NaN)

Visualization API:
    plot_grid_potentials()        — Grid-level raster map (GW/cell)
    plot_bus_capacity_density()   — Bus regions colored by capacity density (MW/km²)
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pycountry
import geohash2
from tqdm import tqdm

# Visualization imports (required)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader as shprdr

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Region/technology compatibility rules.
REGION_TECH_COMPAT = {
    "onshore": {"onwind", "solar"},
    "offshore": {"offwind-ac"},
}

# Plotting style constants
MAP_STYLE = {
    "ne_scale": "50m",
    "buffer": 1.0,
    "ocean_color": "#e6f2ff",
    "land_color": "#f5f5f5",
    "coastline_color": "#1a1a1a",
    "coastline_width": 1.0,
    "border_color": "#1a1a1a",
    "border_width": 1.5,
    "grid_color": "gray",
}

# Cache Natural Earth data (expensive to load)
_NE_COUNTRIES_CACHE = None


def load_pypsa_earth_profiles(pypsa_earth_path, technologies=None):
    """
    Load renewable technology profiles from PyPSA-Earth.

    Parameters
    ----------
    pypsa_earth_path : str or Path
        Path to pypsa-earth repository
    technologies : list, optional
        List of technologies to load. Default: ["onwind", "offwind-ac", "solar"]

    Returns
    -------
    dict
        Technology-keyed dict of xarray Datasets
    """
    if technologies is None:
        technologies = ["onwind", "offwind-ac", "solar"]

    pypsa_earth_path = Path(pypsa_earth_path)
    tech_profiles = {}

    logger.info("Loading renewable technology profiles from PyPSA-Earth...")
    for tech in technologies:
        path = (
            pypsa_earth_path / "resources" / "renewable_profiles" / f"profile_{tech}.nc"
        )
        try:
            ds = xr.open_dataset(path)
            tech_profiles[tech] = ds
            n_hours = len(ds.coords.get("time", ds.coords.get("hour", [])))
            logger.info(
                f"  ✓ {tech}: {n_hours} hours, {len(ds.bus)} buses, "
                f"grid {len(ds.x)}×{len(ds.y)}"
            )
        except FileNotFoundError:
            logger.warning(f"  ✗ {tech}: File not found at {path}")
        except Exception as e:
            logger.warning(
                f"  ✗ {tech}: Failed to load ({type(e).__name__}: {str(e)[:50]})"
            )

    if not tech_profiles:
        raise RuntimeError("No renewable technology profiles could be loaded!")

    logger.info(f"✓ Loaded {len(tech_profiles)} profiles: {list(tech_profiles.keys())}")
    return tech_profiles


def load_region_boundaries(pypsa_earth_path):
    """
    Load onshore and offshore region boundaries from GeoJSON.

    Parameters
    ----------
    pypsa_earth_path : str or Path
        Path to pypsa-earth repository

    Returns
    -------
    tuple
        (onshore_gpd, offshore_gpd) — GeoDataFrames
    """
    pypsa_earth_path = Path(pypsa_earth_path)

    logger.info("Loading geographic region boundaries from GeoJSON...")
    onshore_gpd = gpd.read_file(
        pypsa_earth_path / "resources" / "bus_regions" / "regions_onshore.geojson"
    )
    offshore_gpd = gpd.read_file(
        pypsa_earth_path / "resources" / "bus_regions" / "regions_offshore.geojson"
    )

    logger.info(f"  Onshore: {len(onshore_gpd)} regions")
    logger.info(f"  Offshore: {len(offshore_gpd)} regions")

    return onshore_gpd, offshore_gpd


def _prepare_regions(onshore_gpd, offshore_gpd, country_codes=None):
    """
    Prepare regions: normalize country codes and calculate areas.

    Returns
    -------
    GeoDataFrame
        Combined onshore + offshore regions with standardized columns
    """
    onshore_gpd = onshore_gpd.copy()
    offshore_gpd = offshore_gpd.copy()

    # Convert ISO2 → ISO3 country codes
    onshore_gpd["country"] = onshore_gpd["country"].apply(
        lambda iso2: pycountry.countries.get(alpha_2=iso2).alpha_3
    )
    offshore_gpd["country"] = offshore_gpd["country"].apply(
        lambda iso2: pycountry.countries.get(alpha_2=iso2).alpha_3
    )

    # Filter countries if specified
    if country_codes is not None:
        onshore_gpd = onshore_gpd[onshore_gpd["country"].isin(country_codes)]
        offshore_gpd = offshore_gpd[offshore_gpd["country"].isin(country_codes)]
        logger.info(f"Filtered to countries: {country_codes}")

    # Keep original PyPSA region IDs unchanged for robust matching to raw profiles.
    onshore_gpd["name"] = onshore_gpd["name"].astype(str)
    offshore_gpd["name"] = offshore_gpd["name"].astype(str)

    # Calculate area in km² using EPSG:6933 projection
    onshore_gpd["area_km2"] = onshore_gpd.to_crs("EPSG:6933").geometry.area / 1e6
    offshore_gpd["area_km2"] = offshore_gpd.to_crs("EPSG:6933").geometry.area / 1e6

    # Mark region type
    onshore_gpd["onshore_offshore"] = "onshore"
    offshore_gpd["onshore_offshore"] = "offshore"

    # Combine
    all_regions = pd.concat([onshore_gpd, offshore_gpd], ignore_index=True)
    logger.info(f"Combined: {len(all_regions)} total regions")

    return all_regions


def _generate_bus_ids(regions, geohash_precision=6):
    """Generate bus IDs as ISO3_ON/OFF_geohash with collision handling."""
    centroids = regions.geometry.centroid
    lons = centroids.x.values
    lats = centroids.y.values

    # Vectorized geohash encoding
    geohash_func = np.vectorize(
        lambda lat, lon: geohash2.encode(lat, lon, precision=geohash_precision)
    )
    geohashes = geohash_func(lats, lons)
    country_codes = regions["country"].astype(str).values
    region_flags = np.where(
        regions["onshore_offshore"].astype(str).values == "offshore",
        "OFF",
        "ON",
    )
    base_ids = [
        f"{country}_{flag}_{gh}"
        for country, flag, gh in zip(country_codes, region_flags, geohashes)
    ]

    # Handle collisions
    bus_ids = []
    collision_count = 0
    seen_hashes = {}

    for base_id in base_ids:
        if base_id in seen_hashes:
            seen_hashes[base_id] += 1
            bus_id = f"{base_id}_{seen_hashes[base_id]:03d}"
            collision_count += 1
        else:
            seen_hashes[base_id] = 0
            bus_id = base_id
        bus_ids.append(bus_id)

    logger.info(
        f"Generated {len(bus_ids)} bus IDs ({collision_count} collisions handled)"
    )
    return bus_ids


def _cache_tech_data(profile_datasets):
    """
    Cache technology datasets: trim to 8760 hours, sort grids, pre-compute fast lookups.

    Returns
    -------
    dict
        Technology-keyed cache with profiles, potentials, p_nom_max arrays,
        coordinates, and normalized bus→index mapping.
    """
    tech_data_cache = {}

    for tech, ds in profile_datasets.items():
        profile_8760 = ds["profile"].isel(time=slice(0, 8760))
        if "bus" in profile_8760.dims and "time" in profile_8760.dims:
            profile_8760 = profile_8760.transpose("bus", "time")
        profile_8760 = profile_8760.values.astype(np.float32)

        potential = ds["potential"].values.astype(np.float32)
        x_coords = ds.coords["x"].values.astype(np.float32)
        y_coords = ds.coords["y"].values.astype(np.float32)
        p_nom_max = ds["p_nom_max"].values.astype(np.float32)

        # Ensure grids are sorted for binary search
        if not np.all(np.diff(x_coords) > 0):
            x_sort_idx = np.argsort(x_coords)
            x_coords = x_coords[x_sort_idx]
            potential = potential[:, x_sort_idx]

        if not np.all(np.diff(y_coords) > 0):
            y_sort_idx = np.argsort(y_coords)
            y_coords = y_coords[y_sort_idx]
            potential = potential[y_sort_idx, :]

        # Pre-compute bus→index lookup
        bus_to_idx = {str(bus_id): idx for idx, bus_id in enumerate(ds.bus.values)}

        tech_data_cache[tech] = {
            "profile": profile_8760,
            "potential": potential,
            "p_nom_max": p_nom_max,
            "x": x_coords,
            "y": y_coords,
            "bus_to_idx": bus_to_idx,
        }

    logger.info(f"Cached {len(tech_data_cache)} technology datasets")
    return tech_data_cache


def _nearest_index(sorted_values, target):
    """Return index of nearest value in an ascending 1D array."""
    idx = np.searchsorted(sorted_values, target)
    if idx <= 0:
        return 0
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    return idx - 1 if abs(target - left) <= abs(right - target) else idx


def _extract_profiles_for_bus(
    region_name, region_type, lon, lat, tech, tech_data_cache
):
    """
    Extract profile data for a single bus-technology pair.

    Returns
    -------
    tuple or None
        (cf, p_nom_max, avg_cf, quality_flag) or None if missing/incompatible
    """
    allowed_techs = REGION_TECH_COMPAT.get(region_type, set())
    if tech not in allowed_techs:
        return None

    cache = tech_data_cache[tech]
    bus_to_idx = cache["bus_to_idx"]
    region_name_str = str(region_name)

    # Offshore regions are prefixed as OFF_* in prepared regions, while raw
    # PyPSA-Earth bus IDs are often unprefixed numeric names.
    candidate_names = [region_name_str]
    if region_name_str.startswith("OFF_"):
        candidate_names.append(region_name_str[4:])

    bus_idx = None
    for candidate in candidate_names:
        if candidate in bus_to_idx:
            bus_idx = bus_to_idx[candidate]
            break

    if bus_idx is None:
        return None

    try:
        cf = cache["profile"][bus_idx, :]
        p_nom_max = float(cache["p_nom_max"][bus_idx])

        x_idx = _nearest_index(cache["x"], lon)
        y_idx = _nearest_index(cache["y"], lat)
        potential_val = cache["potential"][y_idx, x_idx]

        avg_cf = float(np.nanmean(cf)) if not np.isnan(cf).all() else np.nan

        if np.isnan(p_nom_max) and not np.isnan(potential_val):
            p_nom_max = float(potential_val)

        quality_flag = (avg_cf > 0) and (p_nom_max > 0)
        return (cf, p_nom_max, avg_cf, quality_flag)

    except Exception as e:
        logger.debug(
            f"Error extracting {region_name}/{tech}: {type(e).__name__}: {str(e)[:80]}"
        )
        return None


def _reconcile_grids(tech_data_cache, technologies):
    """
    Reconcile grid extents across all technologies.

    Technologies may have identical spacing (e.g., 0.25°) but different start/end points.
    This function finds the union of all extents and returns a unified grid that covers all data.

    Returns
    -------
    tuple
        (grid_x, grid_y, reconciled_potentials_dict)
    """
    logger.info("Reconciling grid extents across technologies...")

    # Extract grids from all technologies
    grids = {
        tech: (tech_data_cache[tech]["x"], tech_data_cache[tech]["y"])
        for tech in technologies
    }

    # Get reference spacing from first technology
    ref_x, ref_y = grids[technologies[0]]
    reference_dx = np.diff(ref_x).mean()
    reference_dy = np.diff(ref_y).mean()

    logger.info(f"  Reference spacing: dx={reference_dx:.6f}, dy={reference_dy:.6f}")

    # Verify all technologies have compatible spacing
    for tech in technologies[1:]:
        x_tech, y_tech = grids[tech]
        dx = np.diff(x_tech).mean()
        dy = np.diff(y_tech).mean()

        if not (
            np.isclose(dx, reference_dx, rtol=1e-3)
            and np.isclose(dy, reference_dy, rtol=1e-3)
        ):
            raise ValueError(
                f"Spacing mismatch for {tech}: dx={dx:.6f} vs {reference_dx:.6f}, "
                f"dy={dy:.6f} vs {reference_dy:.6f}. All technologies must have compatible spacing."
            )

    # Find union bounds
    x_min_union = min(x.min() for x, _ in grids.values())
    x_max_union = max(x.max() for x, _ in grids.values())
    y_min_union = min(y.min() for _, y in grids.values())
    y_max_union = max(y.max() for _, y in grids.values())

    # Reconstruct unified grid with proper uniform spacing
    n_x = int(np.round((x_max_union - x_min_union) / reference_dx)) + 1
    n_y = int(np.round((y_max_union - y_min_union) / reference_dy)) + 1

    grid_x = np.linspace(x_min_union, x_max_union, n_x)
    grid_y = np.linspace(y_min_union, y_max_union, n_y)

    logger.info(
        f"  Union bounds: X=[{x_min_union:.4f}, {x_max_union:.4f}], Y=[{y_min_union:.4f}, {y_max_union:.4f}]"
    )
    logger.info(
        f"  Unified grid: {len(grid_x)} × {len(grid_y)}, spacing dx={np.diff(grid_x).mean():.6f}, dy={np.diff(grid_y).mean():.6f}"
    )

    # Map each technology's potential to unified grid
    reconciled_potentials = {}
    for tech in technologies:
        x_old, y_old = grids[tech]
        potential_old = tech_data_cache[tech]["potential"]

        if potential_old.shape != (len(grid_y), len(grid_x)):
            # Calculate where old grid starts in new grid (in grid indices)
            dx_new = np.diff(grid_x).mean()
            dy_new = np.diff(grid_y).mean()

            x_offset = int(np.round((x_old[0] - grid_x[0]) / dx_new))
            y_offset = int(np.round((y_old[0] - grid_y[0]) / dy_new))

            # Clamp to valid range
            x_offset = max(0, min(x_offset, len(grid_x)))
            y_offset = max(0, min(y_offset, len(grid_y)))

            # Pad with NaN
            padded = np.full((len(grid_y), len(grid_x)), np.nan, dtype=np.float32)
            y_end = min(y_offset + potential_old.shape[0], len(grid_y))
            x_end = min(x_offset + potential_old.shape[1], len(grid_x))

            padded[y_offset:y_end, x_offset:x_end] = potential_old[
                : y_end - y_offset, : x_end - x_offset
            ]

            reconciled_potentials[tech] = padded
            logger.info(
                f"  {tech}: padded {potential_old.shape} → {padded.shape} (offset: y={y_offset}, x={x_offset})"
            )
        else:
            reconciled_potentials[tech] = potential_old

    logger.info("✓ Grid reconciliation complete")
    return grid_x, grid_y, reconciled_potentials


def build_profiles(
    profile_datasets,
    onshore_regions_gpd,
    offshore_regions_gpd,
    config=None,
):
    """
    Main orchestrator: 7-stage pipeline to build renewable profiles.

    ============================================================================
    OUTPUT DATA FORMAT
    ============================================================================

    This function produces TWO complementary data structures:

    1. xarray.Dataset (Energy Data)
       ─────────────────────────────
       Dimensions: [bus, technology, hour, y_grid, x_grid]

       Coordinates:
         • bus: Unique renewable region IDs representing a voronoi cell (format: ISO3_ON/OFF_geohash)
         • technology: ["onwind", "offwind-ac", "solar"]
         • hour: 0–8759 (hourly steps in a year, Jan 1 – Dec 30)
         • x_grid, y_grid: 0.25° × 0.25° grid cell corners

       Data Variables (all float32):
         • capacity_factor[bus, tech, hour]: Hourly CF timeseries (0–1)
         • p_nom_max[bus, tech]: Max installable capacity (MW)
         • avg_cf[bus, tech]: Annual average capacity factor
         • potential[y_grid, x_grid, tech]: Grid-level potential (GW/cell)
         • weight[bus]: Area-normalized weight (sum=1 across all buses)
         • data_quality_flag[bus, tech]: Boolean indicating data completeness

       → Saved to NetCDF (.nc) with zlib compression, chunked by bus

    2. GeoDataFrame (Geometry & Attributes)
       ──────────────────────────────────
       Columns:
         • bus_id: Unique identifier (matches Dataset bus coordinate)
         • pypsa_region_id: Original PyPSA-Earth region name
         • country: ISO3 country code
         • onshore_offshore: "onshore" or "offshore"
         • x_centroid, y_centroid: Polygon centroid (lon, lat)
         • area_km2: Voronoi cell area in km²
         • geometry: WKT polygon (Voronoi cell boundary)

       → Saved to GeoJSON (.geojson) with full spatial reference

    Note: GIS data (geometries, country, area) are stored ONLY in GeoJSON,
          not duplicated in NetCDF (reduces file size from ~27GB → ~558MB).
          Use geometry_gdf for all spatial operations and attribute lookups.

    ============================================================================

    Parameters
    ----------
    profile_datasets : dict
        Technology-keyed dict of xarray Datasets
    onshore_regions_gpd : GeoDataFrame
        Onshore regions
    offshore_regions_gpd : GeoDataFrame
        Offshore regions
    config : dict, optional
        Configuration with keys:
        - country_codes (list): ISO3 country codes to filter to [None = all]
        - geohash_precision (int): 1-12 [default: 6]
        - process_by_country (bool): Sequential processing to reduce memory [default: True]

    Returns
    -------
    tuple
        (dataset, geometry_gdf) where:
        - dataset: xr.Dataset with capacity_factor, p_nom_max, avg_cf, potential
        - geometry_gdf: GeoDataFrame with bus_id, country, onshore_offshore, centroids, area_km2
    """
    if config is None:
        config = {}

    country_codes = config.get("country_codes", None)
    geohash_precision = config.get("geohash_precision", 6)
    process_by_country = config.get("process_by_country", True)

    technologies = list(profile_datasets.keys())
    logger.info(f"Building profiles for: {technologies}")

    # ===== STAGE 0: Prepare Regions =====
    logger.info("STAGE 0: Prepare regions")
    all_regions = _prepare_regions(
        onshore_regions_gpd, offshore_regions_gpd, country_codes
    )

    # Validate region typing contract used throughout extraction.
    if "onshore_offshore" not in all_regions.columns:
        raise ValueError(
            "Missing required column 'onshore_offshore' after region preparation"
        )
    valid_region_types = {"onshore", "offshore"}
    found_region_types = set(all_regions["onshore_offshore"].astype(str).unique())
    if not found_region_types.issubset(valid_region_types):
        raise ValueError(
            f"Invalid values in 'onshore_offshore': {sorted(found_region_types)}. "
            f"Expected subset of {sorted(valid_region_types)}"
        )

    # ===== STAGE 1: Generate Bus IDs =====
    logger.info("STAGE 1: Generate bus IDs")
    all_regions["pypsa_region_id"] = all_regions["name"].astype(str)
    bus_ids = _generate_bus_ids(all_regions, geohash_precision)
    all_regions["bus_id"] = bus_ids
    all_regions["geometry_wkt"] = all_regions.geometry.apply(lambda geom: geom.wkt)

    # ===== STAGE 2: Cache Tech Data =====
    logger.info("STAGE 2: Cache technology data")
    tech_data_cache = _cache_tech_data(profile_datasets)

    # ===== STAGE 3: Extract Profiles =====
    logger.info("STAGE 3: Extract profiles for all buses")
    n_buses = len(all_regions)
    n_techs = len(technologies)
    n_hours = 8760

    cf_array = np.full((n_buses, n_techs, n_hours), np.nan, dtype=np.float32)
    p_nom_max_array = np.full((n_buses, n_techs), np.nan, dtype=np.float32)
    avg_cf_array = np.full((n_buses, n_techs), np.nan, dtype=np.float32)
    quality_flag_array = np.full((n_buses, n_techs), False, dtype=bool)

    # Pre-extract arrays once to avoid expensive per-row DataFrame access.
    region_names = all_regions["name"].astype(str).values
    region_types = all_regions["onshore_offshore"].astype(str).values
    countries_arr = all_regions["country"].astype(str).values
    centroids = all_regions.geometry.centroid
    x_coords = centroids.x.values.astype(np.float32)
    y_coords = centroids.y.values.astype(np.float32)

    if process_by_country:
        countries_list = sorted(np.unique(countries_arr))
        logger.info(f"Processing {len(countries_list)} countries sequentially")

        for country_idx, country in enumerate(countries_list):
            bus_indices = np.where(countries_arr == country)[0]
            logger.info(
                f"  [{country_idx + 1}/{len(countries_list)}] {country}: {len(bus_indices)} regions"
            )

            for bus_idx in bus_indices:
                for tech_idx, tech in enumerate(technologies):
                    result = _extract_profiles_for_bus(
                        region_name=region_names[bus_idx],
                        region_type=region_types[bus_idx],
                        lon=float(x_coords[bus_idx]),
                        lat=float(y_coords[bus_idx]),
                        tech=tech,
                        tech_data_cache=tech_data_cache,
                    )

                    if result is not None:
                        cf, p_nom_max, avg_cf, flag = result
                        cf_array[bus_idx, tech_idx, :] = cf
                        p_nom_max_array[bus_idx, tech_idx] = p_nom_max
                        avg_cf_array[bus_idx, tech_idx] = avg_cf
                        quality_flag_array[bus_idx, tech_idx] = flag
    else:
        for bus_idx in tqdm(range(n_buses), total=n_buses):
            for tech_idx, tech in enumerate(technologies):
                result = _extract_profiles_for_bus(
                    region_name=region_names[bus_idx],
                    region_type=region_types[bus_idx],
                    lon=float(x_coords[bus_idx]),
                    lat=float(y_coords[bus_idx]),
                    tech=tech,
                    tech_data_cache=tech_data_cache,
                )

                if result is not None:
                    cf, p_nom_max, avg_cf, flag = result
                    cf_array[bus_idx, tech_idx, :] = cf
                    p_nom_max_array[bus_idx, tech_idx] = p_nom_max
                    avg_cf_array[bus_idx, tech_idx] = avg_cf
                    quality_flag_array[bus_idx, tech_idx] = flag

    logger.info(
        f"Extracted {np.sum(~np.isnan(p_nom_max_array))} bus-technology combinations"
    )

    # ===== STAGE 4: Reconcile Grids =====
    logger.info("STAGE 4: Reconcile grid extents")
    grid_x, grid_y, reconciled_potentials = _reconcile_grids(
        tech_data_cache, technologies
    )

    # ===== STAGE 5: Create xarray Dataset =====
    logger.info("STAGE 5: Create xarray dataset")

    # Vectorized extraction (replaces 6× iterrows() calls with direct array access)
    bus_ids_final = all_regions["bus_id"].values
    countries = countries_arr
    onshore_offshore = region_types
    area_km2_coords = all_regions["area_km2"].values.astype(np.float32)

    area_sum = np.sum(area_km2_coords)
    weights = area_km2_coords / area_sum

    # 3D potential array
    n_y_grid = len(grid_y)
    n_x_grid = len(grid_x)
    grid_potential_3d = np.full((n_y_grid, n_x_grid, n_techs), np.nan, dtype=np.float32)
    for tech_idx, tech in enumerate(technologies):
        grid_potential_3d[:, :, tech_idx] = reconciled_potentials[tech]

    # Create dataset (only energy/profile data, no GIS metadata)
    dataset = xr.Dataset(
        {
            "capacity_factor": (["bus", "technology", "hour"], cf_array),
            "p_nom_max": (["bus", "technology"], p_nom_max_array),
            "avg_cf": (["bus", "technology"], avg_cf_array),
            "potential": (["y_grid", "x_grid", "technology"], grid_potential_3d),
            "weight": (["bus"], weights),
            "data_quality_flag": (["bus", "technology"], quality_flag_array),
        },
        coords={
            "bus": bus_ids_final,
            "technology": technologies,
            "hour": np.arange(n_hours, dtype=np.int32),
            "x_grid": grid_x,
            "y_grid": grid_y,
        },
    )

    # ===== STAGE 6: Add Metadata =====
    logger.info("STAGE 6: Add metadata")

    valid_entries = np.sum(quality_flag_array)
    total_entries = n_buses * n_techs

    dataset.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "created": datetime.now().isoformat(),
            "technologies": ",".join(technologies),
            "geohash_precision": geohash_precision,
            "total_buses": len(bus_ids_final),
            "complete_rate": f"{100 * valid_entries / total_entries:.1f}%",
        }
    )

    # ===== Prepare Geometry GeoDataFrame =====
    logger.info("Preparing geometry GeoDataFrame")

    geometry_gdf = gpd.GeoDataFrame(
        {
            "bus_id": bus_ids_final,
            "pypsa_region_id": all_regions["pypsa_region_id"].values.astype(str),
            "country": countries,
            "onshore_offshore": onshore_offshore,
            "x_centroid": x_coords,
            "y_centroid": y_coords,
            "area_km2": area_km2_coords,
        },
        geometry=all_regions.geometry.values,
        crs="EPSG:4326",
    )

    logger.info("✓ Profile building complete")
    return dataset, geometry_gdf


def save_profiles(
    dataset, geometry_gdf, output_dir, filename_prefix="renewable_profiles"
):
    """
    Save profiles to NetCDF + GeoJSON + metadata.json.

    Parameters
    ----------
    dataset : xr.Dataset
        From build_profiles()
    geometry_gdf : GeoDataFrame
        From build_profiles()
    output_dir : str or Path
        Output directory
    filename_prefix : str
        Filename prefix (before timestamp)

    Returns
    -------
    tuple
        (nc_path, geojson_path, metadata_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nc_filename = f"{filename_prefix}__{timestamp}.nc"
    geojson_filename = f"{filename_prefix}__{timestamp}.geojson"
    metadata_filename = f"{filename_prefix}__{timestamp}_metadata.json"

    nc_path = output_dir / nc_filename
    geojson_path = output_dir / geojson_filename
    metadata_path = output_dir / metadata_filename

    # Save NetCDF with compression
    logger.info(f"Saving NetCDF to {nc_path}")

    # Build encoding dict only for float variables in energy data
    float_vars = ["capacity_factor", "p_nom_max", "avg_cf", "potential", "weight"]
    encoding = {}
    for var in float_vars:
        if var in dataset.data_vars:
            encoding[var] = {"dtype": "float32", "zlib": True, "complevel": 4}

    # Save with bus as unlimited dimension (allows future appending)
    dataset.to_netcdf(nc_path, encoding=encoding, unlimited_dims=["bus"])

    # Save GeoJSON
    logger.info(f"Saving GeoJSON to {geojson_path}")
    geometry_gdf.to_file(geojson_path, driver="GeoJSON")

    # Save metadata with version pinning
    logger.info(f"Saving metadata to {metadata_path}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "technologies": dataset.attrs.get("technologies", "").split(","),
        "n_buses": int(dataset.attrs.get("total_buses", 0)),
        "nc_file": nc_filename,
        "geojson_file": geojson_filename,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved: {nc_path.name}, {geojson_path.name}, {metadata_path.name}")
    return str(nc_path), str(geojson_path), str(metadata_path)


def load_profiles(nc_path, geojson_path, metadata_path=None):
    """
    Load saved profiles with schema version check.

    Parameters
    ----------
    nc_path : str or Path
        Path to NetCDF file
    geojson_path : str or Path
        Path to GeoJSON file
    metadata_path : str or Path, optional
        Path to metadata.json for version check

    Returns
    -------
    tuple
        (dataset, geometry_gdf)
    """
    nc_path = Path(nc_path)
    geojson_path = Path(geojson_path)

    # Load metadata and check version
    if metadata_path:
        metadata_path = Path(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        if metadata.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"Schema version mismatch: file={metadata.get('schema_version')}, "
                f"code={SCHEMA_VERSION}. Attempting to load anyway..."
            )

    # Load NetCDF
    logger.info(f"Loading NetCDF from {nc_path}")
    dataset = xr.open_dataset(nc_path)

    # Load GeoJSON
    logger.info(f"Loading GeoJSON from {geojson_path}")
    geometry_gdf = gpd.read_file(geojson_path)

    logger.info(
        f"✓ Loaded {len(dataset.bus)} buses, {len(dataset.technology)} technologies"
    )
    return dataset, geometry_gdf


def _array_stats(arr):
    """Compute min/max/mean and NaN counts for a numeric array."""
    arr = np.asarray(arr)
    total = int(arr.size)
    nan_count = int(np.isnan(arr).sum())
    valid = arr[~np.isnan(arr)]

    if valid.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "nan_count": nan_count,
            "total": total,
            "nan_share_pct": float(100 * nan_count / max(total, 1)),
        }

    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "nan_count": nan_count,
        "total": total,
        "nan_share_pct": float(100 * nan_count / max(total, 1)),
    }


def audit_profiles_against_raw(profiles_ds, raw_profile_datasets, logger_instance=None):
    """
    Compare processed dataset stats against raw PyPSA-Earth datasets.

    Parameters
    ----------
    profiles_ds : xr.Dataset
        Processed dataset from build_profiles().
    raw_profile_datasets : dict[str, xr.Dataset]
        Raw technology-keyed datasets from load_pypsa_earth_profiles().
    logger_instance : logging.Logger, optional
        Logger to use for output. Defaults to module logger.

    Returns
    -------
    dict
        Nested stats by technology and variable.
    """
    log = logger_instance or logger
    audit = {}

    log.info("=" * 70)
    log.info("PROCESSED vs RAW AUDIT")
    log.info("=" * 70)

    for tech in profiles_ds.technology.values:
        tech_key = str(tech)
        if tech_key not in raw_profile_datasets:
            log.warning(
                f"Skipping {tech_key}: technology not present in raw_profile_datasets"
            )
            continue

        raw_ds = raw_profile_datasets[tech_key]
        tech_result = {}

        log.info("\n" + "-" * 70)
        log.info(f"Technology: {tech_key}")
        log.info("-" * 70)

        # Potential stats
        proc_potential = profiles_ds["potential"].sel(technology=tech_key).values
        raw_potential = raw_ds["potential"].values
        proc_potential_stats = _array_stats(proc_potential)
        raw_potential_stats = _array_stats(raw_potential)

        log.info("potential stats")
        log.info(
            f"  processed: min={proc_potential_stats['min']:.6g}, max={proc_potential_stats['max']:.6g}, "
            f"mean={proc_potential_stats['mean']:.6g}, NaN={proc_potential_stats['nan_count']}/{proc_potential_stats['total']} "
            f"({proc_potential_stats['nan_share_pct']:.2f}%)"
        )
        log.info(
            f"  raw:       min={raw_potential_stats['min']:.6g}, max={raw_potential_stats['max']:.6g}, "
            f"mean={raw_potential_stats['mean']:.6g}, NaN={raw_potential_stats['nan_count']}/{raw_potential_stats['total']} "
            f"({raw_potential_stats['nan_share_pct']:.2f}%)"
        )

        tech_result["potential"] = {
            "processed": proc_potential_stats,
            "raw": raw_potential_stats,
        }

        # p_nom_max stats
        proc_p_nom = profiles_ds["p_nom_max"].sel(technology=tech_key).values
        proc_p_nom_stats = _array_stats(proc_p_nom)

        log.info("p_nom_max stats")
        log.info(
            f"  processed: min={proc_p_nom_stats['min']:.6g}, max={proc_p_nom_stats['max']:.6g}, "
            f"mean={proc_p_nom_stats['mean']:.6g}, NaN={proc_p_nom_stats['nan_count']}/{proc_p_nom_stats['total']} "
            f"({proc_p_nom_stats['nan_share_pct']:.2f}%)"
        )

        p_nom_section = {"processed": proc_p_nom_stats}
        if "p_nom_max" in raw_ds.data_vars:
            raw_p_nom = raw_ds["p_nom_max"].values
            raw_p_nom_stats = _array_stats(raw_p_nom)
            log.info(
                f"  raw:       min={raw_p_nom_stats['min']:.6g}, max={raw_p_nom_stats['max']:.6g}, "
                f"mean={raw_p_nom_stats['mean']:.6g}, NaN={raw_p_nom_stats['nan_count']}/{raw_p_nom_stats['total']} "
                f"({raw_p_nom_stats['nan_share_pct']:.2f}%)"
            )
            p_nom_section["raw"] = raw_p_nom_stats
        else:
            log.info("  raw:       p_nom_max not present in raw dataset")

        proc_non_nan = int(np.sum(~np.isnan(proc_p_nom)))
        log.info(
            f"  processed non-NaN p_nom_max buses: {proc_non_nan}/{proc_p_nom.size}"
        )
        p_nom_section["processed_non_nan_buses"] = {
            "non_nan": proc_non_nan,
            "total": int(proc_p_nom.size),
        }

        tech_result["p_nom_max"] = p_nom_section
        audit[tech_key] = tech_result

    log.info("\n✓ Audit complete")
    return audit


# ============================================================================
# VISUALIZATION FUNCTIONS (Refactored from notebook)
# ============================================================================


def _get_natural_earth_countries():
    """Load and cache Natural Earth country shapes (expensive operation)."""
    global _NE_COUNTRIES_CACHE
    if _NE_COUNTRIES_CACHE is None:
        nat_earth_shp = shprdr.natural_earth(
            resolution="110m", category="cultural", name="admin_0_countries"
        )
        _NE_COUNTRIES_CACHE = list(shprdr.Reader(nat_earth_shp).records())
    return _NE_COUNTRIES_CACHE


def _parse_region_list(region, dataset=None, geometry_gdf=None):
    """
    Parse region input (str or list) into normalized list.
    Handles comma-separated strings and nested lists.
    If region is None and dataset is provided, returns all unique countries in dataset.

    Parameters
    ----------
    region : str, list, or None
        Region(s) to parse. If None and dataset is provided, uses all countries in dataset.
    dataset : xr.Dataset, optional
        Dataset to extract countries from if region is None.

    Returns
    -------
    list
        Normalized list of region codes.
    """
    # If region is None, extract all unique countries from dataset
    if region is None:
        if geometry_gdf is not None and "country" in geometry_gdf.columns:
            unique_countries = sorted(
                geometry_gdf["country"].dropna().astype(str).unique().tolist()
            )
            logger.info(
                "Region not specified. Using all countries from geometry_gdf: "
                f"{unique_countries}"
            )
            return unique_countries

        if dataset is not None and "country" in dataset:
            unique_countries = sorted(list(set(dataset["country"].values)))
            logger.info(
                f"Region not specified. Using all countries in dataset: {unique_countries}"
            )
            return unique_countries

        raise ValueError(
            "region is None but no country metadata is available. "
            "Pass region explicitly, or provide geometry_gdf with a 'country' column."
        )

    if isinstance(region, str):
        return [r.strip() for r in region.split(",") if r.strip()]

    region = [item for item in region if item is not None]
    expanded = []
    for item in region:
        if isinstance(item, str) and "," in item:
            expanded.extend([r.strip() for r in item.split(",") if r.strip()])
        else:
            expanded.append(item)
    return expanded


def _get_country_extent(regions):
    """
    Get map extent (bounds) for a list of country codes/names using Natural Earth data.

    Returns
    -------
    (min_lon, max_lon, min_lat, max_lat) : tuple
        Geographic bounds with buffer applied, ordered for cartopy.Axes.set_extent
    """
    countries = _get_natural_earth_countries()

    def normalize_label(label):
        return label.strip().upper() if isinstance(label, str) else ""

    def country_matches(label, country_record):
        label_upper = normalize_label(label)
        if not label_upper:
            return False
        return any(
            label_upper == str(country_record.attributes.get(attr, "")).upper()
            for attr in ("ISO_A2", "ISO_A3", "NAME_LONG", "NAME", "ABBREV")
        )

    matched_geometries = []
    unmatched_regions = []
    for r in regions:
        geom = next(
            (c.geometry.buffer(0) for c in countries if country_matches(r, c)),
            None,
        )
        if geom is None:
            unmatched_regions.append(r)
        else:
            matched_geometries.append(geom)

    shapes = gpd.GeoDataFrame(geometry=matched_geometries, crs="EPSG:4326")

    if unmatched_regions:
        logger.warning(
            f"No Natural Earth country match for: {unmatched_regions}. "
            "These labels are ignored for extent calculation."
        )

    if shapes.empty:
        logger.warning(
            f"No Natural Earth matches for regions {regions}. Using global extent."
        )
        return -180, -90, 180, 90

    # Natural Earth geometries are lon/lat; bounds can be taken directly in EPSG:4326.
    minx, miny, maxx, maxy = shapes.total_bounds
    buffer = MAP_STYLE["buffer"]
    extent = (minx - buffer, maxx + buffer, miny - buffer, maxy + buffer)
    logger.info(
        "Computed extent from %d matched countries [min_lon, max_lon, min_lat, max_lat]: "
        "[%.2f, %.2f, %.2f, %.2f]",
        len(shapes),
        extent[0],
        extent[1],
        extent[2],
        extent[3],
    )
    return extent


def _setup_map_features(ax, extent):
    """Apply standard map styling and background features."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ne_scale = MAP_STYLE["ne_scale"]

    # Background
    ax.add_feature(
        cfeature.OCEAN.with_scale(ne_scale),
        facecolor=MAP_STYLE["ocean_color"],
        zorder=0,
        alpha=0.3,
    )
    ax.add_feature(
        cfeature.LAND.with_scale(ne_scale),
        facecolor=MAP_STYLE["land_color"],
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale(ne_scale),
        linewidth=MAP_STYLE["coastline_width"],
        zorder=1,
        alpha=0.7,
        color=MAP_STYLE["coastline_color"],
    )

    # Borders on top
    ax.add_feature(
        cfeature.BORDERS.with_scale(ne_scale),
        linewidth=MAP_STYLE["border_width"],
        linestyle="-",
        zorder=10,
        alpha=0.7,
        color=MAP_STYLE["border_color"],
        edgecolor=MAP_STYLE["border_color"],
    )


def plot_grid_potentials(
    dataset,
    region=None,
    technology="onwind",
    geometry_gdf=None,
    figsize=(14, 11),
    projection=ccrs.PlateCarree(),
    cmap="Blues",
    title=None,
    filename=None,
    gridlabels=True,
):
    """
    Plot grid-level renewable potential as raster map.

    Parameters
    ----------
    dataset : xr.Dataset
        Output from build_profiles() or load_profiles()
    region : str, list, or None
        Region(s) to display: country codes (ISO2/ISO3) or names, e.g., "US,CA" or ["US", "CA"].
        If None, uses all countries present in dataset. Default: None
    technology : str
        Technology: "onwind", "offwind-ac", or "solar". Default: "onwind"
    geometry_gdf : GeoDataFrame, optional
        GeoDataFrame with bus geometries and country data (from build_profiles() or load_profiles()).
        Used to look up countries when region is None. Default: None
    figsize : tuple
        Figure size (width, height) in inches. Default: (14, 11)
    projection : cartopy CRS
        Map projection. Default: PlateCarree (lat/lon)
    cmap : str
        Matplotlib colormap name. Default: "Blues"
    title : str
        Plot title. If None, auto-generated. Default: None
    filename : str
        Save path if provided (e.g., "plot.png", "plot.pdf"). Default: None
    gridlabels : bool
        Show latitude/longitude gridlines. Default: True

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    regions = _parse_region_list(region, dataset=dataset, geometry_gdf=geometry_gdf)
    extent = _get_country_extent(regions)

    font_scale = figsize[0] / 10
    plt.rcParams.update({"font.size": 10 * font_scale})

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    # Setup map
    _setup_map_features(ax, extent)

    # Extract grid
    tech_idx = list(dataset.technology.values).index(technology)
    potential_gw = dataset["potential"].values[:, :, tech_idx] / 1e3  # MW → GW

    x_grid = dataset.coords["x_grid"].values
    y_grid = dataset.coords["y_grid"].values

    # Convert cell centers to edges for proper pcolormesh alignment
    # If spacing is uniform (from linspace), compute half-cell offsets
    dx = (x_grid[-1] - x_grid[0]) / (len(x_grid) - 1) if len(x_grid) > 1 else 0.25
    dy = (y_grid[-1] - y_grid[0]) / (len(y_grid) - 1) if len(y_grid) > 1 else 0.25

    # Create edge arrays: add half-cell boundaries
    x_edges = np.concatenate(
        [[x_grid[0] - dx / 2], (x_grid[:-1] + x_grid[1:]) / 2, [x_grid[-1] + dx / 2]]
    )
    y_edges = np.concatenate(
        [[y_grid[0] - dy / 2], (y_grid[:-1] + y_grid[1:]) / 2, [y_grid[-1] + dy / 2]]
    )

    X, Y = np.meshgrid(x_edges, y_edges)

    im = ax.pcolormesh(
        X,
        Y,
        potential_gw,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="flat",  # 'flat' works with edges
        zorder=2,
        alpha=0.85,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.08, aspect=25)
    cbar.set_label(
        "Renewable Potential (GW)", fontsize=11 * font_scale, fontweight="bold"
    )
    cbar.ax.tick_params(labelsize=9 * font_scale)

    # Labels and title
    if title is None:
        title = f"{technology.upper().replace('-', ' ')} - Grid-Level Potential"
    ax.set_title(title, fontsize=14 * font_scale, fontweight="bold", pad=20)
    ax.set_xlabel("Longitude (°E)", fontsize=10 * font_scale, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=10 * font_scale, fontweight="bold")

    # Gridlines
    if gridlabels:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color=MAP_STYLE["grid_color"],
            alpha=0.3,
            linestyle="--",
            zorder=2,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9 * font_scale, "color": "#555555"}
        gl.ylabel_style = {"size": 9 * font_scale, "color": "#555555"}

    # Border
    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_linewidth(1.5)
    ax.spines["geo"].set_edgecolor(MAP_STYLE["border_color"])

    # Save
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"✓ Saved plot to {filename}")

    return fig, ax


def plot_bus_capacity_density(
    dataset,
    region=None,
    technology="onwind",
    geometry_gdf=None,
    figsize=(14, 11),
    projection=ccrs.PlateCarree(),
    cmap="Blues",
    vmin=None,
    vmax=None,
    title=None,
    filename=None,
    gridlabels=True,
    edgecolor="black",
    linewidth=0.3,
):
    """
    Plot bus region geometries colored by installable capacity density.

    Each bus region (Voronoi polygon) is colored by its p_nom_max per unit area.
    Useful for comparing capacity potential across regions.

    Parameters
    ----------
    dataset : xr.Dataset
        Output from build_profiles() or load_profiles()
    region : str, list, or None
        Region(s) to display: country codes (ISO2/ISO3) or names.
        If None, uses all countries present in dataset. Default: None
    technology : str
        Technology: "onwind", "offwind-ac", or "solar". Default: "onwind"
    geometry_gdf : GeoDataFrame
        **REQUIRED**. GeoDataFrame with bus geometries, country, and area_km2 (from build_profiles() or load_profiles()).
        GIS metadata is now stored only in GeoJSON, not in the xarray dataset.
    figsize : tuple
        Figure size (width, height) in inches. Default: (14, 11)
    projection : cartopy CRS
        Map projection. Default: PlateCarree (lat/lon)
    cmap : str
        Matplotlib colormap. Default: "Blues"
    vmin, vmax : float, optional
        Color normalization limits (MW/km²). If None, uses 5-95 percentile for automatic scaling.
    title : str
        Plot title. If None, auto-generated. Default: None
    filename : str
        Save path if provided. Default: None
    gridlabels : bool
        Show latitude/longitude gridlines. Default: True
    edgecolor : str
        Color of region boundaries. Default: "black"
    linewidth : float
        Width of region edges. Default: 0.3

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects

    Examples
    --------
    # Load from saved files
    dataset, geometry_gdf = load_profiles("data.nc", "data.geojson")
    # Plot specific region
    fig, ax = plot_bus_capacity_density(dataset, "US", geometry_gdf=geometry_gdf)

    # Plot all countries in dataset
    fig, ax = plot_bus_capacity_density(dataset, geometry_gdf=geometry_gdf)
    """
    # geometry_gdf is now REQUIRED (GIS data no longer in xarray)
    if geometry_gdf is None:
        raise ValueError(
            "geometry_gdf parameter is required. GIS data (geometries, country, area) "
            "are now stored only in GeoJSON, not in the xarray dataset."
        )

    regions = _parse_region_list(region, dataset=dataset, geometry_gdf=geometry_gdf)

    # Convert ISO2 → ISO3 for dataset filtering
    region_iso3 = []
    for r in regions:
        r_upper = r.upper()
        if len(r_upper) == 2:
            try:
                iso3 = pycountry.countries.get(alpha_2=r_upper).alpha_3
                region_iso3.append(iso3)
            except AttributeError:
                region_iso3.append(r_upper)
        else:
            region_iso3.append(r_upper)

    extent = _get_country_extent(regions)

    font_scale = figsize[0] / 10
    plt.rcParams.update({"font.size": 10 * font_scale})

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    # Setup map
    _setup_map_features(ax, extent)

    # Extract technology index
    tech_idx = list(dataset.technology.values).index(technology)
    p_nom_max_data = dataset.isel(technology=tech_idx)["p_nom_max"].values
    bus_ids = dataset["bus"].values

    # Get country and area data from geometry_gdf
    geometry_gdf = geometry_gdf.copy()
    if "bus_id" not in geometry_gdf.columns:
        geometry_gdf["bus_id"] = geometry_gdf.get("name", geometry_gdf.index)

    countries_in_gdf = set(geometry_gdf.get("country", ["UNK"]).unique())
    matching_countries = [c for c in countries_in_gdf if c in region_iso3]

    if not matching_countries:
        raise ValueError(
            f"No data for region {region_iso3} and technology {technology}. "
            f"Available countries in GeoJSON: {sorted(countries_in_gdf)}"
        )

    # Filter geometry_gdf to matching countries
    geometry_gdf_filtered = geometry_gdf[
        geometry_gdf.get("country", "UNK").isin(matching_countries)
    ]

    # Extract geometries and calculate densities
    geometries = []
    densities = []

    logger.info(
        f"Loading {len(geometry_gdf_filtered)} bus geometries for {technology}..."
    )

    for bus_idx, (_, gdf_row) in enumerate(
        tqdm(
            geometry_gdf_filtered.iterrows(),
            total=len(geometry_gdf_filtered),
            desc="Processing geometries",
        )
    ):
        try:
            # Get bus ID to match with dataset
            bus_id = gdf_row.get(
                "bus_id",
                gdf_row.name if isinstance(gdf_row.name, str) else str(bus_idx),
            )

            # Find matching index in dataset
            dataset_idx = None
            for ds_idx, ds_bus_id in enumerate(bus_ids):
                if str(ds_bus_id) == str(bus_id):
                    dataset_idx = ds_idx
                    break

            if dataset_idx is None:
                continue

            p_nom_max = p_nom_max_data[dataset_idx]
            area_km2 = gdf_row.get("area_km2", 1.0)
            geom = gdf_row.geometry

            if not (
                np.isnan(p_nom_max)
                or np.isnan(area_km2)
                or p_nom_max <= 0
                or area_km2 <= 0
            ):
                geometries.append(geom)
                densities.append(p_nom_max / area_km2)  # MW/km²
        except Exception as e:
            logger.debug(f"Skipped bus {bus_idx}: {e}")

    if not geometries:
        raise ValueError(f"No valid geometries for {region} and {technology}")

    logger.info(
        f"Plotting {len(geometries)} regions, density range: "
        f"{np.min(densities):.3f} - {np.max(densities):.3f} MW/km²"
    )

    # Normalize color scale
    if vmin is None or vmax is None:
        vmin, vmax = np.nanpercentile(densities, [5, 95])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot geometries
    for geom, density in zip(geometries, densities):
        color = plt.cm.get_cmap(cmap)(norm(density))
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.85,
            zorder=5,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.08, aspect=25)
    cbar.set_label(
        "Capacity Density (MW/km²)", fontsize=11 * font_scale, fontweight="bold"
    )
    cbar.ax.tick_params(labelsize=9 * font_scale)

    # Labels and title
    if title is None:
        title = f"{technology.upper().replace('-', ' ')} - Installable Capacity Density"
    ax.set_title(title, fontsize=14 * font_scale, fontweight="bold", pad=20)
    ax.set_xlabel("Longitude (°E)", fontsize=10 * font_scale, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=10 * font_scale, fontweight="bold")

    # Gridlines
    if gridlabels:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color=MAP_STYLE["grid_color"],
            alpha=0.3,
            linestyle="--",
            zorder=2,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9 * font_scale, "color": "#555555"}
        gl.ylabel_style = {"size": 9 * font_scale, "color": "#555555"}

    # Border
    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_linewidth(1.5)
    ax.spines["geo"].set_edgecolor(MAP_STYLE["border_color"])

    # Save
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"✓ Saved plot to {filename}")

    return fig, ax


if __name__ == "__main__":
    # Placeholder for future CLI/Snakemake integration
    raise NotImplementedError(
        "CLI interface not yet implemented. Use as a module: "
        "from workflow.scripts.renewable_profiles import build_profiles"
    )
