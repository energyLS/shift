"""Consolidate regional renewable supply NetCDF files into one unified file.

Input: regional files from `data/new_renewables/supply_{Region}_2013_cleaned.nc`.
Output: `data/new_renewables_consolidated.nc` with dimensions `(region, site_id, time)`.

Usage:
    python workflow/scripts/preprocess_consolidate_renewables.py [--input-dir data/new_renewables] [--output data/new_renewables_consolidated.nc]
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import xarray as xr
import pandas as pd

from _helpers import setup_logging

logger = setup_logging(__name__, log_filename="preprocess_consolidate_renewables.log")


def extract_region_from_filename(filepath: Path) -> str:
    """Extract region name from supply_{Region}_2013_cleaned.nc filename."""
    name = filepath.stem  # Remove .nc extension
    # Format: supply_{Region}_2013_cleaned
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "supply":
        # Join all middle parts (handle multi-word regions like 'East_Asia')
        region = "_".join(
            parts[1:-2]
        )  # Exclude 'supply' prefix and '2013_cleaned' suffix
        return region
    raise ValueError(f"Could not extract region from filename: {filepath.name}")


def load_and_flatten_region(
    filepath: Path,
) -> Tuple[str, np.ndarray, np.ndarray, List[str], int]:
    """
    Load a regional NetCDF file, preserving technology dimension but flattening class.

    Returns:
        (region_name, capacity_by_tech, capacity_factor_by_tech, tech_names, n_classes)
        where capacity_by_tech is (n_tech, n_classes) and capacity_factor_by_tech is (n_tech, n_classes, n_time)
    """
    region = extract_region_from_filename(filepath)
    logger.info(f"Loading {region} from {filepath.name}")

    ds = xr.open_dataset(filepath)

    # Find capacity variable (2D: technology × class)
    cap_var = None
    for var_name in ds.data_vars:
        if "capacity" in var_name.lower() and ds[var_name].ndim == 2:
            cap_var = var_name
            break
    if cap_var is None:
        raise ValueError(
            f"Could not find 2D capacity variable in {filepath.name}. Available vars: {list(ds.data_vars)}"
        )

    # Find capacity_factor variable (3D: technology × class × time)
    cf_var = None
    for var_name in ds.data_vars:
        if any(
            x in var_name.lower()
            for x in ["capacity factor", "capacity_factor", "cf", "power", "profile"]
        ):
            if ds[var_name].ndim == 3:
                cf_var = var_name
                break
    if cf_var is None:
        raise ValueError(
            f"Could not find 3D time-series variable in {filepath.name}. Available vars: {list(ds.data_vars)}"
        )

    logger.info(f"  Capacity var: {cap_var}, Time-series var: {cf_var}")
    logger.info(f"  Capacity shape: {ds[cap_var].shape}, CF shape: {ds[cf_var].shape}")

    capacity = ds[cap_var]  # (technology, class)
    capacity_factor = ds[cf_var]  # (technology, class, time)

    # Identify dimension names
    tech_dim, class_dim, time_dim = None, None, None
    for dim in capacity.dims:
        if "tech" in dim.lower():
            tech_dim = dim
        if "class" in dim.lower():
            class_dim = dim

    for dim in capacity_factor.dims:
        if "tech" in dim.lower():
            tech_dim = dim
        if "class" in dim.lower():
            class_dim = dim
        if "time" in dim.lower():
            time_dim = dim

    if tech_dim is None or class_dim is None:
        raise ValueError(
            f"Could not identify technology/class dimensions. Dims: {capacity.dims}"
        )

    # Get technology names from coordinate
    tech_names = ds.coords[tech_dim].values.tolist()
    n_classes = ds.sizes[class_dim]
    n_time = ds.sizes[time_dim]

    logger.info(f"  Technologies: {tech_names}, Classes: {n_classes}, Time: {n_time}")

    # Keep technology dimension intact, just extract data
    cap_array = capacity.values  # (tech, class)
    cf_array = capacity_factor.values  # (tech, class, time)

    # Ensure time is last dimension
    if capacity_factor.dims.index(time_dim) != 2:
        cf_array = np.moveaxis(cf_array, capacity_factor.dims.index(time_dim), -1)

    ds.close()
    return region, cap_array, cf_array, tech_names, n_classes


def consolidate_renewables(input_dir: Path, output_path: Path) -> None:
    """
    Consolidate 15 regional NetCDF files into one unified file, preserving technology dimension.

    Output structure:
        - Dimensions: region (15), technology, class, time (8760)
        - Variables: capacity (region, technology, class), capacity_factor (region, technology, class, time)

    This preserves the technology distinction (wind, solar, etc.) so PyPSA can create separate
    generators per technology and region.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Find all regional files
    nc_files = sorted(input_dir.glob("supply_*_2013_cleaned.nc"))
    logger.info(f"Found {len(nc_files)} regional files")

    if len(nc_files) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")

    # Load all regions (preserve technology and class dimensions)
    regions_data = []
    tech_names = None
    n_classes = None
    time_length = None

    for filepath in nc_files:
        region, cap_array, cf_array, file_tech_names, file_n_classes = (
            load_and_flatten_region(filepath)
        )

        # Verify consistency
        if tech_names is None:
            tech_names = file_tech_names
        elif tech_names != file_tech_names:
            raise ValueError(
                f"Inconsistent technologies: {region} has {file_tech_names}, expected {tech_names}"
            )

        if n_classes is None:
            n_classes = file_n_classes
        elif n_classes != file_n_classes:
            raise ValueError(
                f"Inconsistent class count: {region} has {file_n_classes}, expected {n_classes}"
            )

        if time_length is None:
            time_length = (
                cap_array.shape[-1] if cap_array.ndim == 3 else cf_array.shape[-1]
            )
        elif cf_array.shape[-1] != time_length:
            raise ValueError(
                f"Inconsistent time dimensions: {region} has {cf_array.shape[-1]}, expected {time_length}"
            )

        regions_data.append(
            {
                "region": region,
                "capacity": cap_array,  # (tech, class)
                "capacity_factor": cf_array,  # (tech, class, time)
            }
        )

    logger.info(
        f"Tech names: {tech_names}, Classes: {n_classes}, Time length: {time_length}"
    )

    # Stack all regions into (region, tech, class, time) structure
    region_names = [r["region"] for r in regions_data]

    # Preallocate arrays
    capacity_stacked = np.zeros(
        (len(regions_data), len(tech_names), n_classes), dtype=np.float32
    )
    cf_stacked = np.zeros(
        (len(regions_data), len(tech_names), n_classes, time_length), dtype=np.float32
    )

    for i, data in enumerate(regions_data):
        capacity_stacked[i, :, :] = data["capacity"]
        cf_stacked[i, :, :, :] = data["capacity_factor"]

    logger.info(
        f"Created stacked arrays: capacity {capacity_stacked.shape}, cf {cf_stacked.shape}"
    )

    # Create consolidated xarray Dataset with technology dimension preserved
    time_index = pd.date_range("2013-01-01", periods=time_length, freq="h")
    class_ids = np.arange(n_classes)

    ds_consolidated = xr.Dataset(
        data_vars={
            "capacity": (["region", "technology", "class"], capacity_stacked),
            "capacity_factor": (["region", "technology", "class", "time"], cf_stacked),
        },
        coords={
            "region": region_names,
            "technology": tech_names,
            "class": class_ids,
            "time": time_index,
        },
        attrs={
            "description": "Consolidated renewable supply profiles for 15 global regions, with technology distinction",
            "source": "data/new_renewables/*.nc",
            "temporal_resolution": "hourly",
            "year": 2013,
            "technologies": ", ".join(tech_names),
        },
    )

    # Add variable attributes
    ds_consolidated["capacity"].attrs = {
        "long_name": "Installed capacity",
        "units": "MW",
    }
    ds_consolidated["capacity_factor"].attrs = {
        "long_name": "Capacity factor (power output / installed capacity)",
        "units": "p.u.",
    }

    # Write to NetCDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing consolidated file to {output_path}")
    ds_consolidated.to_netcdf(
        output_path,
        encoding={
            "capacity": {"dtype": "float32", "zlib": True, "complevel": 4},
            "capacity_factor": {"dtype": "float32", "zlib": True, "complevel": 4},
        },
    )

    logger.info(f"✓ Consolidation complete: {output_path}")
    logger.info(f"  Regions: {len(region_names)}")
    logger.info(f"  Technologies: {tech_names}")
    logger.info(f"  Classes per region: {n_classes}")
    logger.info(f"  Timesteps: {time_length}")
    logger.info(f"  Output size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/new_renewables"),
        help="Directory containing supply_*.nc files (default: data/new_renewables)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/new_renewables_consolidated.nc"),
        help="Output path for consolidated file (default: data/new_renewables_consolidated.nc)",
    )

    args = parser.parse_args()

    consolidate_renewables(args.input_dir, args.output)
