import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Colourblind-safe cluster palette (reusable)
PLOTLY_CLUSTER_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
]


def _map_trace_tech(source_tech: str) -> str:
    """Map legacy TRACE technology names to our canonical names."""
    return {"windonshore": "onwind", "pvplant": "solar"}.get(source_tech, source_tech)


def build_timeseries_df(ds, source_name: str, region_name: str, ts_tech_order=None):
    """Build a pandas DataFrame of cluster timeseries from an xarray dataset.

    Returns columns: source, technology, cluster_id, avg_cf, capacity_mw, timeseries (numpy array)
    """
    if ts_tech_order is None:
        ts_tech_order = ["solar", "onwind"]
    rows = []
    if ds is None:
        return pd.DataFrame(rows)

    if source_name == "TRACE":
        tech_map = {"windonshore": "onwind", "pvplant": "solar"}
        for source_tech, compare_tech in tech_map.items():
            if "technology" not in ds.coords:
                continue
            if source_tech not in [str(v) for v in ds.technology.values]:
                continue
            try:
                cap_da = ds["capacity"].sel(region=region_name, technology=source_tech)
                cf_da = ds["capacity_factor"].sel(
                    region=region_name, technology=source_tech
                )
                if "avg_cf" in ds.data_vars:
                    avg_cf_da = ds["avg_cf"].sel(
                        region=region_name, technology=source_tech
                    )
                else:
                    avg_cf_da = cf_da.mean(dim="time", skipna=True)
                for idx in range(len(cap_da.values)):
                    capacity_mw = float(cap_da.values[idx])
                    avg_cf = float(avg_cf_da.values[idx])
                    cf_ts = np.array(cf_da.values[idx], dtype=float)
                    if (
                        np.isnan(capacity_mw)
                        or np.isnan(avg_cf)
                        or np.all(np.isnan(cf_ts))
                    ):
                        continue
                    rows.append(
                        {
                            "source": source_name,
                            "technology": compare_tech,
                            "cluster_id": int(idx),
                            "avg_cf": avg_cf,
                            "capacity_mw": capacity_mw,
                            "timeseries": np.nan_to_num(cf_ts, nan=0.0),
                        }
                    )
            except Exception:
                continue
    else:
        for tech_name in ts_tech_order:
            if tech_name not in [str(v) for v in ds.technology.values]:
                continue
            try:
                cap_da = ds["capacity"].sel(region=region_name, technology=tech_name)
                cf_da = ds["capacity_factor"].sel(
                    region=region_name, technology=tech_name
                )
                avg_cf_da = ds["avg_cf"].sel(region=region_name, technology=tech_name)
                for idx in range(len(cap_da.values)):
                    capacity_mw = float(cap_da.values[idx])
                    avg_cf = float(avg_cf_da.values[idx])
                    cf_ts = np.array(cf_da.values[idx], dtype=float)
                    if (
                        np.isnan(capacity_mw)
                        or np.isnan(avg_cf)
                        or np.all(np.isnan(cf_ts))
                    ):
                        continue
                    rows.append(
                        {
                            "source": source_name,
                            "technology": tech_name,
                            "cluster_id": int(idx),
                            "avg_cf": avg_cf,
                            "capacity_mw": capacity_mw,
                            "timeseries": np.nan_to_num(cf_ts, nan=0.0),
                        }
                    )
            except Exception:
                continue
    return pd.DataFrame(rows)


def _cf_to_width(cf, cf_min, cf_span):
    norm = (cf - cf_min) / cf_span if cf_span > 0 else 0.0
    return float(np.clip(1.0 + 5.0 * (norm**1.6), 1.0, 6.0))


def plot_region_cluster_timeseries(
    region_name: str,
    technology_name: str,
    trace_ds=None,
    pypsa_ds=None,
    tech_colors=None,
    cluster_colors=None,
    show=True,
):
    """Create a two-panel Plotly figure comparing TRACE vs PyPSA-Earth for a region+technology.

    - `trace_ds` and `pypsa_ds` are xarray datasets (or None).
    - `tech_colors` maps technologies to base hex colors (optional).
    - `cluster_colors` is a list of colours to cycle for clusters.
    Returns the Plotly `Figure`.
    """
    if cluster_colors is None:
        cluster_colors = PLOTLY_CLUSTER_COLORS
    if tech_colors is None:
        tech_colors = {"onwind": "#1F77B4", "solar": "#D62728"}

    trace_df = build_timeseries_df(trace_ds, "TRACE", region_name)
    pypsa_df = build_timeseries_df(pypsa_ds, "PyPSA-Earth", region_name)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f"TRACE - {region_name}", f"PyPSA-Earth - {region_name}"),
    )

    for row_idx, df in enumerate([trace_df, pypsa_df], start=1):
        source_name = "TRACE" if row_idx == 1 else "PyPSA-Earth"
        if df.empty:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=(f"x{row_idx} domain" if row_idx > 1 else "x domain"),
                yref=(f"y{row_idx} domain" if row_idx > 1 else "y domain"),
                text=f"No data for {source_name}",
                showarrow=False,
            )
            continue
        tech_df = df[df["technology"] == technology_name].copy()
        if tech_df.empty:
            continue
        cf_min = float(tech_df["avg_cf"].min())
        cf_span = (
            float(tech_df["avg_cf"].max() - cf_min)
            if float(tech_df["avg_cf"].max()) > cf_min
            else 1.0
        )
        legend_names = set()
        for _, row in tech_df.sort_values(by="avg_cf").iterrows():
            cid = int(row["cluster_id"])
            avg_cf = float(row["avg_cf"])
            cap_gw = float(row["capacity_mw"]) / 1e3
            color = cluster_colors[cid % len(cluster_colors)]
            width = _cf_to_width(avg_cf, cf_min, cf_span)
            cf_ts = np.asarray(row["timeseries"], dtype=float)
            hours = np.arange(len(cf_ts))
            customdata = np.column_stack([np.full(len(cf_ts), cid)])
            legend_label = f" {source_name} - cluster {cid} | {cap_gw:.2f} GW | avg cf {avg_cf:.3f}"
            showleg = legend_label not in legend_names
            if showleg:
                legend_names.add(legend_label)
            fig.add_trace(
                go.Scattergl(
                    x=hours,
                    y=cf_ts,
                    mode="lines",
                    line=dict(color=color, width=width),
                    opacity=0.9,
                    customdata=customdata,
                    hovertemplate="Cluster %{customdata[0]}<br>Hour %{x}<br>CF %{y:.3f}<extra></extra>",
                    name=legend_label,
                    legendgroup=f"{technology_name}-{cid}",
                    showlegend=showleg,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        template="plotly_white",
        height=760,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Hour of year", row=2, col=1)
    fig.update_yaxes(title_text="Capacity factor", row=1, col=1)
    fig.update_yaxes(title_text="Capacity factor", row=2, col=1)
    if show:
        fig.show()
    return fig


def plot_bubble_region_comparison(
    region_name: str, trace_ds, pypsa_ds, tech_colors=None, save_path=None, show=True
):
    """Create the bubble plots comparing TRACE vs PyPSA-Earth for a single region.

    This mirrors the notebook bubble plot: x=avg_cf, y=capacity_mw, bubble size ~ generation potential.
    """
    if tech_colors is None:
        tech_colors = {"onwind": "#1F77B4", "solar": "#D62728"}

    if trace_ds is None or pypsa_ds is None:
        raise ValueError(
            "Both TRACE and PyPSA-Earth datasets are required for the bubble plots."
        )

    # build comparison_df in the same format as the notebook's `comparison_df`
    trace_rows = []
    if trace_ds is not None:
        for trace_tech_name, compare_tech_name in {
            "windonshore": "onwind",
            "pvplant": "solar",
        }.items():
            if trace_tech_name not in [str(v) for v in trace_ds.technology.values]:
                continue
            try:
                cap_da = trace_ds["capacity"].sel(technology=trace_tech_name)
                if "avg_cf" in trace_ds.data_vars:
                    avg_cf_da = trace_ds["avg_cf"].sel(technology=trace_tech_name)
                else:
                    avg_cf_da = (
                        trace_ds["capacity_factor"]
                        .sel(technology=trace_tech_name)
                        .mean(dim="time", skipna=True)
                    )
                for region in trace_ds.region.values:
                    if str(region) != str(region_name):
                        continue
                    region_caps = cap_da.sel(region=region).values
                    region_avg_cf = avg_cf_da.sel(region=region).values
                    for class_idx in range(len(region_caps)):
                        capacity_mw = float(region_caps[class_idx])
                        avg_cf = float(region_avg_cf[class_idx])
                        if np.isnan(capacity_mw) or np.isnan(avg_cf):
                            continue
                        trace_rows.append(
                            {
                                "source": "TRACE",
                                "region": str(region),
                                "technology": compare_tech_name,
                                "cluster_id": int(class_idx),
                                "capacity_mw": capacity_mw,
                                "avg_cf": avg_cf,
                                "generation_potential_mwh": capacity_mw
                                * avg_cf
                                * 8760.0,
                            }
                        )
            except Exception:
                continue

    pypsa_rows = []
    if pypsa_ds is not None:
        for tech_name in ["onwind", "solar"]:
            if tech_name not in [str(v) for v in pypsa_ds.technology.values]:
                continue
            try:
                cap_da = pypsa_ds["capacity"].sel(technology=tech_name)
                avg_cf_da = pypsa_ds["avg_cf"].sel(technology=tech_name)
                for region in pypsa_ds.region.values:
                    if str(region) != str(region_name):
                        continue
                    region_caps = cap_da.sel(region=region).values
                    region_avg_cf = avg_cf_da.sel(region=region).values
                    for class_idx in range(len(region_caps)):
                        capacity_mw = float(region_caps[class_idx])
                        avg_cf = float(region_avg_cf[class_idx])
                        if np.isnan(capacity_mw) or np.isnan(avg_cf):
                            continue
                        pypsa_rows.append(
                            {
                                "source": "PyPSA-Earth",
                                "region": str(region),
                                "technology": tech_name,
                                "cluster_id": int(class_idx),
                                "capacity_mw": capacity_mw,
                                "avg_cf": avg_cf,
                                "generation_potential_mwh": capacity_mw
                                * avg_cf
                                * 8760.0,
                            }
                        )
            except Exception:
                continue

    bubble_df = pd.concat(
        [pd.DataFrame(trace_rows), pd.DataFrame(pypsa_rows)], ignore_index=True
    )
    if bubble_df.empty:
        raise ValueError(f"No comparison data available for {region_name}.")

    if "generation_potential_mwh" not in bubble_df.columns:
        bubble_df["generation_potential_mwh"] = (
            bubble_df["capacity_mw"] * bubble_df["avg_cf"] * 8760.0
        )

    max_potential = bubble_df["generation_potential_mwh"].max()
    size_scale = 1500.0 / max_potential if max_potential > 0 else 1.0

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharex=True, sharey=True, constrained_layout=True
    )
    bubble_source_order = ["TRACE", "PyPSA-Earth"]
    bubble_panel_labels = {"TRACE": "TRACE panel", "PyPSA-Earth": "PyPSA-Earth panel"}

    for ax, source_name in zip(axes, bubble_source_order):
        subset = bubble_df[bubble_df["source"] == source_name].copy()
        if subset.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for {source_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        for tech_name in ["solar", "onwind"]:
            tech_subset = subset[subset["technology"] == tech_name]
            if tech_subset.empty:
                continue
            sizes = np.clip(
                tech_subset["generation_potential_mwh"].to_numpy() * size_scale,
                20,
                1600,
            )
            ax.scatter(
                tech_subset["avg_cf"],
                tech_subset["capacity_mw"],
                s=sizes,
                alpha=0.45,
                color=tech_colors.get(tech_name, "#888888"),
                edgecolor="white",
                linewidth=0.6,
                label=tech_name,
            )

        ax.set_title(
            f"{source_name} - {region_name}", fontsize=13, fontweight="semibold"
        )
        ax.set_xlabel("Average capacity factor", fontsize=11, labelpad=8)
        ax.set_ylabel("Capacity (MW)", fontsize=11, labelpad=8)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        tech_legend = ax.legend(title="Technology", loc="upper right", frameon=True)
        ax.add_artist(tech_legend)
        ax.text(
            0.02,
            0.97,
            bubble_panel_labels[source_name],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="semibold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )

    fig.suptitle(
        "Bubble plot of capacity factor vs capacity", fontsize=16, fontweight="semibold"
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def build_cluster_comparison_df(ds_consolidated, ds_clustered, df_cluster_meta=None):
    """Build cluster_comparison_df used by the stacked-region plotting.

    Returns a DataFrame with columns: source, region, technology, cluster_id, cluster_label, capacity_mw, avg_cf
    """
    rows = []
    techs_to_compare = ["onwind", "solar"]
    consolidated_map = {"windonshore": "onwind", "pvplant": "solar"}

    if ds_consolidated is not None:
        for consolidated_tech_name, compare_tech_name in consolidated_map.items():
            try:
                cap_da = ds_consolidated["capacity"].sel(
                    technology=consolidated_tech_name
                )
                cf_da = ds_consolidated["capacity_factor"].sel(
                    technology=consolidated_tech_name
                )
                avg_cf_da = cf_da.mean(dim="time", skipna=True)
                for region_name in ds_consolidated.region.values:
                    region_caps = cap_da.sel(region=region_name).values
                    region_avg_cf = avg_cf_da.sel(region=region_name).values
                    for class_idx in range(len(region_caps)):
                        capacity_mw = float(region_caps[class_idx])
                        avg_cf = float(region_avg_cf[class_idx])
                        if np.isnan(capacity_mw) or np.isnan(avg_cf):
                            continue
                        rows.append(
                            {
                                "source": "Consolidated",
                                "region": str(region_name),
                                "technology": str(compare_tech_name),
                                "cluster_id": int(class_idx),
                                "cluster_label": f"{compare_tech_name} {class_idx + 1}",
                                "capacity_mw": capacity_mw,
                                "avg_cf": avg_cf,
                            }
                        )
            except Exception:
                continue

    if df_cluster_meta is not None and not df_cluster_meta.empty:
        subset = df_cluster_meta[
            df_cluster_meta["technology"].isin(techs_to_compare)
        ].copy()
        for _, row in subset.iterrows():
            rows.append(
                {
                    "source": "Clustered",
                    "region": str(row["region"]),
                    "technology": str(row["technology"]),
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_label": f"{row['technology']} {int(row['cluster_id']) + 1}",
                    "capacity_mw": float(row["total_capacity_mw"]),
                    "avg_cf": float(row["avg_cf"]),
                }
            )
    elif ds_clustered is not None:
        for tech_name in techs_to_compare:
            try:
                cap_da = ds_clustered["capacity"].sel(technology=tech_name)
                avg_cf_da = ds_clustered["avg_cf"].sel(technology=tech_name)
                for region_name in ds_clustered.region.values:
                    region_caps = cap_da.sel(region=region_name).values
                    region_avg_cf = avg_cf_da.sel(region=region_name).values
                    for class_idx in range(len(region_caps)):
                        capacity_mw = float(region_caps[class_idx])
                        avg_cf = float(region_avg_cf[class_idx])
                        if np.isnan(capacity_mw) or np.isnan(avg_cf):
                            continue
                        rows.append(
                            {
                                "source": "Clustered",
                                "region": str(region_name),
                                "technology": str(tech_name),
                                "cluster_id": int(class_idx),
                                "cluster_label": f"{tech_name} {class_idx + 1}",
                                "capacity_mw": capacity_mw,
                                "avg_cf": avg_cf,
                            }
                        )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["source"] = pd.Categorical(
        df["source"], categories=["Consolidated", "Clustered"], ordered=True
    )
    df["technology"] = pd.Categorical(
        df["technology"], categories=["solar", "onwind"], ordered=True
    )
    df["region"] = df["region"].astype(str)
    df["avg_cf"] = df["avg_cf"].astype(float)
    return df


def plot_stacked_region_comparison(
    cluster_comparison_df, tech_palette=None, save_path=None, show=True
):
    """Draw the stacked region comparison (matplotlib) used in the notebook.

    `cluster_comparison_df` should be the output of `build_cluster_comparison_df`.
    """
    if cluster_comparison_df is None or cluster_comparison_df.empty:
        raise ValueError("cluster_comparison_df is required and must not be empty")
    if tech_palette is None:
        tech_palette = {"solar": "#C55C5C", "onwind": "#4C78A8"}

    sns.set_style("whitegrid")
    source_offsets = {"Consolidated": -0.18, "Clustered": 0.18}
    tech_width = 0.24
    tech_order = ["solar", "onwind"]
    source_order = ["Consolidated", "Clustered"]

    def blend_with_white(color, intensity):
        base = np.array(sns.color_palette([color])[0])
        intensity = float(np.clip(intensity, 0.0, 1.0))
        return tuple(base * intensity + np.array([1.0, 1.0, 1.0]) * (1.0 - intensity))

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

    for ax, tech_name in zip(axes, tech_order):
        df = cluster_comparison_df[
            cluster_comparison_df["technology"] == tech_name
        ].copy()
        regions = sorted(df["region"].unique())
        x_positions = np.arange(len(regions)) * 1.35

        for region_idx, region_name in enumerate(regions):
            for source_name in source_order:
                subset = df[
                    (
                        (df["region"] == region_name)
                        & (df["technology"] == tech_name)
                        & (df["source"] == source_name)
                    )
                ].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values(by="avg_cf")
                bottom = 0.0
                cf_min = float(subset["avg_cf"].min())
                cf_max = float(subset["avg_cf"].max())
                cf_span = cf_max - cf_min if cf_max > cf_min else 1.0
                border_color = "black" if source_name == "Consolidated" else "white"
                border_width = 0.55 if source_name == "Consolidated" else 0.3

                for _, row in subset.iterrows():
                    norm_cf = (row["avg_cf"] - cf_min) / cf_span
                    shade = 0.10 + 0.90 * (norm_cf**0.65)
                    color = blend_with_white(tech_palette[tech_name], shade)
                    ax.bar(
                        x_positions[region_idx] + source_offsets[source_name],
                        row["capacity_mw"],
                        width=tech_width,
                        bottom=bottom,
                        color=color,
                        edgecolor=border_color,
                        linewidth=border_width,
                    )
                    bottom += row["capacity_mw"]

        import matplotlib.patches as mpatches

        source_handles = [
            mpatches.Patch(
                facecolor="#E0E0E0", edgecolor="black", label="Consolidated"
            ),
            mpatches.Patch(facecolor="#A9A9A9", edgecolor="white", label="Clustered"),
        ]
        cf_handles = [
            mpatches.Patch(
                facecolor="#f2f2f2", edgecolor="#cccccc", label="low avg_cf"
            ),
            mpatches.Patch(
                facecolor="#5a5a5a", edgecolor="#cccccc", label="high avg_cf"
            ),
        ]
        tech_handles = [mpatches.Patch(color=tech_palette[tech_name], label=tech_name)]

        source_legend = ax.legend(
            handles=source_handles, title="Source", loc="upper left"
        )
        ax.add_artist(source_legend)
        cf_legend = ax.legend(handles=cf_handles, title="CF shade", loc="upper right")
        ax.add_artist(cf_legend)
        ax.legend(handles=tech_handles, title="Technology", loc="center right")

        ax.set_title(
            f"{tech_name.capitalize()}: consolidated vs clustered by region",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Region", fontsize=11)
        ax.set_ylabel("Capacity (MW)", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(regions, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(x=0.04)

    fig.suptitle(
        "Region-wise stacked capacity comparison", fontsize=15, fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
