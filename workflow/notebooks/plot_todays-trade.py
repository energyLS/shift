"""
Today's steel trade:

Data: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37  
-> https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS22_V202601.zip (2022-2024) - Latest update from BACI.  
-> workflow/notebooks/BACI_HS22_V202601

Process (in general): 
Iron ore -> DRI/HBI -> steel (raw)

Relevant HS product codes for iron ore trade:
- 260111,Iron ores and concentrates: non-agglomerated
- 260112,Iron ores and concentrates: agglomerated (excluding roasted iron pyrites)
- Source: product_codes_HS22_V202601.csv

Relevant HS product codes for DRI/HBI trade:
- 720310,"Ferrous products: obtained by direct reduction of iron ore, in lumps, pellets or similar forms"
- 720390,"Ferrous products: spongy ferrous products and iron having a minimum purity by weight of 99.94%, in lumps, pellets or similar forms"
- Source: product_codes_HS22_V202601.csv

Relevant HS product codes for steel trade (raw):
- 720711,"Iron or non-alloy steel: semi-finished products of iron or non-alloy steel: containing by weight less than 0.25% of carbon, of rectangular (including square) cross-section, width less than twice thickness"
- 720712,"Iron or non-alloy steel: semi-finished products of iron or non-alloy steel: containing by weight less than 0.25% of carbon, of rectangular (other than square) cross-section"
- 720719,"Iron or non-alloy steel: semi-finished products of iron or non-alloy steel, containing by weight less than 0.25% of carbon, other than rectangular or square cross-section"
- 720720,"Iron or non-alloy steel: semi-finished products of iron or non-alloy steel, containing by weight 0.25% or more of carbon"
- Source: product_codes_HS22_V202601.csv
"""


import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

warnings.filterwarnings("ignore")
plt.style.use("bmh")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Resolved at runtime from snakemake (see __main__ block below)
PRODUCT_OPTIONS = {
    "Iron Ore":    [260111, 260112],
    "DRI-HBI":     [720310, 720390],
    # Semi-finished (7207) + flat-rolled (7208-7212) + long products (7213-7217)
    # + stainless (7218-7223) + other alloy steel (7224-7229).
    # Excludes pig iron/ferro-alloys/scrap (7201-7205) and DRI/HBI (7203, already above).
    "Steel (raw)": [
        # Semi-finished billets / slabs
        720711, 720712, 720719, 720720,
        # Hot-rolled flat, wide strip
        720810, 720825, 720826, 720827, 720836, 720837, 720838, 720839,
        720840, 720851, 720852, 720853, 720854, 720890,
        # Cold-rolled flat, wide strip
        720915, 720916, 720917, 720918, 720925, 720926, 720927, 720928, 720990,
        # Coated / painted flat, wide strip
        721011, 721012, 721020, 721030, 721041, 721049, 721050,
        721061, 721069, 721070, 721090,
        # Flat-rolled, narrow strip (< 600 mm)
        721113, 721114, 721119, 721123, 721129, 721190,
        721210, 721220, 721230, 721240,
        # Wire rod
        721310, 721320, 721391, 721399,
        # Bars and rods
        721410, 721420, 721430, 721491, 721499,
        721510, 721550, 721590,
        # Sections (angles, beams, channels)
        721610, 721621, 721622, 721631, 721632, 721633,
        721640, 721650, 721661, 721669, 721691, 721699,
        # Wire
        721710, 721720, 721730, 721790,
        # Stainless steel (flat, long, wire)
        721810, 721891, 721899,
        721911, 721912, 721913, 721914, 721921, 721922, 721923, 721924,
        721931, 721932, 721933, 721934, 721935,
        722011, 722012, 722020, 722090,
        722110, 722190, 722210, 722220, 722230, 722240, 722290,
        722300,
        # Other alloy steel (flat, long, wire)
        722410, 722490,
        722511, 722519, 722520, 722530, 722540, 722550,
        722611, 722619, 722620, 722691, 722699,
        722710, 722720, 722790,
        722810, 722820, 722830, 722840, 722850, 722860, 722870, 722880, 722890,
        722910, 722920, 722990,
    ],
}

# Safe filename stems matching Snakemake output keys
PRODUCT_SAFE_NAMES = {
    "Iron Ore":    "Iron_Ore",
    "DRI-HBI":     "DRI-HBI",
    "Steel (raw)": "Steel_raw",
}

# How many bilateral net-flow arrows to draw on the map.
# Larger values add more (smaller) trade arrows; 300 captures most meaningful flows.
TOP_N_FLOWS = 300

# Which column to size bubbles and lines by.
# "quantity"    → physical volume (metric tonnes) — best for comparing trade magnitude.
# "trade_value" → USD value — useful if price differences between products matter.
VALUE_COLUMN = "quantity"



# ==============================================================================
# 1. PREPARE TRADE DATA
# ==============================================================================
def prepare_trade(baci_folder, product_name, product_codes, out_csv, year):
    country_trade_path = os.path.join(baci_folder, f"BACI_HS22_Y{year}_V202601.csv")
    country_codes_path = os.path.join(baci_folder, "country_codes_V202601.csv")

    print(f"Reading {country_trade_path}")
    # i: exporter, j: importer, k: product, v: value, q: quantity
    df = pd.read_csv(country_trade_path, usecols=["i", "j", "k", "v", "q"])

    print(f"Filtering {product_name} products: {product_codes}")
    df = df[df["k"].isin(product_codes)].copy()

    print("Aggregating bilateral trade")
    trade_pair = (
        df
        .groupby(["i", "j"], as_index=False)
        .agg(trade_value=("v", "sum"), quantity=("q", "sum"))
        .rename(columns={"i": "exporter", "j": "importer"})
    )

    print("Loading country codes")
    cc = pd.read_csv(country_codes_path)
    cc["country_code"] = cc["country_code"].astype(int)

    print("Mapping ISO3 codes")
    trade_iso3 = trade_pair.merge(
        cc[["country_code", "country_iso3"]].rename(
            columns={"country_code": "exporter", "country_iso3": "exporter_iso3"}
        ),
        on="exporter", how="left"
    ).merge(
        cc[["country_code", "country_iso3"]].rename(
            columns={"country_code": "importer", "country_iso3": "importer_iso3"}
        ),
        on="importer", how="left"
    )

    out_iso3 = out_csv
    os.makedirs(os.path.dirname(out_iso3), exist_ok=True)
    trade_iso3.to_csv(out_iso3, index=False)

    print(f"Saved processed trade file: {out_iso3}")
    return out_iso3


# ==============================================================================
# 2. COMPUTE NET PAIRWISE FLOWS
# ==============================================================================
def get_net_pairwise_flows(df_trade, exp_col, imp_col, val_col):
    pairs = df_trade.groupby([exp_col, imp_col])[val_col].sum().reset_index()
    pairs.columns = ["node1", "node2", "val"]

    pairs["pair_key"] = pairs.apply(
        lambda x: "-".join(sorted([str(x["node1"]), str(x["node2"])])),
        axis=1
    )

    net_flows = []
    for _, group in pairs.groupby("pair_key"):
        if len(group) == 1:
            row = group.iloc[0]
            net_flows.append({
                "exporter": row["node1"],
                "importer": row["node2"],
                "quantity": row["val"]
            })
        else:
            row_a = group.iloc[0]
            row_b = group.iloc[1]
            diff = row_a["val"] - row_b["val"]

            if diff > 0:
                net_flows.append({
                    "exporter": row_a["node1"],
                    "importer": row_a["node2"],
                    "quantity": diff
                })
            elif diff < 0:
                net_flows.append({
                    "exporter": row_b["node1"],
                    "importer": row_b["node2"],
                    "quantity": abs(diff)
                })

    return pd.DataFrame(net_flows)


# ==============================================================================
# 3. PROCESS DATA FOR PLOTTING
# ==============================================================================
def process_data_for_plot(trade_file, world_gdf, value_column):
    print(f"Reading processed trade file: {trade_file}")
    df_trade = pd.read_csv(trade_file)

    exp_col = "exporter_iso3"
    imp_col = "importer_iso3"
    val_col = value_column

    total_exp = (
        df_trade.groupby(exp_col)[val_col]
        .sum()
        .reset_index()
        .rename(columns={exp_col: "iso3", val_col: "total_exp"})
    )

    total_imp = (
        df_trade.groupby(imp_col)[val_col]
        .sum()
        .reset_index()
        .rename(columns={imp_col: "iso3", val_col: "total_imp"})
    )

    net_pairwise = get_net_pairwise_flows(df_trade, exp_col, imp_col, val_col)
    top_net_flows = net_pairwise.sort_values("quantity", ascending=False).head(TOP_N_FLOWS)

    temp_world = world_gdf.copy()
    temp_world = temp_world.merge(total_exp, on="iso3", how="left")
    temp_world = temp_world.merge(total_imp, on="iso3", how="left")
    temp_world[["total_exp", "total_imp"]] = temp_world[["total_exp", "total_imp"]].fillna(0)
    temp_world["net_export"] = temp_world["total_exp"] - temp_world["total_imp"]
    temp_world["centroid"] = temp_world.geometry.centroid

    return temp_world, top_net_flows


# ==============================================================================
# 4. PLOT
# ==============================================================================
def plot(data_plot, year, product_name, value_column, colors, output_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    world_df, flow_df = data_plot
    world_df.plot(ax=ax, color=colors["land"], edgecolor=colors["border"], linewidth=0.4)

    global_max_bubble = world_df["net_export"].abs().max()
    global_max_line = flow_df["quantity"].max() if len(flow_df) > 0 else 0

    bounds = world_df.total_bounds
    map_width = bounds[2] - bounds[0]
    max_r = map_width * 0.03
    min_r = max_r * 0.1
    loc_map = world_df.set_index("iso3")["centroid"].to_dict()

    def get_radius(val):
        if abs(val) <= 0 or global_max_bubble == 0:
            return 0
        return min_r + (max_r - min_r) * (np.sqrt(abs(val)) / np.sqrt(global_max_bubble))

    # --- trade flow lines (thickness ∝ quantity) ---
    for _, row in flow_df.iterrows():
        if row["exporter"] in loc_map and row["importer"] in loc_map and global_max_line > 0:
            p1, p2 = loc_map[row["exporter"]], loc_map[row["importer"]]
            lw = (row["quantity"] / global_max_line) * 5.5
            ax.plot(
                [p1.x, p2.x], [p1.y, p2.y],
                color=colors["flow_line"],
                linewidth=max(lw, 0.3),
                alpha=0.35,
                zorder=2
            )

    # --- net-export / net-import bubbles (area ∝ quantity) ---
    for _, row in world_df.iterrows():
        net_val = row["net_export"]
        if global_max_bubble > 0 and abs(net_val) > global_max_bubble * 0.001:
            r = get_radius(net_val)
            color = colors["surplus"] if net_val > 0 else colors["deficit"]
            ax.add_patch(
                Circle(
                    (row.centroid.x, row.centroid.y),
                    r,
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=5
                )
            )

    ax.set_title(
        f"Global {product_name} Trade ({year})",
        fontsize=22,
        fontweight="bold",
        pad=20
    )
    ax.axis("off")

    # ------------------------------------------------------------------ legend
    # 1) Category legend (surplus / deficit / flow line)
    cat_elements = [
        Line2D([0], [0], marker='o', color='w', label='Net exporter',
               markerfacecolor=colors["surplus"], markeredgecolor='white',
               markersize=12, alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='Net importer',
               markerfacecolor=colors["deficit"], markeredgecolor='white',
               markersize=12, alpha=0.7),
        Line2D([0], [0], color=colors["flow_line"], lw=2, label='Trade flow'),
    ]
    leg1 = ax.legend(
        handles=cat_elements,
        loc="lower left",
        frameon=True,
        fontsize=11,
        title="Legend",
        title_fontsize=11,
    )
    leg1.get_frame().set_facecolor(colors["legend_bg"])
    leg1.get_frame().set_edgecolor(colors["legend_edge"])
    ax.add_artist(leg1)   # keep leg1 when adding leg2

    # 2) Bubble-size scale: show 3 reference circles at 25 %, 50 %, 100 % of max
    if global_max_bubble > 0:
        scale_fracs = [0.25, 0.50, 1.0]
        scale_handles = []
        unit = "Mt" if value_column == "quantity" else "M USD"
        # BACI quantities are in metric tonnes → convert to Mt for display
        denom = 1e6 if value_column == "quantity" else 1e6
        for frac in scale_fracs:
            val = frac * global_max_bubble
            # marker size in points: radius in data units → approximate pt size
            # We use markersize proportional to sqrt(frac) scaled to a readable range
            ms = 6 + 14 * np.sqrt(frac)
            label_val = val / denom
            label = f"{label_val:.0f} {unit}"
            color = colors["surplus"]
            scale_handles.append(
                Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=color, markeredgecolor='white',
                       markersize=ms, alpha=0.7)
            )
        leg2 = ax.legend(
            handles=scale_handles,
            loc="lower right",
            frameon=True,
            fontsize=10,
            title="Net export volume\n(sizes proportional, not to map scale)",
            title_fontsize=9,
        )
        leg2.get_frame().set_facecolor(colors["legend_bg"])
        leg2.get_frame().set_edgecolor(colors["legend_edge"])
        ax.add_artist(leg2)

    # 3) Line-width scale: show 3 reference flow sizes
    if global_max_line > 0:
        line_fracs = [0.25, 0.50, 1.0]
        line_handles = []
        unit = "Mt" if value_column == "quantity" else "M USD"
        denom = 1e6 if value_column == "quantity" else 1e6
        for frac in line_fracs:
            val = frac * global_max_line
            lw = max((frac) * 5.5, 0.3)
            label = f"{val / denom:.0f} {unit}"
            line_handles.append(
                Line2D([0], [0], color=colors["flow_line"], lw=lw,
                       alpha=0.7, label=label)
            )
        leg3 = ax.legend(
            handles=line_handles,
            loc="upper right",
            frameon=True,
            fontsize=10,
            title="Trade flow volume",
            title_fontsize=10,
        )
        leg3.get_frame().set_facecolor(colors["legend_bg"])
        leg3.get_frame().set_edgecolor(colors["legend_edge"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    plt.close(fig)


# ==============================================================================
# 5. MAIN
# ==============================================================================
if __name__ == "__main__":
    from _helpers import mock_snakemake

    if "snakemake" not in globals():
        snakemake = mock_snakemake("plot_trade_today")

    # --- resolve inputs / outputs / config -----------------------------------
    BACI_FOLDER = str(snakemake.input.baci_folder)
    YEAR = 2024

    colors = snakemake.config["colors"]["trade_today"]

    # Map product names to snakemake output paths
    output_map = {
        "Iron Ore":    {"png": str(snakemake.output.iron_ore),  "csv": str(snakemake.output.iron_ore_csv)},
        "DRI-HBI":     {"png": str(snakemake.output.dri_hbi),   "csv": str(snakemake.output.dri_hbi_csv)},
        "Steel (raw)": {"png": str(snakemake.output.steel_raw), "csv": str(snakemake.output.steel_raw_csv)},
    }

    print("Loading world geometry")
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = world.rename(columns={"ADM0_A3": "iso3"}).to_crs("ESRI:54030")
    world = world[world["NAME"] != "Antarctica"]

    # --- loop over all products ----------------------------------------------
    for product_name, product_codes in PRODUCT_OPTIONS.items():
        print(f"\n=== {product_name} ===")
        trade_file = prepare_trade(
            baci_folder=BACI_FOLDER,
            product_name=product_name,
            product_codes=product_codes,
            out_csv=output_map[product_name]["csv"],
            year=YEAR,
        )

        data_plot = process_data_for_plot(
            trade_file=trade_file,
            world_gdf=world,
            value_column=VALUE_COLUMN,
        )

        plot(
            data_plot=data_plot,
            year=YEAR,
            product_name=product_name,
            value_column=VALUE_COLUMN,
            colors=colors,
            output_path=output_map[product_name]["png"],
        )

    print("\nDone.")
