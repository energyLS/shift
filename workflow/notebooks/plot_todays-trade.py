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
    "Steel (raw)": [720711, 720712, 720719, 720720],
}

# Safe filename stems matching Snakemake output keys
PRODUCT_SAFE_NAMES = {
    "Iron Ore":    "Iron_Ore",
    "DRI-HBI":     "DRI-HBI",
    "Steel (raw)": "Steel_raw",
}

# How many bilateral net-flow arrows to draw on the map.
# Larger values add more (smaller) trade arrows; 100 keeps the map readable
# while still capturing the dominant flows.
TOP_N_FLOWS = 100

# Which column to size bubbles and lines by.
# "quantity"    → physical volume (metric tonnes) — best for comparing trade magnitude.
# "trade_value" → USD value — useful if price differences between products matter.
VALUE_COLUMN = "quantity"



# ==============================================================================
# 1. PREPARE TRADE DATA
# ==============================================================================
def prepare_trade(baci_folder, product_name, product_codes, out_dir, year):
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

    out_iso3 = os.path.join(out_dir, f"{product_name}_trade_iso3.csv")
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
    max_r = (bounds[2] - bounds[0]) * 0.03
    min_r = max_r * 0.1
    loc_map = world_df.set_index("iso3")["centroid"].to_dict()

    def get_radius(val):
        if abs(val) <= 0 or global_max_bubble == 0:
            return 0
        return min_r + (max_r - min_r) * (np.sqrt(abs(val)) / np.sqrt(global_max_bubble))

    for _, row in flow_df.iterrows():
        if row["exporter"] in loc_map and row["importer"] in loc_map and global_max_line > 0:
            p1, p2 = loc_map[row["exporter"]], loc_map[row["importer"]]
            lw = (row["quantity"] / global_max_line) * 5.5
            ax.plot(
                [p1.x, p2.x], [p1.y, p2.y],
                color=colors["flow_line"],
                linewidth=max(lw, 0.5),
                alpha=0.3,
                zorder=2
            )

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
        f"Global {product_name} Trade",
        fontsize=22,
        fontweight="bold",
        pad=20
    )
    ax.axis("off")

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Net exporter',
            markerfacecolor=colors["surplus"],
            markeredgecolor='white',
            markersize=12,
            alpha=0.7,
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Net importer',
            markerfacecolor=colors["deficit"],
            markeredgecolor='white',
            markersize=12,
            alpha=0.7,
        ),
        Line2D(
            [0], [0],
            color=colors["flow_line"],
            lw=2,
            label='Trade flow'
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        fontsize=12
    )

    leg.get_frame().set_facecolor(colors["legend_bg"])
    leg.get_frame().set_edgecolor(colors["legend_edge"])

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
        "Iron Ore":    str(snakemake.output.iron_ore),
        "DRI-HBI":     str(snakemake.output.dri_hbi),
        "Steel (raw)": str(snakemake.output.steel_raw),
    }

    print("Loading world geometry")
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = world.rename(columns={"ADM0_A3": "iso3"}).to_crs("ESRI:54030")
    world = world[world["NAME"] != "Antarctica"]

    # --- loop over all products ----------------------------------------------
    for product_name, product_codes in PRODUCT_OPTIONS.items():
        safe_name = PRODUCT_SAFE_NAMES[product_name]
        out_dir = os.path.join(BACI_FOLDER, f"_outputs_{safe_name}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== {product_name} ===")
        trade_file = prepare_trade(
            baci_folder=BACI_FOLDER,
            product_name=product_name,
            product_codes=product_codes,
            out_dir=out_dir,
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
            output_path=output_map[product_name],
        )

    print("\nDone.")
