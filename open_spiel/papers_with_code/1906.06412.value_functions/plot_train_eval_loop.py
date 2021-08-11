import grid_experiments.plot as plot
import grid_experiments.process as process
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np

# base_dir="./experiments/snapshot_pbs_training"
base_dir="./experiments/iigs_kn"

select = dict(
    randomization=".*",
    arch="particle_vf",
    game_name="goof.*",
    num_loops=".*",
    bootstrap_from_move=".*",
    depth=".*",
    sparse_particles=".*",
    seed=".*",
    save_values_policy="average",
    zero_sum_regression=".*",
    safe="false",
    kn="3-5$",
    norm=".*",
)
pipeline = [
    process.read,
    # process.average("seed"),
    # process.concat("zero_sum_regression"),
    process.concat("safe"),
]
layout = dict(
    y=[],
    x=["kn"]
    # y=["game_name", "depth", "num_loops", "bootstrap_from_move"]
    # y=["game_name", "depth", "save_values_policy", "seed"]
)
cell_size = [3, 1]



translation_map = {
    "goofspiel(players=2,num_cards=3,imp_info=True)": "GS 3 (rand)",
    "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)":
        "GS 3 (asc)",
    "goofspiel(players=2,num_cards=3,imp_info=True,points_order=descending)":
        "GS 3 (desc)",
    "goofspiel(players=2,num_cards=4,imp_info=True)": "GS 4 (rand)",
    "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)":
        "GS 4 (asc)",
    "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)":
        "GS 4 (desc)",
    "goofspiel(players=2,num_cards=5,imp_info=True)": "GS 5 (rand)",
    "goofspiel(players=2,num_cards=5,imp_info=True,points_order=ascending)":
        "GS 5 (asc)",
    "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)":
        "GS 5 (desc)",
    "goofspiel(players=2,num_cards=6,imp_info=True)": "GS 6 (rand)",
    "goofspiel(players=2,num_cards=6,imp_info=True,points_order=ascending)":
        "GS 6 (asc)",
    "kuhn_poker": "Kuhn Poker",
    "leduc_poker": "Leduc Poker",
    "matrix_biased_mp": "Biased MP",
    "small_matrix": "Small Matrix",
    "matrix_mp": "Matching Pennies",
    "PredictiveRegretMatching": "PRM",
    "PredictiveRegretMatchingPlus": "PRM+",
    "RegretMatching": "RM",
    "RegretMatchingPlus": "RM+",
    "limit_particle_count": "train ps",
    "support_threshold": "th"
}

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]


def plot_bootstrap(axes, params, data):
    df = data
    axes[0, 0].semilogy(df.loop, df["expl[100]"],
                        label=f"expl", alpha=1)
    axes[1, 0].semilogy(df.loop, df.avg_loss,
                        label="mse loss", alpha=1)

    for idx in range(0, 10*512, 512):
        move = 10 - idx / 512
        if move > 1:
            axes[0, 0].annotate( f'Move: {int(move)}', (idx+64, 0.05))
        else:
            axes[0, 0].annotate(f'Final retraining', (idx+64, 0.05))

        axes[0, 0].semilogy([idx] * 2, [0.05, 2], "k--", alpha=0.2)
        axes[1, 0].semilogy([idx] * 2, [1e-2, 1e-6], "k--", alpha=0.2)

    axes[1, 0].set_ylim([1e-6, 5e-3])


def plot_cell(axes, params, data):
    for normalize, dfs in data:
        df=dfs[0]
        axes[0, 0].plot(df.loop, df["expl[100]"],
                        label=f"normalize={normalize} expl", alpha=1)
        axes[1, 0].semilogy(df.loop, df.avg_loss,
                            label=f"normalize={normalize} mse loss", alpha=1)

def plot_cell2(axes, params, data):
    x = data[0][1][0]
    axes[0, 0].semilogy(x.loop, x["expl[10]"], label=f"expl 10", alpha=1)
    axes[0, 0].semilogy(x.loop, x["expl[50]"], label=f"expl 50", alpha=1)
    axes[0, 0].semilogy(x.loop, x["expl[100]"], label=f"expl 100", alpha=1)
    axes[0, 0].legend(loc="lower left")
    axes[2, 0].semilogy(x.loop, x.avg_loss, label=f"mse loss", alpha=1)
    axes[2, 0].legend(loc="lower left")

    for safe, df in data:
        df = df[0]
        axes[1, 0].semilogy(df.loop, df["br"].rolling(20).mean(), label=f"br safe={safe} (mean)", alpha=1)
        axes[1, 0].semilogy(df.loop, df["br"], label=f"br safe={safe}", alpha=0.3)
    axes[1, 0].legend(loc="lower left")




lazy_pipeline = process.make_lazy_pipeline(base_dir, select, pipeline)
# print(lazy_pipeline)
# plot.display(lazy_pipeline, layout, cell_size, plot_cell, translation_map)
plot.display(lazy_pipeline, layout, cell_size, plot_cell2, translation_map)
# plot.display(lazy_pipeline, layout, cell_size, plot_bootstrap, translation_map)
