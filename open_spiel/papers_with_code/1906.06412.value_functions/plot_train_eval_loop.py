import grid_experiments.plot as plot
import grid_experiments.process as process
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np

base_dir="./experiments/snapshot_pbs_training"

select = dict(
    randomization=".*",
    arch="particle_vf",
    game_name=".*",
    depth=".*",
    sparse_particles=".*",
    seed=".*",
    save_values_policy="average",
    zero_sum_regression=".*",
)
pipeline = [
    process.read,
    process.average("seed"),
    process.concat("zero_sum_regression"),
]
layout = dict(
    x=[],
    y=["game_name", "depth", "save_values_policy"]
)
cell_size = [2, 1]



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
    for arch, df in data:
        axes[0, 0].semilogy(df.loop, df["expl[100]"].rolling(10).mean(),
                            label=f"{arch}: expl", alpha=1)
        axes[1, 0].semilogy(df.loop, df.avg_loss.rolling(10).mean(),
                            label="mse loss", alpha=1)

        for idx in range(0, 2305,256):
            axes[0, 0].semilogy([idx] * 2, [0.1, 2], "k--", alpha=0.2)
            axes[1, 0].semilogy([idx] * 2, [1e-3, 1e-14], "k--", alpha=0.2)

        # ax.semilogy(df.loop, df.replay_visits / 10, label="replay_visits", alpha=1)


def plot_cell(axes, params, data):
    print(params)
    for zero_sum, df in data:
        axes[0, 0].semilogy(df.loop, df["expl[100]"],
                            label=f"zero_sum={zero_sum} expl", alpha=1)
        axes[1, 0].semilogy(df.loop, df.avg_loss,
                            label=f"zero_sum={zero_sum} mse loss", alpha=1)



lazy_pipeline = process.make_lazy_pipeline(base_dir, select, pipeline)
# print(lazy_pipeline)
plot.display(lazy_pipeline, layout, cell_size, plot_cell, translation_map)
