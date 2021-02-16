import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import os

param_sweep = [
  ("game_name", "goof.*"),
  ("depth", ".*"),
  ("sparse_roots_depth", ".*"),
  ("support_threshold", ".*"),
]

display_perm = [
  ("game_name", "depth", "sparse_roots_depth",),
  ("support_threshold", ),
]

base_dir = "./experiments/eq_threshold_particles"
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
  "b_depth": "trunk depth",
  "c_depth": "trunk depth",
  "c_trunk_eval_iterations": "trunk iters",
  "d_subgame_cfr_iterations": "subgame iters",
  "limit_particle_count": "train ps",
  "support_threshold": "th"
}

target_expls = {
  "kuhn_poker": {
    3: [ 0.0694444,
         0.0138889,
         0.037106,
         0.00567496,
         0.00598514,
         0.00520348,
         0.00367275],
    4: [0.347222,
        0.222222,
        0.0512305,
        0.0519562,
        0.0158763,
        0.00953119,
        0.0073928]
  },
  "leduc_poker": {
    4: [0.237882,
        0.0767588,
        0.144847,
        0.0717062,
        0.0328893,
        0.0159417,
        0.00916333],
    6: [ 0.90119,
         0.507419,
         0.227721,
         0.111212,
         0.070699,
         0.0456274,
         0.0316788],
    8: [ 1.89087,
         1.35069,
         1.03251,
         0.625757,
         0.22752,
         0.094673,
         0.0607483,]
  },
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)": {
    1: [0.444444,
        0.148148,
        0.0296296,
        0.00808081,
        0.0021164,
        0.000348584,
        8.80088e-05]
  },
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=descending)": {
    1: [0.5,
        0.166667,
        0.0333333,
        0.00909091,
        0.00238095,
        0.000392157,
        9.90099e-05]
  },
  "goofspiel(players=2,num_cards=4,imp_info=True)": {
    1: [0.320064,
        0.278664,
        0.0942615,
        0.0587052,
        0.0235283,
        0.0107692,
        0.00857602,],
    2: [0.576389,
        0.410799,
        0.284437,
        0.191588,
        0.0614958,
        0.0177474,
        0.0217346]
  },
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)": {
    1: [0.416667,
        0.138889,
        0.0373442,
        0.0264351,
        0.0202786,
        0.0141592,
        0.0104897],
    2: [0.638889,
        0.353324,
        0.18645,
        0.136918,
        0.0279113,
        0.0138864,
        0.0102248]
  },
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)": {
    1: [0.25,
        0.66397,
        0.144617,
        0.0568192,
        0.0380868,
        0.0198911,
        0.013044],
    2: [0.541667,
        0.447222,
        0.491152,
        0.320039,
        0.0703319,
        0.0146563,
        0.0156218]
  },
  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)": {
    1: [0.295238,
        0.364061,
        0.12123,
        0.145935,
        0.0397219,
        0.0446541,
        0.0340776],
    2: [0.519444,
        0.811044,
        0.573685,
        0.360794,
        0.155102,
        0.0492832,
        0.023247]
  }
}

def file_from_params(full_params):
  full_dir = f"{base_dir}"
  for (param_name, mask) in param_sweep:
    full_dir += f"/{param_name}/{full_params[param_name]}"
  return f"{full_dir}/stdout"


colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]

def plot_item(ax, file, display_params, full_params):
  try:
    game = str(full_params["game_name"])
    # depth = int(full_params["depth"])
    col = "expl[100]"
    # v = target_expls[game][depth][6]
    # ax.semilogy([0, 1000], [v, v], label=f"DL-CFR target")

    print(file)
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    ax.semilogy(df.loop, df[col], label=f"{col}", alpha=1)
    # ax.semilogy(df.loop, df[col].rolling(10).mean(), c=colors[i], label=f"{col} {net_model}")
    ax.semilogy(df.loop, df.avg_loss, label="mse loss", alpha=1)

    ax.set_ylim([1e-5, 1])
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
