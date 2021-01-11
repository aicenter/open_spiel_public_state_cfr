import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
import pandas as pd

param_sweep = [
  ("a_game_name", ".*"),
  ("b_depth", "(3|4)"),
  ("c_trunk_eval_iterations", "[1-8]"),
  ("d_data_generation", "dl_cfr"),
]

display_perm = [

  # ("c_trunk_eval_iterations",),
  ("a_game_name", "b_depth", "d_data_generation",),
  ("c_trunk_eval_iterations",),
]

base_dir = "./experiments/mix_data_generation"
translation_map = {
  "goofspiel(players=2,num_cards=3,imp_info=True)": "GS 3 (rand)",
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)":
    "GS 3 (asc)",
  "goofspiel(players=2,num_cards=4,imp_info=True)": "GS 4 (rand)",
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)":
    "GS 4 (asc)",
  "goofspiel(players=2,num_cards=5,imp_info=True)": "GS 5 (rand)",
  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=ascending)":
    "GS 5 (asc)",
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
  "d_subgame_cfr_iterations": "subgame iters"
}


def plot_item(ax, file, display_params, full_params):
  try:
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    print(file)
    df.loc[df.exploitability == 0, "exploitability"] = 1e-13

    # ax.semilogy(df.loop, df.exploitability.rolling(window=20).mean(),
    #             label="expl (rolling mean)", c="r")
    ax.semilogy(df.loop, df.exploitability, alpha=0.9, c="r")
    ax.semilogy(df.loop, df.avg_loss.rolling(window=20).mean(),
                label="loss (rolling mean)", c="g")
    ax.semilogy(df.loop, df.avg_loss, alpha=0.2, c="g")
    target_expls = {
      "kuhn_poker": {
        3: [0.0694444, 0.0138889, 0.0586623, 0.0579407, 0.037106, 0.0334998, 0.0166316, 0.0122922, 0.0136074, 0.00567496, 0.0049296, 0.00542391, 0.00439325, 0.00293351, 0.00415672, 0.00363367],
        4: [0.347222, 0.222222, 0.130093, 0.0642981, 0.0512305, 0.0490274, 0.0538898, 0.0629207, 0.0565416, 0.0519562, 0.0404916, 0.031254, 0.0315923, 0.0281698, 0.0246861, 0.0182519]
      },
      "leduc_poker": {
         3: [0.0551441, 0.0253679, 0.0606634, 0.0614051, 0.0506499, 0.0408017, 0.0302935, 0.0203831, 0.0188982, 0.0147639, 0.0113608, 0.0117267, 0.0108591, 0.00972253, 0.00828659, 0.00812331, 0.00678446],
         4: [0.237882, 0.0767588, 0.179079, 0.181522, 0.144847, 0.115943, 0.0939782, 0.0844154, 0.0791962, 0.0717062, 0.0633049, 0.0595763, 0.0553061, 0.0518049, 0.046881, 0.043043],
      }
    }
    game = str(full_params["a_game_name"])
    depth = int(full_params["b_depth"])
    iters = int(full_params["c_trunk_eval_iterations"])
    v = target_expls[game][depth][iters - 1]
    ax.semilogy([df.loop.min(), df.loop.max()], [v, v], label="DL-CFR target")

    ax.set_ylim([1e-6, 1])
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
