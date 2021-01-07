import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
import pandas as pd

param_sweep = [
  ("a_game_name", ".*kuhn_poker.*"),
  ("b_depth", ".*"),
  # ("b_cfr_oracle_iters", ".*"),
  ("c_git_version", ".*"),
  # ("b_bandit", ".*"),
  # ("c_ball", ".*"),
]

display_perm = [
  # ("a_game_name",),
  ("a_game_name", "b_depth"),
  # ("b_cfr_oracle_iters",),
  ("c_git_version",),
  # ("b_bandit", "c_ball",),
]

base_dir = "./experiments/train_eval_loop"
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
  "d_subgame_cfr_iterations": "subgame iters"
}

def plot_item(ax, file, display_params, full_params):
  try:
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    print(file)

    ax.semilogy(df.loop, df.exploitability.rolling(window=20).mean(), label="expl (rolling mean)", c="r")
    ax.semilogy(df.loop, df.exploitability, alpha=0.2, c="r")
    ax.semilogy(df.loop, df.avg_loss.rolling(window=20).mean(), label="loss (rolling mean)", c="g")
    ax.semilogy(df.loop, df.avg_loss, alpha=0.2, c="g")
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
