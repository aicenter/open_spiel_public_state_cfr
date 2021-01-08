import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
import pandas as pd

param_sweep = [
  ("a_game_name", ".*kuhn_poker.*"),
  ("b_depth", ".*"),
  ("c_trunk_eval_iterations", ".*"),
]

display_perm = [
  ("a_game_name", "b_depth",),
  ("c_trunk_eval_iterations",),
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
  "c_trunk_eval_iterations": "iters",
  "d_subgame_cfr_iterations": "subgame iters"
}


def plot_item(ax, file, display_params, full_params):
  try:
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    print(file)
    df.loc[df.exploitability == 0, "exploitability"] = 1e-13

    ax.semilogy(df.loop, df.exploitability.rolling(window=20).mean(),
                label="expl (rolling mean)", c="r")
    ax.semilogy(df.loop, df.exploitability, alpha=0.2, c="r")
    ax.semilogy(df.loop, df.avg_loss.rolling(window=20).mean(),
                label="loss (rolling mean)", c="g")
    ax.semilogy(df.loop, df.avg_loss, alpha=0.2, c="g")
    target_expls = {
      3: [0.0694444, 0.0347222, 0.0277778, 0.0208333, 0.0305556, 0.0266204,
          0.0228175, 0.0210069, 0.0187463, 0.0168717, 0.0154247, 0.0141393,
          0.0143215, 0.0132985, 0.0125376, 0.011754, 0.0112125, 0.0105896,
          0.0115995, 0.0110195, 0.0106958],
      4: [0.347222, 0.229167, 0.164352, 0.138641, 0.0992965, 0.0783782,
          0.074789, 0.0671453, 0.0573168, 0.0496838, 0.0469723, 0.04122,
          0.0396299, 0.0392783, 0.0386818, 0.0376868, 0.0355828, 0.0325483,
          0.02994, 0.0269017]}
    depth = int(full_params["b_depth"])
    iters = int(full_params["c_trunk_eval_iterations"])
    v = target_expls[depth][iters - 1]
    ax.semilogy([df.loop.min(), df.loop.max()], [v, v], label="DL-CFR target")

    ax.set_ylim([1e-6, 1])
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
