import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
import pandas as pd

param_sweep = [
  ("a_game_name", ".*"),
  ("b_depth", "[13-8]"),
  # ("c_trunk_eval_iterations", "[1-8]"),
  ("c_data_generation", "random"),
]

display_perm = [

  # ("c_trunk_eval_iterations",),
  ("a_game_name", "c_data_generation",),
  ("b_depth",),
]

base_dir = "./experiments/eval_iters"
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

def file_from_params(full_params):
  full_dir = f"{base_dir}"
  for (param_name, mask) in param_sweep:
    full_dir += f"/{param_name}/{full_params[param_name]}"
  return f"{full_dir}/stdout"

def plot_item(ax, file, display_params, full_params):
  try:
    file_random = file_from_params(full_params)
    full_params2 = full_params
    full_params2.update({"c_data_generation": "dl_cfr"})
    file_dlcfr = file_from_params(full_params2)
    df = pd.read_csv(file_random, comment="#", skip_blank_lines=True)
    df2 = pd.read_csv(file_dlcfr, comment="#", skip_blank_lines=True)
    print(file)
    # df.loc[df.exploitability == 0, "exploitability"] = 1e-13

    # ax.semilogy(df.loop, df.exploitability.rolling(window=20).mean(),
    #             label="expl (rolling mean)", c="r")
    expl_cols = [col for col in df.columns if col.startswith("expl")]

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
      }
    }

    game = str(full_params["a_game_name"])
    depth = int(full_params["b_depth"])

    # for i, col in enumerate(expl_cols):
    #   ax.semilogy(df.loop, df[col], alpha=i / len(expl_cols), c="r", label=col)
    #   v = target_expls[game][depth][i]
    #   ax.semilogy([df.loop.min(), df.loop.max()], [v, v], label=f"DL-CFR target {col}")

    col = "expl[20]"
    i = 4
    ax.semilogy(df.loop, df[col], c="r", label=f"{col} random")
    ax.semilogy(df2.loop, df2[col], c="g", label=f"{col} dlcfr")
    v = target_expls[game][depth][i]
    ax.semilogy([df.loop.min(), df.loop.max()], [v, v], label=f"DL-CFR target {col}")

    # ax.semilogy(df.loop, df.avg_loss.rolling(window=20).mean(),
    #             label="loss (rolling mean)", c="g")
    # ax.semilogy(df.loop, df.avg_loss, alpha=0.2, c="g")

    # ax.set_ylim([1e-6, 1])
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
