import glob
from os.path import basename

import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

base_dir = "cfr_comparison"
games = glob.glob(f"{base_dir}/a_game_name/*")
translation_map={
  "goofspiel(players=2,num_cards=3,imp_info=True)": "GS 3 (random)",
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)": "GS 3 (ascending)",
  "goofspiel(players=2,num_cards=4,imp_info=True)": "GS 4 (random)",
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)": "GS 4 (ascending)",
  "goofspiel(players=2,num_cards=5,imp_info=True)": "GS 5 (random)",
  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=ascending)": "GS 5 (ascending)",
  "kuhn_poker": "Kuhn Poker",
  "leduc_poker": "Leduc Poker",
  "matrix_biased_mp": "Biased MP",
  "small_matrix": "Small Matrix",
  "matrix_mp": "Matching Pennies",
  "PredictiveRegretMatching": "PRM",
  "PredictiveRegretMatchingPlus": "PRM+",
  "RegretMatching": "RM",
  "RegretMatchingPlus": "RM+",
}


def l(label_lookup):
  return (translation_map[label_lookup]
          if label_lookup in translation_map else label_lookup)


def plot_files(dirs, ax):
  for dir in natsorted(dirs):
    file = f"{dir}/stdout"
    print(file)
    df = pd.read_csv(file)
    label = basename(dir)
    ax.loglog(df.iters, df.avg_expl, label=l(label))


fig, axes = plt.subplots(1, len(games))
axes[0].set_ylabel("exploitability (avg strat)")

for i, game in enumerate(natsorted(games)):
  axes[i].set_title(l(basename(game)))
  plot_files(glob.glob(f"{game}/b_bandit_name/*"), axes[i])

for ax in axes:
  ax.legend(loc="lower left")

plt.show()
