import os
import re
from os.path import basename
from pprint import pprint

from itertools import product, chain
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

# Comparison at 100 subgame iters:
# param_sweep = [
#   ("a_game_name", "(matrix_biased_mp|leduc_poker|kuhn_poker|goofspiel.*)"),
#   ("b_bandit_name", ".*"),
#   ("c_depth", "(1|2|3|4|5)"),
#   ("d_subgame_cfr_iterations", "(100)")
# ]
# display_perm = [
#   ("a_game_name", "d_subgame_cfr_iterations"),
#   ("c_depth",),
#   ("b_bandit_name", )
# ]

# Comparison of subgame iters at specific games
param_sweep = [
  ("a_game_name", "(goofspiel\(players=2,num_cards=4,imp_info=True\))"),
  ("b_bandit_name", ".*"),
  ("c_depth", "(1|2)"),
  ("d_subgame_cfr_iterations", ".*")
]
display_perm = [
  ("a_game_name", "b_bandit_name",),
  ("c_depth",),
  ("d_subgame_cfr_iterations", )
]

base_dir = "./value_fn_convergence_cfr"
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
  "kuhn_poker": "Kuhn Poker",
  "leduc_poker": "Leduc Poker",
  "matrix_biased_mp": "Biased MP",
  "small_matrix": "Small Matrix",
  "matrix_mp": "Matching Pennies",
  "PredictiveRegretMatching": "PRM",
  "PredictiveRegretMatchingPlus": "PRM+",
  "RegretMatching": "RM",
  "RegretMatchingPlus": "RM+",
  "c_depth": "trunk depth",
  "d_subgame_cfr_iterations": "subgame iters"
}

def plot_item(ax, display_params, full_params):
  try:
    df = pd.read_csv(file_from_params(full_params))
    df.avg_expl = df.avg_expl.apply(lambda x: max(1e-13, x))
    print(".", end="")
    ax.semilogy(df.iters, df.avg_expl, label=lbls(display_params))
  except Exception as e:
    pass
    # print(e)


def file_from_params(full_params):
  full_dir = f"{base_dir}"
  for (param_name, mask) in param_sweep:
    full_dir += f"/{param_name}/{full_params[param_name]}"
  return f"{full_dir}/stdout"


def lbl(label_lookup):
  return (translation_map[label_lookup]
          if label_lookup in translation_map else label_lookup)

def lbls(label_lookups, sep=", "):
  if isinstance(label_lookups, dict):
    # print(label_lookups)
    return sep.join(f"{lbl(k)}={lbl(v)}" if lbl(v) == v else lbl(v)
                     for k, v in label_lookups.items())
  else:
    return sep.join(lbl(lookup) for lookup in label_lookups)


def glob_regex(pattern):
  # print(pattern)
  match_dir = re.compile(pattern)
  ds = []
  for dirpath, dirnames, filenames in os.walk(".", topdown=True):
    if match_dir.search(dirpath):
      ds.append(dirpath)
    # print(dirpath)
    if pattern.count("/") - dirpath.count("/") <= 0:
      del dirnames[:]
  return ds

def kv_dict(ks, vs):
  return {k: v for k,v in zip(ks, vs)}


shape = {param: 1 for (param, card) in param_sweep}
params = {param: set() for (param, card) in param_sweep}
lookup = base_dir
for i, (param, card) in enumerate(param_sweep):
  lookup += f"/{param}/{card}"

skip = len(base_dir)
for file in glob_regex(f"{lookup}$"):
  path_dirs = file[skip:].split("/")[1:]
  for x in range(0, len(path_dirs), 2):
    (param_idx, value_idx) = x, x+1
    param, value = path_dirs[param_idx], path_dirs[value_idx]
    print(param, value)
    params[param].add(value)

for param, value_set in params.items():
  shape[param] = len(value_set)

for param in params.keys():
  params[param] = natsorted(list(params[param]))

# else:
#   raise Exception(f"No match for {lookup}")

print("-" * 80)
print("Data shape: ")
pprint(shape)
print("-" * 80)
print("Param sweep: ")
pprint(params)
print("-" * 80)

display_shape = [1] * 3
display_gen = []
display_lists = [[], [], []]
assert 1 <= len(display_perm) <= 3
for i, items in enumerate(display_perm):
  for item in items:
    display_shape[i] *= shape[item]
    display_lists[i].append(params[item])
  display_gen.append(lambda: product(*display_lists[i]))

fig, axes = plt.subplots(nrows=display_shape[0], ncols=display_shape[1],
                         sharex='col', sharey='row', squeeze=False,
                         gridspec_kw=dict(wspace=0.1, hspace=0.1))

for i, items in enumerate(product(*display_lists[0])):
  plt.setp(axes[i][0], ylabel=lbls(kv_dict(display_perm[0], items), sep=",\n"))
  axes[i][0].yaxis.label.set_size(7)

for j, items in enumerate(product(*display_lists[1])):
  plt.setp(axes[-1][j], xlabel=lbls(kv_dict(display_perm[1], items), sep=",\n"))


for i, a in enumerate(product(*display_lists[0])):
  for j, b in enumerate(product(*display_lists[1])):
    for k, c in enumerate(product(*display_lists[2])):
      full_plot_params = dict()
      for k, v in chain(zip(display_perm[0], a),
                        zip(display_perm[1], b),
                        zip(display_perm[2], c)):
        full_plot_params[k] = v

      plot_params = {k: v for k,v in zip(display_perm[2], c)}
      plot_item(axes[i][j], plot_params, full_plot_params)

for i in range(display_shape[0]):
  for j in range(display_shape[1]):
    if i < display_shape[0] - 1:
      plt.setp(axes[i][j].get_xticklabels(), visible=False)
    if j > 0:
      plt.setp(axes[i][j].get_yticklabels(), visible=False)

axes[-1][-1].legend(loc="lower left")

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
