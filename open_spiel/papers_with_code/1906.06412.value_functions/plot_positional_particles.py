import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import os

goof_pos = "./experiments/paper/paper_positional/game_name/goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)/depth/2/seed/%d/stdout"
goof_part = "./experiments/paper/paper_particles/game_name/goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)/depth/2/seed/%d/stdout"
leduc_pos = "./experiments/paper/paper_positional/game_name/leduc_poker/depth/7/seed/%d/stdout"
leduc_part = "./experiments/paper/paper_particles/game_name/leduc_poker/depth/7/seed/%d/stdout"

file_tpls = [goof_pos, goof_part, leduc_pos, leduc_part]
keys = ["goof_pos", "goof_part", "leduc_pos", "leduc_part"]

data = dict()
for key, file_tpl in zip(keys, file_tpls):
  data[key] = dict(loss=dict(), expl=dict())
  for seed in range(10):
    file = file_tpl % seed
    print(file)
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    df.set_index("loop")
    data[key]["loss"][seed] = df.avg_loss
    data[key]["expl"][seed] = df["expl[100]"]

for key in keys:
  data[key]["loss_df"] = pd.DataFrame(data[key]["loss"])
  data[key]["expl_df"] = pd.DataFrame(data[key]["expl"])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
fig, axes = plt.subplots(2,2, figsize=(5,3),
                         gridspec_kw=dict(wspace=0., hspace=0.))
axes[0, 0].set_title("Leduc Poker")
axes[0, 0].semilogy(data["leduc_pos"]["expl_df"].index, data["leduc_pos"]["expl_df"].mean(axis=1).rolling(10).mean())
axes[0, 0].semilogy(data["leduc_part"]["expl_df"].index, data["leduc_part"]["expl_df"].mean(axis=1).rolling(10).mean())
axes[1, 0].semilogy(data["leduc_pos"]["loss_df"].index, data["leduc_pos"]["loss_df"].mean(axis=1))
axes[1, 0].semilogy(data["leduc_part"]["loss_df"].index, data["leduc_part"]["loss_df"].mean(axis=1))

axes[0, 1].set_title("II GoofSpiel N=5")
axes[0, 1].semilogy(data["goof_pos"]["expl_df"].index, data["goof_pos"]["expl_df"].mean(axis=1).rolling(10).mean())
axes[0, 1].semilogy(data["goof_part"]["expl_df"].index, data["goof_part"]["expl_df"].mean(axis=1).rolling(10).mean())
axes[1, 1].semilogy(data["goof_pos"]["loss_df"].index, data["goof_pos"]["loss_df"].mean(axis=1), label="Positional model")
axes[1, 1].semilogy(data["goof_part"]["loss_df"].index, data["goof_part"]["loss_df"].mean(axis=1), label="Particle model")

axes[0, 0].set_ylim([5e-2, 1])
axes[0, 1].set_ylim([5e-2, 1])

axes[1, 0].set_ylim([1e-5, 5e-2])
axes[1, 1].set_ylim([1e-5, 5e-2])

axes[0,0].set_ylabel("Exploitability")
axes[0,1].set_yticks([])
axes[1,0].set_ylabel("MSE loss")
axes[1,1].set_yticks([])

axes[0,0].set_xticks([])
axes[0,1].set_xticks([])
axes[1,0].set_xlabel("Training iterations")
axes[1,1].set_xlabel("Training iterations")

leduc = data["leduc_pos"]["expl_df"].mean(axis=1).rolling(10).mean() / data["leduc_part"]["expl_df"].mean(axis=1).rolling(10).mean()
leduc = leduc.values[-100:]
print("Improvement leduc: ", (leduc.mean()))
goof = data["goof_pos"]["expl_df"].mean(axis=1).rolling(10).mean() / data["goof_part"]["expl_df"].mean(axis=1).rolling(10).mean()
goof = goof.values[-100:]
print("Improvement goof: ", (goof.mean()))

# axes[1, 1].legend(bbox_to_anchor=(0,0), loc="bottom left", ncol=2)
plt.tight_layout()
plt.show()

