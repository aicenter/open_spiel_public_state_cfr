import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter

goof_th = ["0.0000000", "0.0001614", "0.0010183", "0.0013163", "0.0032164",
           "0.0040236", "0.0045824", "0.0078920", "0.0107214", "0.0181064",
           "0.6150000", ]
leduc_th = ["0.00000000", "0.00031032", "0.00037930", "0.00063324",
            "0.00093430", "0.00116098", "0.00145379", "0.00219816",
            "0.00261279", "0.00477722", "0.00562000", ]
file_tpl_goof = "experiments/paper/paper_sparse_eq/game_name/goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)/depth/3/sparse_roots_depth/2/support_threshold/%s/seed/%d/stdout"
file_tpl_leduc = "experiments/paper/paper_sparse_eq/game_name/leduc_poker/depth/7/sparse_roots_depth/5/support_threshold/%s/seed/%d/stdout"

def reject_outliers(sr, iq_range=0.5):
  pcnt = (1 - iq_range) / 2
  qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
  iqr = qhigh - qlow
  return sr[ (sr - median).abs() <= iqr]


def get_expls(file_tpl, ths, seeds):
  results = []
  for th in ths:
    expls = []
    for seed in range(seeds):
      file = file_tpl % (th, seed)
      print(file)
      df = pd.read_csv(file, comment="#", skip_blank_lines=True)
      expls.append(df["expl[100]"].values[-5:][:])
    x = np.array(expls).flatten()
    expls = reject_outliers(pd.Series(x), iq_range=0.99)
    results.append([float(th), expls.mean(), expls.std()])
  return np.array(results)

seeds = 10
goof_expl = get_expls(file_tpl_goof, goof_th, seeds)
goof_unif = 0.419816
supp_hist_goof = 60
frac_single_goof = 1. / supp_hist_goof
full_hist_goof = 400

leduc_expl = get_expls(file_tpl_leduc, leduc_th, seeds)
leduc_unif = 0.0793766
supp_hist_leduc = 290
frac_single_leduc = 2. / supp_hist_leduc
full_hist_leduc = 330

fractions = np.linspace(1, 0.1, 10)


fig, axes = plt.subplots(1,2, figsize=(5 ,3), sharey=True, gridspec_kw=dict(wspace=0., hspace=0.))
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
axes[0].set_ylabel("Exploitability")

axes[0].set_title("Leduc Poker")
leduc_x = list(fractions) + [frac_single_leduc]
axes[0].errorbar(x=leduc_x, y=leduc_expl[:, 1], yerr=leduc_expl[:, 2])
axes[0].set_xticks(leduc_x)
axes[0].plot([min(leduc_x), max(leduc_x)], [leduc_unif, leduc_unif])
axes[0].tick_params(axis="x", rotation=50)

axes[1].set_title("II GoofSpiel N=5")
goof_x = list(fractions) + [frac_single_goof]
axes[1].errorbar(x=goof_x, y=goof_expl[:, 1], yerr=goof_expl[:, 2])
axes[1].set_xticks(goof_x)
axes[1].tick_params(axis="x", rotation=50)
# axes[1].plot([min(goof_x), max(goof_x)], [goof_unif, goof_unif])
plt.tight_layout()
plt.show()

# agg_plot(axes[0], "./experiments/model_cmp_paper/model/particles/game_name/leduc_poker/depth/7/seed/%d/stdout")
# plt.show()
# ax.semilogy(df.index, df[col], label=f"{col}", alpha=1)
#
# df_expl = pd.DataFrame(expls)
# means = df_expl.mean(axis=1) #.rolling(10).mean()
# # err = df_expl.var(axis=1)
#
# ax.semilogy(df_expl.index, means, label=f"{col}", alpha=1)
# ax.semilogy(df_expl.index, err, label=f"{col}", alpha=1)
# ax.fill_between(df_expl.index, means-err, means+err, color="r", alpha=0.5)
