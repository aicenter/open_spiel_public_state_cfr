import re
import grid_experiments.plot as plot
import grid_experiments.process as process
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np

# base_dir="./experiments/snapshot_pbs_training"
base_dir="./experiments/iigs_kn/particle_generation/training_all_particles"

select = dict(
    randomization=".*",
    arch="particle_vf",
    game_name=".*num_cards=[4567].*",
    num_loops=".*",
    bootstrap_from_move=".*",
    depth=".*",
    sparse_particles=".*",
    seed=".*",
    save_values_policy="average",
    zero_sum_regression=".*",
    safe_resolving="true",
    particle_generation="(pareto).*",
    cfr_oracle_iterations=".*",
    cfr_iterations=".*",
    max_particles=".*",
    particle_selection=".*",
    set_pooling=".*",
    kn="3-5$",
    norm=".*",
)
pipeline = [
    process.read,
    process.average("seed"),
    # process.concat("zero_sum_regression"),
    # process.concat("seed"),
    process.concat("set_pooling"),
    process.concat("max_particles"),
]
layout = dict(
    x=["game_name"],
    y=[]
    # y=["game_name", "depth", "num_loops", "bootstrap_from_move"]
    # y=["game_name", "depth", "save_values_policy", "seed"]
)
cell_size = [3, 1]

uniform_br = {
    "4":0.791667,
    "5":0.883333,
    "6":0.925,
    "7":0.947619,
    "8":0.96131,
    "9":0.970238,
    "10":0.976389,
}

oracle_br = {
    # 64 particles
    "4": 0.000569459,
    "5": 0.0027706,
    "6": 0.0692271,

    # All particles:
    # 0.000569459
    # 0.0027706
    # 0.0692271
}

translation_map = {
    "goofspiel(players=2,num_cards=4,num_turns=3,imp_info=True,points_order=descending)": "GS(3,4)",
    "goofspiel(players=2,num_cards=5,num_turns=3,imp_info=True,points_order=descending)": "GS(3,5)",
    "goofspiel(players=2,num_cards=6,num_turns=3,imp_info=True,points_order=descending)": "GS(3,6)",
    "goofspiel(players=2,num_cards=7,num_turns=3,imp_info=True,points_order=descending)": "GS(3,7)",
    "goofspiel(players=2,num_cards=8,num_turns=3,imp_info=True,points_order=descending)": "GS(3,8)",
    "goofspiel(players=2,num_cards=9,num_turns=3,imp_info=True,points_order=descending)": "GS(3,9)",
    "goofspiel(players=2,num_cards=10,num_turns=3,imp_info=True,points_order=descending)": "GS(3,10)",
    "goofspiel(players=2,num_cards=11,num_turns=3,imp_info=True,points_order=descending)": "GS(3,11)",

    "goofspiel(players=2,num_cards=5,num_turns=4,imp_info=True,points_order=descending)": "GS(4,5)",
    "goofspiel(players=2,num_cards=6,num_turns=4,imp_info=True,points_order=descending)": "GS(4,6)",
    "goofspiel(players=2,num_cards=7,num_turns=4,imp_info=True,points_order=descending)": "GS(4,7)",


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
    df = data
    axes[0, 0].semilogy(df.loop, df["expl[100]"],
                        label=f"expl", alpha=1)
    axes[1, 0].semilogy(df.loop, df.avg_loss,
                        label="mse loss", alpha=1)

    for idx in range(0, 10*512, 512):
        move = 10 - idx / 512
        if move > 1:
            axes[0, 0].annotate( f'Move: {int(move)}', (idx+64, 0.05))
        else:
            axes[0, 0].annotate(f'Final retraining', (idx+64, 0.05))

        axes[0, 0].semilogy([idx] * 2, [0.05, 2], "k--", alpha=0.2)
        axes[1, 0].semilogy([idx] * 2, [1e-2, 1e-6], "k--", alpha=0.2)

    axes[1, 0].set_ylim([1e-6, 5e-3])


def plot_cell(axes, params, data):
    for normalize, dfs in data:
        df=dfs[0]
        axes[0, 0].plot(df.loop, df["expl[100]"],
                        label=f"normalize={normalize} expl", alpha=1)
        axes[1, 0].semilogy(df.loop, df.avg_loss,
                            label=f"normalize={normalize} mse loss", alpha=1)

def plot_oracle(axes, params, data):
    print(data)
    # x = data[0][0][0]
    # axes[0, 0].semilogy(x.loop, x["expl[10]"], label=f"expl 10", alpha=1)
    # axes[0, 0].semilogy(x.loop, x["expl[50]"], label=f"expl 50", alpha=1)
    # axes[0, 0].semilogy(x.loop, x["expl[100]"], label=f"expl 100", alpha=1)
    # axes[0, 0].legend(loc="lower left")
    # axes[2, 0].semilogy(x.loop, x.avg_loss, label=f"mse loss", alpha=1)
    # axes[2, 0].legend(loc="lower left")
    p = re.compile("num_cards=(\d+)")
    cards = p.search(params["game_name"]).group(1)

    for particle_selection, dfs in data:
        x = []
        for max_particles, df in dfs:
            try:
                x.append((int(max_particles), df[0].tail(1)["br"][0]))
            except Exception: pass
        x = sorted(x, key=lambda i: i[0])
        for p, v in x:
            if particle_selection == "max_q_values_oracle":
                mark = "o"
                axes[0, 0].loglog(p, v, mark)
                axes[0, 0].annotate(str(p), xy=(p, v),  xycoords='data',
                                    xytext=(p, v))
            elif particle_selection == "value_invariant_prm":
                mark = "+"
                axes[0, 0].loglog(p, v, mark)
                axes[0, 0].annotate(str(p), xy=(p, v),  xycoords='data',
                                    xytext=(p, v))
            elif particle_selection == "pareto":
                mark = "o"
                # axes[0, 0].loglog(p, v, mark)
                # axes[0, 0].annotate(str(p), xy=(p, v),  xycoords='data',
                #                     xytext=(p, v))

            # axes[0, 0].loglog(p, v, mark)
            # axes[0, 0].annotate(str(p), xy=(p, v),  xycoords='data',
            #                     xytext=(p, v))


def plot_net(axes, params, data):
    print(data)
    p = re.compile("num_cards=(\d+)")
    cards = p.search(params["game_name"]).group(1)
    br_value = uniform_br[cards]
    oracle_value = oracle_br[cards]

    for num_parts, df_parts in data:
        for pool, df in df_parts:
            # df= df[0]
            axes[0, 0].semilogy(df.loop, df["br"], label=f"br pool={pool} parts={num_parts}")
            axes[1, 0].plot(df.loop, df["returns"], label=f"returns pool={pool} parts={num_parts}")
            axes[2, 0].semilogy(df.loop, df["avg_loss"], label=f"mse loss pool={pool} parts={num_parts}")

    df = data[0][1][0][1]
    # df = data[0][1]
    axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
                        [br_value]*2, label=f"br against uniform")
    axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
                        [br_value*0.1]*2, label=f"br target")
    axes[1, 0].plot([df.loop.head(1), df.loop.tail(1)],
                        [0]*2, label=f"zero")
    # axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
    #                     [oracle_value]*2, label=f"br oracle")

    axes[0, 0].legend(loc="lower left")
    axes[1, 0].legend(loc="lower left")

# def plot_cell2(axes, params, data):
#     # print(data)
#     # x = data[0][0][0]
#     # axes[0, 0].semilogy(x.loop, x["expl[10]"], label=f"expl 10", alpha=1)
#     # axes[0, 0].semilogy(x.loop, x["expl[50]"], label=f"expl 50", alpha=1)
#     # axes[0, 0].semilogy(x.loop, x["expl[100]"], label=f"expl 100", alpha=1)
#     # axes[0, 0].legend(loc="lower left")
#     # axes[2, 0].semilogy(x.loop, x.avg_loss, label=f"mse loss", alpha=1)
#     # axes[2, 0].legend(loc="lower left")
#     p = re.compile("num_cards=(\d+)")
#     cards = p.search(params["game_name"]).group(1)
#     br_value = uniform_br[cards]
#
#     for safe, df in data:
#         x = df[0]
#         print(x)
#         axes[0, 0].semilogy(x.loop, x["br"], label=f"br safe={safe}")
#         # axes[0, 0].semilogy(x.loop, x["br"].rolling(5).mean(), label=f"br safe={safe} (mean)")
#         axes[1, 0].semilogy(x.loop, x["avg_loss"], label=f"br safe={safe} (mean)")
#     #
#     df = data[0][1][0]
#     axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
#                         [br_value]*2, label=f"br against uniform")
#     axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
#                         [br_value*0.1]*2, label=f"br target 10x less")
#     axes[0, 0].semilogy([df.loop.head(1), df.loop.tail(1)],
#                         [br_value*0.2]*2, label=f"br target 5x less")
#
#     axes[0, 0].legend(loc="lower left")
#     axes[1, 0].legend(loc="lower left")



lazy_pipeline = process.make_lazy_pipeline(base_dir, select, pipeline)


# base_dir="./experiments/iigs_kn_oracle"
# pipeline = [
#     process.read,
#     process.concat("max_particles"),
#     process.concat("particle_selection"),
#
# ]
# layout = dict(
#     y=["game_name"],
#     x=["cfr_oracle_iterations", "cfr_iterations"]
# )
# cell_size = [1, 1]
# plot.display(lazy_pipeline, layout, cell_size, plot_oracle, translation_map)

# base_dir="./experiments/iigs_kn"
# pipeline = [
#     process.read,
#     process.average("seed"),
#     # process.concat("zero_sum_regression"),
#     process.concat("safe_resolving"),
# ]
# layout = dict(
#     x=["game_name"],
#     y=["particle_generation"]
# )
# cell_size = [3, 1]
plot.display(lazy_pipeline, layout, cell_size, plot_net, translation_map)


# plot.display(lazy_pipeline, layout, cell_size, plot_bootstrap, translation_map)
