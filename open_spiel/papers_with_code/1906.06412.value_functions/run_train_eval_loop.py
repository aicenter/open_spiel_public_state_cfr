import grid_plot.sweep as sweep

# home = "/storage/praha1/home/sustrmic/"
# backend = "meta",
# backend_params = [
#                    "-l", "select=1:ncpus=2:mem=4gb",
#                    "-l", "walltime=08:00:00",
#                    "-v", f"LD_LIBRARY_PATH="
#                          f"{home}/open_spiel/open_spiel/libtorch/libtorch/lib/:"
#                          f"{home}/open_spiel/open_spiel/ortools/lib/",
#                  ],

home = "/home/michal/Code/open_spiel/open_spiel/papers_with_code/1906.06412.value_functions"
backend = "parallel"
backend_params = []
experiment_name = "limit_particles_kuhn"


def param_fn(param, context):
  if param == "game_name":
    return [
      # "kuhn_poker",
      "leduc_poker",
      # "goofspiel(players=2,num_cards=3,imp_info=True,points_order=descending)",
      # "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)",
      "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)",
    ]
  elif param == "limit_particle_count":
    if context["game_name"] == "leduc_poker":
      return list(range(1, 12+1)) + [-1]
    elif "num_cards=5" in context["game_name"]:
      return list(range(10, 80+1, 10)) + [-1]
  elif param == "depth":
    if context["game_name"] == "leduc_poker":
      return [7]
    elif "num_cards=5" in context["game_name"]:
      return [3]
  elif param == "sparse_roots_depth":
    if context["game_name"] == "leduc_poker":
      return [5]
    elif "num_cards=5" in context["game_name"]:
      return [2]
  elif param == "support_threshold":
    if context["game_name"] == "leduc_poker":
      return [
        -1,
        0.,
        0.0183325 + 1e-4,
        0.0355551 + 1e-4,
        0.0550263 + 1e-4,
        0.0552292 + 1e-4,
        0.0723499 + 1e-4,
        0.102431  + 1e-4,
        0.149214  + 1e-4,
        0.150184  + 1e-4,
        0.15372   + 1e-4,
        0.247954  + 1e-4,
        0.24823   + 1e-4,
        0.258139  + 1e-4,
      ]
    elif "num_cards=5" in context["game_name"]:
      return [
       -1,
        0.,
        0.0132248 + 1e-4,
        0.0549339 + 1e-4,
        0.0827586 + 1e-4,
        0.106355  + 1e-4,
        0.137484  + 1e-4,
        0.147508  + 1e-4,
      ]


sweep.run_sweep("dryrun",
                backend_params,
                binary_path=f"{home}/experiments/train_eval_loop",
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    cfr_oracle_iterations=100,
                    num_loops=5000,
                    use_bandits_for_cfr="RegretMatchingPlus",
                    trunk_eval_iterations="1,2,5,10,20,50,100",
                    train_batches=8,
                    num_trunks=1000,
                    batch_size=64,
                    prob_pure_strat=0.1,
                    shuffle_input="true",
                    shuffle_output="true",
                    limit_particle_count=-1,
                    data_generation="random"
                ),
                comb_params=[
                  "game_name",
                  "depth",
                  "sparse_roots_depth",
                  "support_threshold",
                ],
                comb_param_fn=param_fn)
