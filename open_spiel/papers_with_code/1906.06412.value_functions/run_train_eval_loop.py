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
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=3,imp_info=True,points_order=descending)",
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)",
      "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)",
    ]
  elif param == "limit_particle_count":
    if context["game_name"] == "kuhn_poker":
      return list(range(1, 6+1))
    elif context["game_name"] == "leduc_poker":
      return list(range(1, 12+1))
    elif "num_cards=3" in context["game_name"]:
      return list(range(1, 6+1))
    elif "num_cards=4" in context["game_name"]:
      return list(range(1, 8+1))
    elif "num_cards=5" in context["game_name"]:
      return list(range(1, 10+1))
  elif param == "depth":
    if context["game_name"] == "kuhn_poker":
      return [3, 4]
    elif context["game_name"] == "leduc_poker":
      return [4, 6, 8]
    elif "num_cards=3" in context["game_name"]:
      return [1]
    elif "num_cards=4" in context["game_name"]:
      return [1, 2]
    elif "num_cards=5" in context["game_name"]:
      return [1, 2]
  elif param == "data_generation":
    return ["dl_cfr", "random"]


sweep.run_sweep(backend,
                backend_params,
                binary_path=f"{home}/experiments/train_eval_loop",
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    cfr_oracle_iterations=100,
                    num_loops=1000,
                    use_bandits_for_cfr="RegretMatchingPlus",
                    trunk_eval_iterations="1,2,5,10,20,50,100",
                    train_batches=64,
                    num_trunks=400,
                    batch_size=64,
                    prob_pure_strat=0.1,
                    shuffle_input="true",
                    shuffle_output="false"
                ),
                comb_params=[
                  "game_name",
                  "depth",
                  "data_generation",
                  "limit_particle_count",
                ],
                comb_param_fn=param_fn)
