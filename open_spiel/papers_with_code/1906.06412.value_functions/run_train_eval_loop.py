import sweep

# home = "/storage/praha1/home/sustrmic/"
# backend = "meta"
# backend_params = [
#                    "-l", "select=1:ncpus=2:mem=4gb",
#                    "-l", "walltime=08:00:00",
#                    "-v", f"LD_LIBRARY_PATH="
#                          f"{home}/open_spiel/open_spiel/libtorch/libtorch/lib/:"
#                          f"{home}/open_spiel/open_spiel/ortools/lib/",
#                  ]

home = "/home/sustr"
backend = "parallel"
backend_params = []
experiment_name = "randomization_and_bandit"

def param_fn(param, context):
  if param == "game_name":
    return [
      "kuhn_poker",
      # "leduc_poker",
      # "goofspiel(players=2,num_cards=3,imp_info=True,
      # points_order=descending)",
      # "goofspiel(players=2,num_cards=4,imp_info=True,
      # points_order=descending)",
      "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)",
    ]
  elif param == "use_bandits_for_cfr":
    return ["RegretMatchingPlus", "RMPlusWithEps"]
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
      return [2]
  elif param == "data_generation":
    return ["dl_cfr", "random"]
  elif param == "prob_pure_strat":
    if context["data_generation"] == "dl_cfr":
      return [0]
    else:
      return [0, 0.1, 0.2]



sweep.run_sweep(backend,
                backend_params,
                binary_path=f"{home}/experiments/train_eval_loop",
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    cfr_oracle_iterations=100,
                    num_loops=300,
                    trunk_eval_iterations="1,2,5,10,20,50,100",
                    train_batches=64,
                    num_trunks=400,
                    batch_size=-1,
                    num_layers=5,
                    num_width=6,
                ),
                comb_params=[
                  "game_name",
                  "use_bandits_for_cfr",
                  "depth",
                  "data_generation",
                  "prob_pure_strat"
                ],
                comb_param_fn=param_fn)
