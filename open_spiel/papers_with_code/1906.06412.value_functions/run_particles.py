import sweep as sweep

home = "/storage/praha1/home/sustrmic"
backend = "meta"
backend_params = [
  "-l", "select=1:ncpus=1:mem=4gb",
  "-l", "walltime=08:00:00",
  "-v", f"LD_LIBRARY_PATH="
        f"{home}/open_spiel/open_spiel/libtorch/libtorch/lib/:"
        f"{home}/open_spiel/open_spiel/ortools/lib/",
]
experiment_name = "paper_particles"


def param_fn(param, context):
  if param == "game_name":
    return [
      "leduc_poker",
      "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)",
    ]
  elif param == "depth":
    if context["game_name"] == "leduc_poker":
      return [7]
    elif "num_cards=5" in context["game_name"]:
      return [2]
  elif param == "seed":
    return list(range(10))


sweep.run_sweep(backend,
                backend_params,
                binary_path=f"{home}/experiments/tel_particles",  # git 27a9dca6
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    use_bandits_for_cfr="RegretMatchingPlus",
                    cfr_oracle_iterations=100,
                    num_loops=2048,
                    trunk_expl_iterations="100",
                    train_batches=64,
                    num_trunks=1000,
                    batch_size=64,
                    data_generation="random",
                    prob_pure_strat=0.1,
                    shuffle_input="true",
                    shuffle_output="true",
                ),
                comb_params=[
                  "game_name",           # 2 params
                  "depth",               # 1 param
                  "seed"                 # 10 params
                ],
                comb_param_fn=param_fn)

