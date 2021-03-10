import sweep as sweep

home = "/storage/praha1/home/sustrmic"
backend = "meta"
backend_params = [
  "-l", "select=1:ncpus=1:mem=2gb",
  "-l", "walltime=08:00:00",
  "-v", f"LD_LIBRARY_PATH="
        f"{home}/open_spiel/open_spiel/libtorch/libtorch/lib/:"
        f"{home}/open_spiel/open_spiel/ortools/lib/",
]
binary_path = f"{home}/experiments/train_eval_loop"
base_params = dict(
    use_bandits_for_cfr="RegretMatchingPlus",
    cfr_oracle_iterations=100,
    num_loops=2048,
    trunk_expl_iterations="100",
    train_batches=64,
    num_trunks=1000,
    batch_size=64,
    data_generation="random",
    prob_pure_strat=0.1,
    num_layers=5,
    num_width=5,
    shuffle_input_output="true",
)


# -- Paper experiment: VF comparison -------------------------------------------

def vf_comparison(param, context):
  if param == "arch":
    return ["positional_vf", "particle_vf"]
  elif param == "game_name":
    return ["leduc_poker",
            "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)"]
  elif param == "depth":
    if context["game_name"] == "leduc_poker":
      return [7]
    elif "num_cards=5" in context["game_name"]:
      return [2]
  elif param == "seed":
    return list(range(10))

sweep.run_sweep(backend, backend_params, binary_path,
                base_output_dir=f"{home}/experiments/vf_comparison",
                base_params=base_params,
                comb_params=["arch", "game_name", "depth", "seed"],
                comb_param_fn=vf_comparison)
