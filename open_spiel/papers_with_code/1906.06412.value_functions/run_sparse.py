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
experiment_name = "paper_sparse"

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
      return [3]
  elif param == "sparse_roots_depth":
    if context["game_name"] == "leduc_poker":
      return [5]
    elif "num_cards=5" in context["game_name"]:
      return [2]
  elif param == "support_threshold":
    if context["game_name"] == "leduc_poker":
      return [
        "0.00000000",  # 1.0
        "0.00031032",  # 0.9
        "0.00037930",  # 0.8
        "0.00063324",  # 0.7
        "0.00093430",  # 0.6
        "0.00116098",  # 0.5
        "0.00145379",  # 0.4
        "0.00219816",  # 0.3
        "0.00261279",  # 0.2
        "0.00477722",  # 0.1
        # "0.00562000",  # 2 pl histories -- run separately and use prune_chance_histories
      ]
    elif "num_cards=5" in context["game_name"]:
      return [
        "0.0000000",  # 1.0
        "0.0001614",  # 0.9
        "0.0010183",  # 0.8
        "0.0013163",  # 0.7
        "0.0032164",  # 0.6
        "0.0040236",  # 0.5
        "0.0045824",  # 0.4
        "0.0078920",  # 0.3
        "0.0107214",  # 0.2
        "0.0181064",  # 0.1
        "0.6150000",  # 1 history.
      ]
  elif param == "seed":
    return list(range(10))


sweep.run_sweep(backend,
                backend_params,
                binary_path=f"{home}/experiments/tel_sparse",
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    use_bandits_for_cfr="RegretMatchingPlus",
                    cfr_oracle_iterations=100,
                    num_loops=512,
                    trunk_expl_iterations="100",
                    train_batches=256,
                    num_trunks=1000,
                    batch_size=64,
                    shuffle_input="true",
                    shuffle_output="true",
                    data_generation="random",
                    prob_pure_strat=0.1,
                    prune_chance_histories="false",
                ),
                comb_params=[
                  "game_name",           # 2 params
                  "depth",               # 1 param
                  "sparse_roots_depth",  # 1 param
                  "support_threshold",   # 11 or 10 params
                  "seed"                 # 10 params
                ],
                comb_param_fn=param_fn)


# Special case for Leduc -- there is a number of highest reachable histories
# that are only for chance player -- ie there is no player infostate and nothing
# to fix. We omit these chance histories to have something to fixate.
def special_case(param, context):
  if param == "game_name":
    return ["leduc_poker"]
  elif param == "depth":
    return [7]
  elif param == "sparse_roots_depth":
    return [5]
  elif param == "support_threshold":
    return ["0.00562000"]  # 2 pl histories -- run separately and use prune_chance_histories
  elif param == "seed":
    return list(range(10))


sweep.run_sweep(backend,
                backend_params,
                binary_path=f"{home}/experiments/tel_sparse",  # git 4f8973f1
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(
                    use_bandits_for_cfr="RegretMatchingPlus",
                    cfr_oracle_iterations=100,
                    num_loops=512,
                    trunk_expl_iterations="100",
                    train_batches=256,
                    num_trunks=1000,
                    batch_size=64,
                    shuffle_input="true",
                    shuffle_output="true",
                    data_generation="random",
                    prob_pure_strat=0.1,
                    prune_chance_histories="true",  # <--- !!
                ),
                comb_params=[
                  "game_name",           # 1 params
                  "depth",               # 1 param
                  "sparse_roots_depth",  # 1 param
                  "support_threshold",   # 1 param
                  "seed"                 # 10 params
                ],
                comb_param_fn=special_case)
