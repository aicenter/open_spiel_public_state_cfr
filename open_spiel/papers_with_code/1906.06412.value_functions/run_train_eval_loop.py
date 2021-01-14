import collections
import grid_plot.sweep as sweep


def param_fn(param, context):
  if param == "game_name":
    return [
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)",
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)"]
  elif param == "depth":
    if context["game_name"] == "kuhn_poker":
      return [3, 4]
    elif context["game_name"] == "leduc_poker":
      return [4, 6, 8]
    elif "num_cards=3" in context["game_name"]:
      return [1]
    elif "num_cards=4" in context["game_name"]:
      return [1, 2]
  elif param == "data_generation":
    return ["dl_cfr", "random"]


sweep.run_sweep(backend="meta",
                backend_params=["-l", "select=1:ncpus=2:mem=4gb"
                                "-l", "walltime=02:00:00"],
                binary_path="train_eval_loop",
                base_output_dir="./eval_iters_refactor",
                base_params=collections.OrderedDict([
                  ("cfr_oracle_iterations", 100),
                  ("num_loops", 200),
                  ("use_bandits_for_cfr", "RegretMatchingPlus"),
                  ("trunk_eval_iterations", "1,2,5,10,20,50,100"),
                  ("train_batches", 32),
                  ("batch_size", 128),
                ]),
                comb_params=["game_name", "depth"],
                comb_param_fn=param_fn)
