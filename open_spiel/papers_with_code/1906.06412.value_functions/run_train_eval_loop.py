import sys
import grid_experiments.run as run

# home = "/storage/praha1/home/sustrmic"
# backend = "meta"
# backend_params = [
#   "-l", "select=1:ncpus=1:mem=2gb",
#   "-l", "walltime=08:00:00",
#   "-v", f"LD_LIBRARY_PATH="
#         f"{home}/open_spiel/open_spiel/libtorch/libtorch/lib/:"
#         f"{home}/open_spiel/open_spiel/ortools/lib/",
# ]
# binary_path = f"{home}/experiments/train_eval_loop"

home = "/home/michal/Code/open_spiel/open_spiel/papers_with_code/1906.06412.value_functions"
backend = "dryrun"
backend_params = []
binary_path = f"{home}/experiments/train_eval_loop"

# -- Paper experiment: VF comparison -------------------------------------------

def vf_comparison():
  vf_base_params = dict(
      device="cpu",
      use_bandits_for_cfr="RegretMatchingPlus",
      cfr_oracle_iterations=100,
      num_loops=1024,
      trunk_expl_iterations="100",
      train_batches=64,
      replay_size=10000,
      batch_size=64,
      prob_pure_strat=0.1,
      prob_fully_mixed=0.05,
      num_layers=5,
      num_width=5,
      shuffle_input_output="true",
      # Holds for both games by coincidence!
      max_particles=-1,
      num_inputs_regression=-1,
  )

  def param_fn(param, context):
    if param == "arch":
      return ["positional_vf", "particle_vf"]
    elif param == "exp_init":
      return ["trunk_random", "pbs_random", "sparse_pbs_random"]
    elif param == "game_name":
      return ["leduc_poker",
              "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)"]
    elif param == "depth":
      if context["game_name"] == "leduc_poker":
        return [7]
      elif "num_cards=5" in context["game_name"]:
        return [2]
    elif param == "sparse_particles":
        if context["exp_init"] == "sparse_pbs_random":
            if context["game_name"] == "leduc_poker":
                return list(range(5, 31, 5))
            if "num_cards=5" in context["game_name"]:
                return list(range(20, 290, 25)) + [290]
        else:
            return [0]
    elif param == "seed":
      return list(range(10))

  run.sweep(backend, backend_params, binary_path,
            base_output_dir=f"{home}/experiments/vf_comparison",
            base_params=vf_base_params,
            comb_params=["arch", "exp_init", "game_name", "depth",
                         "sparse_particles", "seed"],
            comb_param_fn=param_fn)

def sparse_roots():
  base_params = dict(
      arch="particle_vf",
      use_bandits_for_cfr="RegretMatchingPlus",
      cfr_oracle_iterations=100,
      num_loops=512,
      sparse_expl_iterations="100",
      train_batches=256,
      replay_size=50000,
      batch_size=64,
      shuffle_input_output="true",
      exp_init="trunk_random",
      prob_pure_strat=0.1,
  )

  def param_fn(param, context):
    if param == "game_name":
      return ["leduc_poker",
              "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)"]
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
    elif param == "sparse_support_threshold":
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

  run.sweep(backend, backend_params, binary_path,
            base_output_dir=f"{home}/experiments/sparse_roots",
            base_params={**base_params,
                         "sparse_prune_chance_histories": "false"},
            comb_params=["game_name", "depth", "sparse_roots_depth",
                         "sparse_support_threshold", "seed"],
            comb_param_fn=param_fn)

  # Special case for Leduc -- there is a number of highest reachable histories
  # that are only for chance player -- ie there is no player infostate and
  # nothing to fix. We omit these chance histories to have something to fixate.
  def special_case(param, context):
    if param == "game_name":
      return ["leduc_poker"]
    elif param == "depth":
      return [7]
    elif param == "sparse_roots_depth":
      return [5]
    elif param == "sparse_support_threshold":
      # 2 pl histories -- run separately and use prune_chance_histories
      return ["0.00562000"]
    elif param == "seed":
      return list(range(10))

  run.sweep(backend, backend_params, binary_path,
            base_output_dir=f"{home}/experiments/sparse_roots",
            base_params={**base_params,
                         "sparse_prune_chance_histories": "true"},
            comb_params=["game_name", "depth", "sparse_roots_depth",
                         "sparse_support_threshold", "seed"],
            comb_param_fn=special_case)

def training_dynamics():
  base_params = dict(
    arch="particle_vf",
    batch_size="64",
    cfr_oracle_iterations="100",
    depth="7",
    device="cpu",
    exp_init="pbs_random",
    exp_loop="nothing",
    exp_loop_new="32",
    exp_update="128",
    game_name="leduc_poker",
    max_particles="-1",
    num_inputs_regression="-1",
    num_layers="5",
    num_loops="2048",
    num_width="5",
    prob_pure_strat="0.1",
    replay_size="2048",
    seed="0",
    shuffle_input_output="true",
    sparse_particles="30",
    train_batches="32",
    trunk_expl_iterations="100",
    use_bandits_for_cfr="RegretMatchingPlus",
    learning_rate="0.001",
    optimizer="adam",
    replay_visits_window="32",
  )

  def param_fn(param, context):
    if param == "exp_loop_new":
      return [16, 32, 64, 128, 256, 512]
    elif param == "exp_update":
      return [64, 128, 512, 1024, 2048]

  run.sweep(backend, backend_params, binary_path,
            base_output_dir=f"{home}/experiments/training_dynamics",
            base_params=base_params,
            comb_params=["exp_loop_new", "exp_update"],
            comb_param_fn=param_fn)

def bootstraped_learning():
  loop_new = 512
  base_params = dict(
      arch="particle_vf",
      batch_size=64,
      bootstrap_reset_nn="true",
      cfr_oracle_iterations=100,
      exp_init="bootstrap",
      exp_loop="bootstrap",
      exp_loop_new=loop_new,
      exp_reset_nn="true",
      exp_update_size=-1,
      learning_rate="0.001",
      lr_decay=0.99,
      max_particles=-1,
      normalize_beliefs="true",
      num_inputs_regression=-1,
      num_layers=5,
      num_width=5,
      prob_pure_strat=0.1,
      replay_size=10000,
      replay_visits_window=10000,
      save_values_policy="average",
      shuffle_input_output="true",
      snapshot_loop="256",
      track_lr="true",
      track_time="true",
      train_batches=64,
      trunk_expl_iterations=100,
      use_bandits_for_cfr="RegretMatchingPlus",
      zero_sum_regression="true",
  )
  def param_fn(param, context):
      if param == "game_name":
          return ["leduc_poker",
                  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)"
                 ]
      elif param == "depth":
          if context["game_name"] == "leduc_poker":
              return [7]
          elif "num_cards=5" in context["game_name"]:
              return [2]
      elif param == "num_loops":
          if context["game_name"] == "leduc_poker":
              return [9*loop_new]
          elif "num_cards=5" in context["game_name"]:
              return [3*loop_new]
      elif param == "seed":
          return list(range(5))

  run.sweep(backend, backend_params, binary_path,
            base_output_dir=f"{home}/experiments/bootstraped_learning",
            base_params=base_params,
            comb_params=["game_name", "depth", "num_loops", "seed"],
            comb_param_fn=param_fn,
            save_snapshot=True)



def snapshot_pbs_training():
    base_params = dict(
        arch="particle_vf",
        batch_size="64",
        cfr_oracle_iterations="100",
        depth="7",
        device="cpu",
        exp_init="pbs_random",
        learning_rate="0.001",
        max_particles="-1",
        num_inputs_regression="-1",
        num_layers="5",
        num_loops="512",
        num_width="5",
        optimizer="adam",
        prob_pure_strat="0.1",
        prob_fully_mixed=0.05,
        replay_size="10000",
        shuffle_input_output="true",
        train_batches="64",
        trunk_expl_iterations="100",
        use_bandits_for_cfr="RegretMatchingPlus",
        snapshot_loop="64",
    )

    def param_fn(param, context):
        if param == "game_name":
            return ["leduc_poker",
                    "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)"]
        elif param == "depth":
            if context["game_name"] == "leduc_poker":
                return [7]
            elif "num_cards=5" in context["game_name"]:
                return [2]
        elif param == "save_values_policy":
            return ["current", "average"]
        elif param == "zero_sum_regression":
            return ["true", "false"]
        elif param == "seed":
            return list(range(5))

    run.sweep(backend, backend_params, binary_path,
              base_output_dir=f"{home}/experiments/snapshot_pbs_training",
              base_params=base_params,
              comb_params=["game_name", "depth", "save_values_policy",
                           "zero_sum_regression", "seed"],
              comb_param_fn=param_fn,
              save_snapshot=True)

EXPERIMENTS_ = dict(vf_comparison=vf_comparison,
                    sparse_roots=sparse_roots,
                    training_dynamics=training_dynamics,
                    bootstraped_learning=bootstraped_learning,
                    snapshot_pbs_training=snapshot_pbs_training
                    )

if __name__ == '__main__':
  for arg in sys.argv:
    if arg in EXPERIMENTS_:
      EXPERIMENTS_[arg]()
