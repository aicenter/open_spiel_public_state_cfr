import io
import itertools
import os
import subprocess
import sys
from functools import reduce
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

FLAGS = flags.FLAGS
flags.DEFINE_bool("with_results", False, "Print actual experiment results, "
                                         "do not just parse them.")

_BUILD_PATH = os.environ.get("CMAKE_BINARY_DIR") or "../../../build"
_BINARY_PATH = f"{_BUILD_PATH}/papers_with_code/1906.06412" \
               f".value_functions/train_eval_loop"

_BASE_ARGS = dict(
    game_name="kuhn_poker",
    depth=3,
    train_batches=32,
    batch_size=32,
    num_loops=100,
    cfr_oracle_iterations=100,
    trunk_expl_iterations=100,
    num_layers=5,
    num_width=5,
    num_inputs_regression=8,
    replay_size=20,
    seed=0,
    use_bandits_for_cfr="RegretMatchingPlus",
    exp_init="trunk_random",
    prob_pure_strat=0.1,
    prob_fully_mixed=0.05,
    max_particles=-1,
    save_values_policy="current"
)

_TEST_GAMES = [
  dict(game_name="kuhn_poker", depth=3, replay_size=2),
  dict(game_name="kuhn_poker", depth=4, replay_size=1),
  dict(game_name="goofspiel(players=2,num_cards=3,imp_info=True,"
                 "points_order=descending)", depth=1, replay_size=3,
       num_inputs_regression=9),
]

_EXP_INIT = [
  dict(exp_init="trunk_dlcfr"),
  dict(exp_init="trunk_random")
]

_VALUE_NETS = [
  dict(arch="positional_vf"),
  dict(arch="particle_vf"),
]

_REFERENCE_KUHN_DLCFR_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0457113,0.0694444,0.0133747,0.0238497
                     1,0.0111931,0.0694444,0.00474906,0.0199332
                     2,0.00680468,0.0694444,0.025,0.0277778
                     3,0.00619676,0.0694444,0.025,0.0277778
                     4,0.00586771,0.0694444,0.025,0.0277778
                     5,0.00569196,0.0694444,0.0231044,0.0254743
                     6,0.00529815,0.0694444,0.0321844,0.0306574
                     7,0.00534635,0.0694444,0.0108332,0.00620486
                     8,0.00533413,0.0694444,0.01791,0.00427941
                     9,0.00525978,0.0694444,0.0132054,0.00267397""",
    particle_vf  ="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0417761,0.0694444,0.160185,0.166667
                     1,0.00898707,0.0694444,0.00170061,0.0054617
                     2,0.00621538,0.0694444,0.0129145,0.0393235
                     3,0.00658145,0.0694444,0.0195157,0.00711847
                     4,0.00622934,0.0694444,0.0182836,0.0138636
                     5,0.00631172,0.0694444,0.0117317,0.000321979
                     6,0.00593974,0.0694444,0.0116007,0.00505571
                     7,0.00598633,0.0694444,0.0124723,0.0039066
                     8,0.00606983,0.0694444,0.0127753,0.000223347
                     9,0.00615289,0.0694444,0.0145473,0.00600354""",
)

_REFERENCE_KUHN_DLCFR_EXPL = """trunk_iter,expl
                                1,0.0694444
                                5,0.0370633
                                10,0.00569295
                                """

_REFERENCE_KUHN_RANDOM_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0570378,0.0694444,0.025,0.0277778
                     1,0.0121155,0.0694444,0.025,0.0277778
                     2,0.00910514,0.0694444,0.025,0.0277778
                     3,0.00716077,0.0694444,0.0335125,0.037701
                     4,0.00504085,0.0694444,0.0238044,0.0223743
                     5,0.00429464,0.0694444,0.0112695,0.0148322
                     6,0.00333418,0.0694444,0.00934847,0.0125359
                     7,0.00311386,0.0694444,0.00712564,0.00893604
                     8,0.00295046,0.0694444,0.00679372,0.00266394
                     9,0.00290238,0.0694444,0.010251,0.0040652""",
    particle_vf=  """loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0511106,0.0694444,0.00848881,0.0109963
                     1,0.0101088,0.0694444,0.025,0.0277778
                     2,0.00873095,0.0694444,0.00810304,0.00271497
                     3,0.00774862,0.0694444,0.00399664,0.00159504
                     4,0.00673806,0.0694444,0.00362019,0.00149238
                     5,0.00671607,0.0694444,0.00916075,0.0169858
                     6,0.00664004,0.0694444,0.0253521,0.0324042
                     7,0.00627518,0.0694444,0.0308089,0.0251733
                     8,0.00600766,0.0694444,0.0376558,0.0285024
                     9,0.00617622,0.0694444,0.0393137,0.0253787"""
)


def df_from_lines(lines: Union[str, List[str]], **kwargs) -> pd.DataFrame:
  return pd.read_csv(io.StringIO(lines), **kwargs)


def read_metric(buffer, tag, comment="# ", single_occurrence=True,
                parser=df_from_lines):
  buffer.seek(0)
  data_lines = []
  tag_opened = False
  for line in buffer.readlines():
    if f"</{tag}>" in line:
      if single_occurrence:
        break
      else:
        tag_opened = False

    if tag_opened and line.startswith(comment):
      data_lines.append(line[len(comment):])

    if f"<{tag}>" in line:
      tag_opened = True

  # print(data_lines)
  data = "\n".join(data_lines)
  return parser(data)


def metric_avg_loss(buffer) -> pd.DataFrame:
  buffer.seek(0)
  return pd.read_csv(buffer, comment="#", skip_blank_lines=True)


def metric_ref_expls(buffer) -> pd.DataFrame:
  """
  Read reference exploitabilities as given by DL-CFR iterations. 
  """
  return read_metric(buffer, "ref_expl")


def read_experiment_results_from_shell(cmd_args: Dict[str, Any],
                                       *read_metrics):
  cmd = [_BINARY_PATH] + [f"--{k}={v}" for k, v in cmd_args.items()]
  with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE) as proc:
    output, error = proc.communicate()
    if proc.returncode == 0:
      with io.StringIO(output.decode()) as buffer:
        if FLAGS.with_results:
          for line in buffer.readlines():
            print(line, end="")
            sys.stdout.flush()

        try:
          results = [metric(buffer) for metric in read_metrics]
        except Exception as e:
          buffer.seek(0)
          print("Buffer:", "--------------", sep="\n")
          print("".join(buffer.readlines()))
          raise e

    else:
      message = ("Shell command returned non-zero exit status: {0}\n\n"
                 "Command was:\n{1}\n\n"
                 "Standard error was:\n{2}")
      raise IOError(message.format(proc.returncode, cmd, error.decode()))

  return results


def dict_prod(*iterables):
  for items in itertools.product(*iterables):
    yield reduce(lambda x, y: {**x, **y}, items)


class VFTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(_VALUE_NETS)
  def test_kuhn_eval_iters_dlcfr_regression(self, **arch_spec):
    args = {**_BASE_ARGS, **arch_spec,
            **dict(num_loops=10, replay_size=200, exp_init="trunk_dlcfr",
                   num_layers=3, num_width=3, trunk_expl_iterations="1,5,10")}
    actual_eval, actual_expl = read_experiment_results_from_shell(
        args, metric_avg_loss, metric_ref_expls)

    arch = arch_spec["arch"]
    expected_eval = df_from_lines(_REFERENCE_KUHN_DLCFR_EVAL[arch])
    expected_expl = df_from_lines(_REFERENCE_KUHN_DLCFR_EXPL)

    np.testing.assert_allclose(actual_eval.values, expected_eval.values,
                               atol=1e-6)
    np.testing.assert_allclose(actual_expl.values, expected_expl.values,
                               atol=1e-6)

  @parameterized.parameters(_VALUE_NETS)
  def test_kuhn_eval_iters_random_regression(self, **arch_spec):
    args = {**_BASE_ARGS, **arch_spec,
            **dict(num_loops=10, replay_size=200, num_layers=3, num_width=3,
                   exp_init="trunk_random", trunk_expl_iterations="1,5,10")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    arch = arch_spec["arch"]
    expected_eval = df_from_lines(_REFERENCE_KUHN_RANDOM_EVAL[arch])
    np.testing.assert_allclose(actual_eval.values, expected_eval.values,
                               atol=1e-6)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_one_trunk(self, **game_spec):
    replay_size = game_spec["replay_size"]
    num_trunks = 1
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=20, batch_size=-1,
                   exp_init="trunk_random",
                   replay_size=replay_size * num_trunks,
                   trunk_expl_iterations="")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    actual_loss = actual_eval["avg_loss"].values.min()
    self.assertLess(actual_loss, 1e-8)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_two_trunks(self, **game_spec):
    replay_size = game_spec["replay_size"]
    num_trunks = 2
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=60, batch_size=-1,
                   learning_rate=0.01, lr_decay=0.99,
                   exp_init="trunk_random",
                   replay_size=replay_size * num_trunks,
                   trunk_expl_iterations=0)}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    actual_loss = actual_eval["avg_loss"].values.min()
    self.assertLess(actual_loss, 1e-6)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_imitate_dlcfr_iterations(self, **game_spec):
    if game_spec["game_name"] == "kuhn_poker" and game_spec["depth"] == 4:
      # Skip this setting, as getting the precise DL-CFR iterations is
      # difficult: the iterations are sensitive to the smallest changes in
      # values.
      return

    num_iters = 3
    replay_size = game_spec["replay_size"]
    args = {**_BASE_ARGS, **game_spec,
            # Run just 1 loop, as per-loop evals are the most expensive part.
            **dict(num_loops=10, train_batches=100,
                   batch_size=-1, exp_init="trunk_dlcfr",
                   replay_size=replay_size * num_iters,
                   trunk_expl_iterations=",".join(str(i) for i in range(1, num_iters + 1))
                   )}
    expected_expl, actual_eval = read_experiment_results_from_shell(
        args, metric_ref_expls, metric_avg_loss)
    expl_cols = [f"expl[{i}]" for i in range(1, num_iters+1)]
    actual_expl = actual_eval.tail(1)[expl_cols]
    np.testing.assert_allclose(actual_expl.values.flatten(),
                               expected_expl["expl"].values.flatten(),
                               atol=5e-3)

  @parameterized.parameters(_TEST_GAMES)
  def test_pbs_iterations_identical(self, **game_spec):
      args = dict(
          arch="particle_vf", batch_size="1", cfr_oracle_iterations="100",
          depth="7", device="cpu", exp_init="pbs_random",
          game_name="leduc_poker", num_inputs_regression="max",
          num_layers=5, num_loops=1, num_width=5, prob_pure_strat=0.1,
          replay_size=1, seed=0, train_batches=256,
          trunk_expl_iterations="1,5,10,50,100",
          use_bandits_for_cfr="RegretMatchingPlus",
          # Upper bounds all of the test games
          max_particles=30
      )
      actual_pbs_random, = read_experiment_results_from_shell(
          {**args, **game_spec, "exp_init": "pbs_random"}, metric_avg_loss)
      actual_sparse_pbs_random, = read_experiment_results_from_shell(
          {**args, **game_spec, "exp_init": "sparse_pbs_random"}, metric_avg_loss)

      np.testing.assert_allclose(actual_pbs_random.values.flatten(),
                                 actual_sparse_pbs_random.values.flatten(),
                                 atol=1e-4)

  # TODO sparse random pbs != random pbs without enough particles.
  #      fit one,two  exp replays.

  def test_bootstrap_leduc(self):
      args = dict(
          arch="particle_vf", batch_size="1",
          cfr_oracle_iterations="100", depth="7", exp_init="bootstrap",
          exp_loop="bootstrap", exp_loop_new="2", exp_update_size=-1,
          game_name="leduc_poker", num_inputs_regression="max", num_layers="5",
          num_loops="18", bootstrap_from_move=10,
          num_width="5", prob_pure_strat="0.1", replay_size="1",
          seed="0", max_particles=30, train_batches="1",
          trunk_expl_iterations="", use_bandits_for_cfr="RegretMatchingPlus")
      actual_bootstrap, = read_experiment_results_from_shell(args, metric_avg_loss)
      # Just check the bootstrap runs.
      self.assertEqual(actual_bootstrap.values.shape[0], 36)

if __name__ == "__main__":
  absltest.main()
