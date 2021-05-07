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
                     0,0.0477652,0.0694444,0.0138837,0.0239882
                     1,0.0109696,0.0694444,0.0119315,0.0234558
                     2,0.00649812,0.0694444,0.0249996,0.0270198
                     3,0.00645481,0.0694444,0.0249996,0.0270198
                     4,0.00574644,0.0694444,0.0249996,0.0270198
                     5,0.00560573,0.0694444,0.0249996,0.0270198
                     6,0.00549846,0.0694444,0.00510111,0.00697368
                     7,0.00525415,0.0694444,0.00362916,0.00345323
                     8,0.00541608,0.0694444,0.0212175,0.0135413
                     9,0.00495693,0.0694444,0.0174392,0.00799724""",
    particle_vf  ="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0437316,0.0694444,0.122753,0.151127
                     1,0.00883338,0.0694444,0.0992095,0.118175
                     2,0.00623254,0.0694444,0.0249996,0.0270198
                     3,0.00701522,0.0694444,0.0125914,0.00343392
                     4,0.006131,0.0694444,0.0133602,0.0186378
                     5,0.00616746,0.0694444,0.0191045,0.00900116
                     6,0.00607027,0.0694444,0.00274036,0.0336344
                     7,0.00579142,0.0694444,0.0112654,0.00307229
                     8,0.00610097,0.0694444,0.00694628,0.0175147
                     9,0.0056572,0.0694444,0.00669134,0.00573536""",
)

_REFERENCE_KUHN_DLCFR_EXPL = """trunk_iter,expl
                                1,0.0694444
                                5,0.0371064
                                10,0.0056756"""

_REFERENCE_KUHN_RANDOM_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0570302,0.0694444,0.0249996,0.0270198
                     1,0.0121103,0.0694444,0.0249996,0.0270198
                     2,0.00910163,0.0694444,0.0249996,0.0270198
                     3,0.00715834,0.0694444,0.0334639,0.0366264
                     4,0.00504014,0.0694444,0.0237848,0.0216047
                     5,0.00429396,0.0694444,0.0113329,0.014882
                     6,0.00333315,0.0694444,0.00927208,0.0125238
                     7,0.00311156,0.0694444,0.00699325,0.00894136
                     8,0.00294633,0.0694444,0.00670952,0.00257615
                     9,0.00289479,0.0694444,0.0103803,0.00426986""",
    particle_vf=  """loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0511091,0.0694444,0.00864576,0.0107971
                     1,0.0101048,0.0694444,0.0249996,0.0270198
                     2,0.008733,0.0694444,0.00828631,0.0022598
                     3,0.00776293,0.0694444,0.0040083,0.00109307
                     4,0.00674488,0.0694444,0.00362559,0.000988697
                     5,0.00671694,0.0694444,0.00860017,0.0163128
                     6,0.00664634,0.0694444,0.0241174,0.0305264
                     7,0.00627316,0.0694444,0.0316654,0.0251712
                     8,0.00599944,0.0694444,0.0379699,0.0286542
                     9,0.00618154,0.0694444,0.0393815,0.0261618"""
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
          game_name="leduc_poker", max_particles=-1, num_inputs_regression=-1,
          num_layers=5, num_loops=1, num_width=5, prob_pure_strat=0.1,
          replay_size=1, seed=0, shuffle_input_output="true",
          train_batches=256, trunk_expl_iterations="1,5,10,50,100",
          use_bandits_for_cfr="RegretMatchingPlus",
          # Upper bounds all of the test games
          sparse_particles=30
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
          game_name="leduc_poker", max_particles=-1,
          num_inputs_regression=-1, num_layers="5",
          num_loops="18", bootstrap_from_move=10,
          num_width="5", prob_pure_strat="0.1", replay_size="1",
          seed="0", shuffle_input_output="true", sparse_particles=30,
          train_batches="1", trunk_expl_iterations="",
          use_bandits_for_cfr="RegretMatchingPlus")
      actual_bootstrap, = read_experiment_results_from_shell(args, metric_avg_loss)
      # Just check the bootstrap runs.
      self.assertEqual(actual_bootstrap.values.shape[0], 36)

if __name__ == "__main__":
  absltest.main()
