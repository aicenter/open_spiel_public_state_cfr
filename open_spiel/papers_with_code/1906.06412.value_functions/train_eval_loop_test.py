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
    max_particles=-1
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
                     6,0.00549846,0.0694444,0.00510109,0.00697373
                     7,0.00525415,0.0694444,0.00362915,0.00345322
                     8,0.00541608,0.0694444,0.0212175,0.0135413
                     9,0.00495693,0.0694444,0.0174392,0.00799725""",
    particle_vf  ="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0568817,0.0694444,0.12671,0.155769
                     1,0.0115792,0.0694444,0.132799,0.157429
                     2,0.00828187,0.0694444,0.0249996,0.0270198
                     3,0.00946522,0.0694444,0.00847591,0.00231151
                     4,0.00819217,0.0694444,0.00766577,0.0336452
                     5,0.00814279,0.0694444,0.0185633,0.00889089
                     6,0.00808118,0.0694444,0.00794043,0.0197588
                     7,0.00768808,0.0694444,0.00682496,0.00186125
                     8,0.00805633,0.0694444,0.00835182,0.0030498
                     9,0.00735389,0.0694444,0.00721325,0.00218905""",
)


_REFERENCE_KUHN_DLCFR_EXPL = """trunk_iter,expl
                                1,0.0694444
                                5,0.0371064
                                10,0.0056756"""

_REFERENCE_KUHN_RANDOM_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.056628,0.0694444,0.0249996,0.0270198
                     1,0.0122226,0.0694444,0.0249996,0.0270198
                     2,0.00887244,0.0694444,0.0249996,0.0270198
                     3,0.00692775,0.0694444,0.0346151,0.0305989
                     4,0.0047296,0.0694444,0.037505,0.0342184
                     5,0.00430878,0.0694444,0.0356805,0.0321265
                     6,0.00364356,0.0694444,0.0322205,0.0246356
                     7,0.00315454,0.0694444,0.0269972,0.0175009
                     8,0.00301205,0.0694444,0.0306483,0.0232433
                     9,0.00266443,0.0694444,0.0273472,0.0183365""",
    particle_vf=  """loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0652568,0.0694444,0.00360499,0.000983077
                     1,0.0138062,0.0694444,0.0155651,0.00475801
                     2,0.0108717,0.0694444,0.00445236,0.00121418
                     3,0.00988,0.0694444,0.00195041,0.00316292
                     4,0.00738435,0.0694444,0.012771,0.013453
                     5,0.00724109,0.0694444,0.0303554,0.0303466
                     6,0.00681276,0.0694444,0.0274885,0.0282416
                     7,0.0062517,0.0694444,0.0465534,0.0485468
                     8,0.00543609,0.0694444,0.0476587,0.0490722
                     9,0.00460068,0.0694444,0.0478753,0.0440243"""
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

  # TODO sparse random pbs == random pbs with enough particles.
  #      sparse random pbs != random pbs without enough particles.
  #      fit one,two  exp replays.

if __name__ == "__main__":
  absltest.main()
