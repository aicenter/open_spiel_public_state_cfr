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
    trunk_eval_iterations=100,
    num_layers=5,
    num_width=5,
    num_trunks=10,
    seed=0,
    use_bandits_for_cfr="RegretMatchingPlus",
    data_generation="random",
    prob_pure_strat=0.1,
    prob_fully_mixed=0.05,
)

_TEST_GAMES = [
  dict(game_name="kuhn_poker", depth=3),
  dict(game_name="kuhn_poker", depth=4),
  # dict(game_name="leduc_poker", depth=5),
  dict(game_name="goofspiel(players=2,num_cards=3,imp_info=True,"
                 "points_order=descending)", depth=1),
]

_DATA_GENERATION = [
  dict(data_generation="dl_cfr"),
  dict(data_generation="random")
]

_VALUE_NETS = [
  dict(arch="particle_vf"),
  dict(arch="positional_vf")
]

_REFERENCE_KUHN_DLCFR_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.105115,0.0694444,0.0509253,0.0542922
                     1,0.0274438,0.0694444,0.0509253,0.0542922
                     2,0.00937781,0.0694444,0.0141567,0.0206512
                     3,0.00830411,0.0694444,0.0171355,0.00288411
                     4,0.00712066,0.0694444,0.0196713,0.0227157
                     5,0.00684092,0.0694444,0.011997,0.0136929
                     6,0.006719,0.0694444,0.00624333,0.00670182
                     7,0.00628849,0.0694444,0.00682326,0.00374965
                     8,0.00639248,0.0694444,0.00636514,0.0148459
                     9,0.00565729,0.0694444,0.00650681,0.00784207""",
    particle_vf  ="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.08465,0.0694444,0.00463028,0.00126331
                     1,0.045953,0.0694444,0.0506477,0.0524562
                     2,0.0117353,0.0694444,0.0900613,0.102181
                     3,0.00991366,0.0694444,0.0294029,0.021842
                     4,0.00843621,0.0694444,0.008172,0.00456634
                     5,0.0082433,0.0694444,0.00434396,0.0148551
                     6,0.00799375,0.0694444,0.00773945,0.00945485
                     7,0.0077094,0.0694444,0.0146659,0.00838035
                     8,0.00808868,0.0694444,0.0102013,0.0308031
                     9,0.00758692,0.0694444,0.0113289,0.0244972""",
)


_REFERENCE_KUHN_DLCFR_EXPL = """trunk_iter,expl
                                1,0.0694444
                                5,0.0371064
                                10,0.0056756"""

_REFERENCE_KUHN_RANDOM_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.113564,0.0694444,0.0509253,0.0542922
                     1,0.0224779,0.0694444,0.044374,0.0525055
                     2,0.00940264,0.0694444,0.0351531,0.0499907
                     3,0.0076646,0.0694444,0.0215225,0.0454593
                     4,0.00558716,0.0694444,0.0169912,0.0384209
                     5,0.00533079,0.0694444,0.0198618,0.0336662
                     6,0.00469655,0.0694444,0.0207966,0.025176
                     7,0.00424157,0.0694444,0.0211623,0.0139415
                     8,0.00402533,0.0694444,0.0257171,0.0175642
                     9,0.00372721,0.0694444,0.0258115,0.0204999""",
    particle_vf=  """loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.0968018,0.0694444,0.00463028,0.00126331
                     1,0.0577754,0.0694444,0.034451,0.0495971
                     2,0.0158033,0.0694444,0.00463028,0.00126331
                     3,0.011827,0.0694444,0.00283357,0.000772691
                     4,0.00956443,0.0694444,0.00314458,0.000857512
                     5,0.00973787,0.0694444,0.00339268,0.000925175
                     6,0.00940136,0.0694444,0.00352629,0.000961614
                     7,0.00918091,0.0694444,0.00345693,0.000942698
                     8,0.00960886,0.0694444,0.00334956,0.000913416
                     9,0.00925711,0.0694444,0.00323938,0.000883365"""
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
            **dict(num_loops=10, num_trunks=100, data_generation="dl_cfr",
                   num_layers=3, num_width=3, trunk_eval_iterations="1,5,10")}
    actual_eval, actual_expl = read_experiment_results_from_shell(
        args, metric_avg_loss, metric_ref_expls)

    arch = arch_spec["arch"]
    expected_eval = df_from_lines(_REFERENCE_KUHN_DLCFR_EVAL[arch])
    expected_expl = df_from_lines(_REFERENCE_KUHN_DLCFR_EXPL)

    np.testing.assert_array_equal(actual_eval.values, expected_eval.values)
    np.testing.assert_array_equal(actual_expl.values, expected_expl.values)

  @parameterized.parameters(_VALUE_NETS)
  def test_kuhn_eval_iters_random_regression(self, **arch_spec):
    args = {**_BASE_ARGS, **arch_spec,
            **dict(num_loops=10, num_trunks=100, num_layers=3, num_width=3,
                   data_generation="random", trunk_eval_iterations="1,5,10")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    arch = arch_spec["arch"]
    expected_eval = df_from_lines(_REFERENCE_KUHN_RANDOM_EVAL[arch])
    np.testing.assert_array_equal(actual_eval.values, expected_eval.values)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_one_sample(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=20, batch_size=-1, data_generation="random",
                   num_trunks=1, trunk_eval_iterations="")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    actual_loss = actual_eval["avg_loss"].values.min()
    self.assertLess(actual_loss, 1e-8)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_two_samples(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=60, batch_size=-1, data_generation="random",
                   learning_rate=0.01, lr_decay=0.99,
                   num_trunks=2, trunk_eval_iterations=0)}
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
    args = {**_BASE_ARGS, **game_spec,
            # Run just 1 loop, as per-loop evals are the most expensive part.
            **dict(num_loops=10, train_batches=100,
                   batch_size=-1, data_generation="dl_cfr",
                   num_trunks=num_iters,
                   trunk_eval_iterations=",".join(str(i) for i in range(1, num_iters + 1))
                   )}
    expected_expl, actual_eval = read_experiment_results_from_shell(
        args, metric_ref_expls, metric_avg_loss)
    expl_cols = [f"expl[{i}]" for i in range(1, num_iters+1)]
    actual_expl = actual_eval.tail(1)[expl_cols]

    np.testing.assert_allclose(actual_expl.values.flatten(),
                               expected_expl["expl"].values.flatten(),
                               atol=5e-3)

if __name__ == "__main__":
  absltest.main()
