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
                     0,0.073733,0.0694444,0.12032,0.151394
                     1,0.0139825,0.0694444,0.116197,0.118997
                     2,0.00857352,0.0694444,0.00412703,0.00326955
                     3,0.00884073,0.0694444,0.00795371,0.00227775
                     4,0.00762462,0.0694444,0.0058992,0.00359951
                     5,0.00742138,0.0694444,0.00955802,0.0105363
                     6,0.00705953,0.0694444,0.0134583,0.000638691
                     7,0.00654629,0.0694444,0.004431,0.00533375
                     8,0.00667238,0.0694444,0.00935365,0.00492868
                     9,0.00605068,0.0694444,0.00984021,0.0045454""",
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
                     0,0.0854954,0.0694444,0.137002,0.157491
                     1,0.0171646,0.0694444,0.013046,0.0035579
                     2,0.0108129,0.0694444,0.00439164,0.00119762
                     3,0.00959847,0.0694444,0.00961244,0.00262208
                     4,0.00729548,0.0694444,0.00937807,0.00255816
                     5,0.00722822,0.0694444,0.0150662,0.00410946
                     6,0.00670416,0.0694444,0.0080709,0.00220166
                     7,0.00625631,0.0694444,0.0419772,0.027697
                     8,0.00492542,0.0694444,0.0495011,0.0412508
                     9,0.00434882,0.0694444,0.048805,0.0399541"""
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
            **dict(num_loops=10, num_trunks=100, num_layers=3, num_width=3,
                   data_generation="random", trunk_expl_iterations="1,5,10")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    arch = arch_spec["arch"]
    expected_eval = df_from_lines(_REFERENCE_KUHN_RANDOM_EVAL[arch])
    np.testing.assert_allclose(actual_eval.values, expected_eval.values,
                               atol=1e-6)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_one_sample(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=20, batch_size=-1, data_generation="random",
                   num_trunks=1, trunk_expl_iterations="")}
    actual_eval, = read_experiment_results_from_shell(args, metric_avg_loss)
    actual_loss = actual_eval["avg_loss"].values.min()
    self.assertLess(actual_loss, 1e-8)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _VALUE_NETS))
  def test_fit_two_samples(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=60, batch_size=-1, data_generation="random",
                   learning_rate=0.01, lr_decay=0.99,
                   num_trunks=2, trunk_expl_iterations=0)}
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
                   trunk_expl_iterations=",".join(str(i) for i in range(1, num_iters + 1))
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
