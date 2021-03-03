import io
import itertools
import subprocess
from functools import reduce
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
from absl.testing import absltest
from absl.testing import parameterized

_REPO = "/home/michal/Code/open_spiel/open_spiel"
_BINARY_PATH = f"{_REPO}/cmake-build-release/papers_with_code/1906.06412" \
               f".value_functions/train_eval_loop"

_BASE_ARGS = dict(
    game_name="kuhn_poker",
    depth=3,
    train_batches=32,
    batch_size=32,
    num_loops=100,
    cfr_oracle_iterations=100,
    trunk_eval_iterations=100,
    num_layers=3,
    num_width=3,
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
  dict(game_name="leduc_poker", depth=5),
  dict(game_name="goofspiel(players=2,num_cards=3,imp_info=True,"
                 "points_order=descending)", depth=1),
]

_DATA_GENERATION = [
  dict(data_generation="dl_cfr"),
  dict(data_generation="random")
]

_REFERENCE_KUHN_DLCFR_EVAL = """
loop,avg_loss,expl[1],expl[5],expl[10]
0,0.105115,0.0694444,0.0509253,0.0542922
1,0.0274438,0.0694444,0.0509253,0.0542922
2,0.00937781,0.0694444,0.0141567,0.0206512
3,0.00830411,0.0694444,0.0171355,0.00288411
4,0.00712066,0.0694444,0.0196713,0.0227157
5,0.00684092,0.0694444,0.011997,0.0136929
6,0.006719,0.0694444,0.00624333,0.00670182
7,0.00628849,0.0694444,0.00682326,0.00374965
8,0.00639248,0.0694444,0.00636514,0.0148459
9,0.00565729,0.0694444,0.00650681,0.00784207
"""

_REFERENCE_KUHN_DLCFR_EXPL = """
trunk_iter,expl
1,0.0694444
5,0.0371064
10,0.0056756
"""

_REFERENCE_KUHN_RANDOM_EVAL = """
loop,avg_loss,expl[1],expl[5],expl[10]
0,0.113564,0.0694444,0.0509253,0.0542922
1,0.0224779,0.0694444,0.044374,0.0525055
2,0.00940264,0.0694444,0.0351531,0.0499907
3,0.0076646,0.0694444,0.0215225,0.0454593
4,0.00558716,0.0694444,0.0169912,0.0384209
5,0.00533079,0.0694444,0.0198618,0.0336662
6,0.00469655,0.0694444,0.0207966,0.025176
7,0.00424157,0.0694444,0.0211623,0.0139415
8,0.00402533,0.0694444,0.0257171,0.0175642
9,0.00372721,0.0694444,0.0258115,0.0204999
"""


def df_from_lines(lines: Union[str, List[str]], **kwargs) -> pd.DataFrame:
  return pd.read_csv(io.StringIO(lines), **kwargs)


def read_evaluation(buffer, tag, comment="# ", single_occurrence=True,
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


def eval_in_time(buffer) -> pd.DataFrame:
  buffer.seek(0)
  return pd.read_csv(buffer, comment="#", skip_blank_lines=True)


def reference_exploitabilities(buffer) -> pd.DataFrame:
  return read_evaluation(buffer, "ref_expl")


def read_experiment_results_from_shell(cmd_args: Dict[str, Any],
                                       *read_evaluations):
  cmd = [_BINARY_PATH] + [f"--{k}={v}" for k, v in cmd_args.items()]
  with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE) as proc:
    output, error = proc.communicate()
    if proc.returncode == 0:
      with io.StringIO(output.decode()) as buffer:
        try:
          results = [evaluator(buffer) for evaluator in read_evaluations]
        except Exception as e:
          buffer.seek(0)
          print("Buffer:", "--------------", sep="\n")
          print("".join(buffer.readlines()))
          raise e
        # buffer.seek(0)
        # print("".join(buffer.readlines()))
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

  def test_kuhn_eval_iters_dlcfr_regression(self, **data_gen_spec):
    args = {**_BASE_ARGS, **data_gen_spec,
            **dict(num_loops=10, num_trunks=100, data_generation="dl_cfr",
                   trunk_eval_iterations="1,5,10")}
    actual_eval, actual_expl = read_experiment_results_from_shell(
        args, eval_in_time, reference_exploitabilities)

    expected_eval = df_from_lines(_REFERENCE_KUHN_DLCFR_EVAL)
    expected_expl = df_from_lines(_REFERENCE_KUHN_DLCFR_EXPL)

    np.testing.assert_array_equal(actual_eval.values, expected_eval.values)
    np.testing.assert_array_equal(actual_expl.values, expected_expl.values)

  def test_kuhn_eval_iters_random_regression(self, **data_gen_spec):
    args = {**_BASE_ARGS, **data_gen_spec,
            **dict(num_loops=10, num_trunks=100, data_generation="random",
                   trunk_eval_iterations="1,5,10")}
    actual_eval, = read_experiment_results_from_shell(args, eval_in_time)
    expected_eval = df_from_lines(_REFERENCE_KUHN_RANDOM_EVAL)
    np.testing.assert_array_equal(actual_eval.values, expected_eval.values)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _DATA_GENERATION))
  def test_fit_one_sample(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=100, num_trunks=1, trunk_eval_iterations=0)}
    df, = read_experiment_results_from_shell(args, eval_in_time)
    actual_loss = df["avg_loss"].values[-1]
    self.assertLess(actual_loss, 1e-8)

  @parameterized.parameters(dict_prod(_TEST_GAMES, _DATA_GENERATION))
  def test_fit_two_samples(self, **game_spec):
    args = {**_BASE_ARGS, **game_spec,
            **dict(num_loops=100, num_trunks=2, trunk_eval_iterations=0)}
    df, = read_experiment_results_from_shell(args, eval_in_time)
    actual_loss = df["avg_loss"].values[-1]
    self.assertLess(actual_loss, 1e-6)


if __name__ == "__main__":
  absltest.main()
