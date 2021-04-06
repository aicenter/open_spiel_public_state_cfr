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
                     0,0.0246027,0.0694444,0.160184,0.164898
                     1,0.00687305,0.0694444,0.160184,0.164898
                     2,0.00569856,0.0694444,0.0943607,0.0930696
                     3,0.00616575,0.0694444,0.01856,0.0382478
                     4,0.0052576,0.0694444,0.0111825,0.0116441
                     5,0.00483679,0.0694444,0.00541107,0.00583599
                     6,0.00451782,0.0694444,0.0198793,0.00163108
                     7,0.00397735,0.0694444,0.0241905,0.00330082
                     8,0.00385557,0.0694444,0.0233131,0.00326965
                     9,0.00322006,0.0694444,0.023124,0.0030644""",
    particle_vf  ="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.072578,0.0694444,0.0728792,0.0872735
                     1,0.0146236,0.0694444,0.160184,0.164898
                     2,0.00904823,0.0694444,0.0249996,0.0270198
                     3,0.00930322,0.0694444,0.0249996,0.0270198
                     4,0.00826801,0.0694444,0.0249996,0.0270198
                     5,0.00828721,0.0694444,0.0249996,0.0270198
                     6,0.00812901,0.0694444,0.0148534,0.0302647
                     7,0.007835,0.0694444,0.00840256,0.0112863
                     8,0.00821255,0.0694444,0.00279685,0.00977285
                     9,0.00761652,0.0694444,0.00285042,0.0178301""",
)


_REFERENCE_KUHN_DLCFR_EXPL = """trunk_iter,expl
                                1,0.0694444
                                5,0.0371064
                                10,0.0056756"""

_REFERENCE_KUHN_RANDOM_EVAL = dict(
    positional_vf="""loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.029887,0.0694444,0.160184,0.164898
                     1,0.00980688,0.0694444,0.0431042,0.0626128
                     2,0.00594901,0.0694444,0.0120451,0.0244726
                     3,0.00392738,0.0694444,0.0293565,0.0265744
                     4,0.00260408,0.0694444,0.0242413,0.0155411
                     5,0.00242774,0.0694444,0.0235615,0.0117968
                     6,0.00210834,0.0694444,0.0261457,0.012872
                     7,0.00194384,0.0694444,0.0235686,0.0116059
                     8,0.0019213,0.0694444,0.0306296,0.0171096
                     9,0.00166692,0.0694444,0.0302273,0.0157716""",
    particle_vf=  """loop,avg_loss,expl[1],expl[5],expl[10]
                     0,0.07817,0.0694444,0.21111,0.21919
                     1,0.0165519,0.0694444,0.0249996,0.0270198
                     2,0.01235,0.0694444,0.0249996,0.0270198
                     3,0.0117545,0.0694444,0.00876945,0.00655138
                     4,0.00932271,0.0694444,0.00342256,0.000933326
                     5,0.00881761,0.0694444,0.00323402,0.000881905
                     6,0.00776797,0.0694444,0.00167879,0.000457752
                     7,0.00698956,0.0694444,0.0161076,0.00789203
                     8,0.00660263,0.0694444,0.0387975,0.0336498
                     9,0.00556111,0.0694444,0.0401356,0.0320836"""
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
