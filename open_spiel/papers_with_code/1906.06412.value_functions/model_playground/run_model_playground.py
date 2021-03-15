import sweep

home = "/home/sustr"
backend = "parallel"
backend_params = []
experiment_name = "model_playground"


def param_fn(param, context):
  if param == "num_parviews":
    return list(range(1, 21))


sweep.run_sweep(backend,
                backend_params,
                binary_path=f"python {home}/experiments/model_playground.py",
                base_output_dir=f"{home}/experiments/{experiment_name}",
                base_params=dict(),
                comb_params=[
                  "num_parviews",
                ],
                comb_param_fn=param_fn)
