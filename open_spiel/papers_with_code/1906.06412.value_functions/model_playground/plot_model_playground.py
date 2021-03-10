import grid_plot.sweep as sweep
import matplotlib.pyplot as plt
import pandas as pd

param_sweep = [
  ("target", ".*"),
  ("model", "(particle_concat_1layer|particle_concat_2layer|particle_dotprod_nopos)"),
]

display_perm = [
  ("target", ),
  ("model",),
]

base_dir = "./experiments/model_playground"
translation_map = {
}

def file_from_params(full_params):
  full_dir = f"{base_dir}"
  for (param_name, mask) in param_sweep:
    full_dir += f"/{param_name}/{full_params[param_name]}"
  return f"{full_dir}/stdout"

def plot_item(ax, file, display_params, full_params):
  try:
    df = pd.read_csv(file, comment="#", skip_blank_lines=True)
    print(file)

    ax.semilogy(df.steps, df.train_loss, label="train_loss")
    ax.semilogy(df.steps, df.test_loss, label="test_loss")
  except Exception as e:
    print(e)
    pass

sweep.display(base_dir, param_sweep, display_perm, translation_map, plot_item)
plt.show()
