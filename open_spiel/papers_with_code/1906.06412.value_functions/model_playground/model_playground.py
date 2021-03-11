import torch
import sys

# A test to learn a linear mapping from R^m -> R^m:
# Unlike a usual linear regression, we ask to predict each of the output values
# `j \in (1 ... m)` one by one, instead of all of them at once.
#
# The way how this is done is that we essentially learn the entire original
# regression, but also incorporate an appropriate projection matrix P to get
# the value for `j`th index.
#
# Each input `i \in (1 ... m)` is transformed to indicate the index of the
# input in addition to the input value: we call this _query_ `q`.
#
# From the queries we can make the whole _context_ as a pooling operation (summation) over
# the queries (ran through a kernel).
#
# Then we concatenate the i-th query with the whole context, to run to learn the
# whole regression.
#
# Architecture diagram:
#
# query i -------------------------------------+
#                                              |
# query 1 -> kernel -> --\                    (@)
# query 2 -> kernel -> --(+) -> regression (@) P --> output i
# ...                     ^
# query n -> kernel -> --/
#
# with n <= m

torch.manual_seed(42)

# dim_b = 2            # Test batch size.
dim_b = 1000         # Batch size.
dim_m = 4
dim_f = dim_m        # Features, i.e. binary mask.
dim_q = dim_f + 1    # Positioned value and a binary mask.
dim_c = dim_m        # Size of the contextual embedding.
min_seq_len = dim_m  # Minimal subset size of queries.

# Make some random instance which we want to fit.
A = torch.rand((dim_m, dim_m))
As = torch.rand((10, dim_m, dim_m))

def linear(xs):
  ys = A @ xs.T
  # Put batch dim always to the front!
  return ys.T

def piece_wise_linear(xs):
  ys = torch.empty(dim_b, dim_m, As.shape[0])
  for i, A in enumerate(As.split(1, dim=0)):
    ys[:, :, i] = (A @ xs.T).T.squeeze(dim=2)
  return ys.max(dim=2).values

def make_data(num_points=dim_b, f=linear):
  xs = torch.rand((num_points, dim_m))
  ys = f(xs)                       ; assert ys.shape == (num_points, dim_m)
  return xs, ys


# def make_query(x):
#   diags = torch.diag_embed(x)
#   eyes = torch.eye(dim_m).expand_as(diags)
#   return torch.cat((diags, eyes), dim=2)


def make_query(x):
  dim_b, dim_m = x.shape
  eyes = torch.eye(dim_m).expand(dim_b, -1, -1)
  xs = x.unsqueeze(dim=2)
  return torch.cat((xs, eyes), dim=2)

def make_particle_data(num_points=dim_b, min_seq_len=dim_m, f=linear, permute=True):
  x = torch.rand((num_points, dim_m))
  y = f(x)                           ; assert y.shape == (num_points, dim_m)

  seq_lens = torch.randint(low=min_seq_len, high=min_seq_len + 1, size=(dim_b,))
  # We will use only seq_lens, but we align to full sequence for xs.
  xs = make_query(x)                 ; assert xs.shape == (num_points, dim_m, dim_q)
  ys = y.unsqueeze(dim=2)            ; assert ys.shape == (num_points, dim_m, 1)

  # Perform equivariant (pairwise xs and ys) permutation,
  # so we take a random subset according to seq lengths.
  if permute:
    for i in range(num_points):
      p = torch.randperm(dim_m)
      xs[i, :, :] = xs[i, p, :]
      ys[i, :, :] = ys[i, p, :]

  return xs, ys, seq_lens


def subset_of_targets(yss, seq_lens):
  assert yss.shape[:1] == seq_lens.shape and yss.shape[1:] == (dim_m, 1)
  subsets = torch.cat(
      [ys.squeeze(dim=0)[:seq_lens[i]]
       for i, ys in enumerate(yss.split(1, dim=0))])    ; assert subsets.shape == (seq_lens.sum(), 1)
  return subsets


class LinearModel(torch.nn.Module):
  def __init__(self):
    super(LinearModel, self).__init__()
    self.layer = torch.nn.Linear(dim_m, dim_m)

  def forward(self, x):
    return self.layer.forward(x)


def linear_model_experiment():
  model = LinearModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
  loss_fn = torch.nn.MSELoss()

  X_train, Y_train = make_data()
  X_test, Y_test = make_data()

  for i in range(3000):
    optimizer.zero_grad()
    Y_train_pred = model.forward(X_train)
    train_loss = loss_fn(Y_train, Y_train_pred)
    train_loss.backward()
    optimizer.step()
    if i % 10 == 0:
      with torch.no_grad():
        Y_test_pred = model.forward(X_test)
        test_loss = loss_fn(Y_test, Y_test_pred)
        print(f"{i},{train_loss.item()},{test_loss.item()}")


class PieceWiseLinearModel(torch.nn.Module):
  def __init__(self):
    super(PieceWiseLinearModel, self).__init__()
    self.fc_1 = torch.nn.Linear(dim_m, dim_m * 3)
    self.fc_2 = torch.nn.Linear(3 * dim_m, dim_m * 3)
    self.fc_3 = torch.nn.Linear(3 * dim_m, dim_m)

  def forward(self, x):
    x = torch.relu(self.fc_1.forward(x))
    x = torch.relu(self.fc_2.forward(x))
    return self.fc_3.forward(x)



def pw_linear_model_experiment():
  model = PieceWiseLinearModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
  loss_fn = torch.nn.MSELoss()

  X_train, Y_train = make_data(dim_b, piece_wise_linear)
  X_test, Y_test = make_data(dim_b, piece_wise_linear)

  i = 0
  while(True):
    optimizer.zero_grad()
    Y_train_pred = model.forward(X_train)
    train_loss = loss_fn(Y_train, Y_train_pred)
    train_loss.backward()
    optimizer.step()
    if i % 100 == 0:
      with torch.no_grad():
        Y_test_pred = model.forward(X_test)
        test_loss = loss_fn(Y_test, Y_test_pred)
        print(f"{i},{train_loss.item()},{test_loss.item()}")
    i += 1


class ContextualModel(torch.nn.Module):
  def __init__(self):
    super(ContextualModel, self).__init__()
    self.fc_context_regression = torch.nn.Linear(dim_c, dim_m, bias=False)
    self.fc_kernel = torch.nn.Linear(dim_q - 1, dim_c, bias=False)

  def kernel(self, qs):
    assert qs.shape[1:] == (dim_q, )
    ks = self.fc_kernel.forward(qs[:, 1:])             ; assert ks.shape[1:]   == (dim_c, )
    return ks * qs[:, 0]

  def pool(self, xs):
    assert xs.shape[1:] == (dim_c, )
    context = torch.sum(xs, dim=0)                     ; assert context.shape  == (dim_c, )
    return context

  def regression(self, contexts):
    dim_b = contexts.shape[0]                          ; assert contexts.shape == (dim_b, dim_c)
    ys = self.fc_context_regression(contexts)          ; assert ys.shape       == (dim_b, dim_m)
    return ys

  def forward(self, xss, seq_lengths):
    # xss.shape = [b, m, q]
    #   where sequence length is upper bounded to be `m`, but only a subset
    #   `seq_lengths[b]` is valid for given a point `b` in the batch.
    dim_b = xss.shape[0]                                   ; assert xss.shape               == (dim_b, dim_m, dim_q)

    contexts = []
    for b, aligned_xs in enumerate(xss.split(1, dim=0)):
      dim_s = seq_lengths[b]
      xs = aligned_xs[:, :dim_s, :].squeeze(dim=0)          ; assert xs.shape                == (dim_s, dim_q)
      context = self.pool(self.kernel(xs)).unsqueeze(dim=0) ; assert context.shape           == (1, dim_c)
      contexts.append(context)
    batches = torch.cat(contexts, dim=0)                    ; assert batches.shape           == (dim_b, dim_c)
    return self.regression(batches)



def contextual_model_experiment():
  model = ContextualModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
  loss_fn = torch.nn.MSELoss()

  X_train, Y_train, X_train_seq_lens = make_particle_data(permute=False)
  X_test, Y_test, X_test_seq_lens = make_particle_data(permute=False)

  try:
    i = 0
    while (True):
      optimizer.zero_grad()
      Y_train_pred = model.forward(X_train, X_train_seq_lens)
      train_loss = loss_fn(Y_train.squeeze(dim=2), Y_train_pred)
      train_loss.backward()
      optimizer.step()

      if i % 10 == 0:
        with torch.no_grad():
          Y_test_pred = model.forward(X_test, X_test_seq_lens)
          test_loss = loss_fn(Y_test.squeeze(dim=2), Y_test_pred)
          print(i, train_loss.item(), test_loss.item())
      i += 1
  except KeyboardInterrupt:
    print("Interrupted contextual_model_experiment!")


class ParticleModel(torch.nn.Module):
  def __init__(self):
    super(ParticleModel, self).__init__()
    self.fc_context_regression = torch.nn.Linear(dim_c, dim_m, bias=False)
    self.fc_change_of_basis = torch.nn.Linear(dim_f, dim_c, bias=False)
    self.fc_query1 = torch.nn.Linear(2*dim_m, 2*dim_m, bias=True)
    self.fc_query2 = torch.nn.Linear(2*dim_m, 2*dim_m, bias=True)
    self.fc_query_last = torch.nn.Linear(2*dim_m, 1, bias=True)
    self.register_parameter("proj", torch.nn.Parameter(torch.randn(dim_c, dim_q)))

  def change_of_basis(self, fs):
    assert fs.shape[1:] == (dim_f,)
    bs = self.fc_change_of_basis.forward(fs)                                    ; assert bs.shape[1:] == (dim_c,)
    return bs

  def base_coordinates(self, bs, scales):
    assert bs.shape[1:] == (dim_c,)
    assert scales.shape[1:] == (1, )
    cs = bs * scales
    assert cs.shape[1:] == (dim_c, )
    return cs

  def pool(self, xs):
    assert xs.shape[1:] == (dim_c, )
    context = torch.sum(xs, dim=0)                                              ; assert context.shape  == (dim_c, )
    return context

  def regression(self, context):
    assert context.shape == (dim_c, )
    ys = self.fc_context_regression(context)                                    ; assert ys.shape       == (dim_m, )
    return ys

  def forward(self, xss, seq_lengths):
    # xss.shape = [b, m, q]
    #   where sequence length is upper bounded to be `m`, but only a subset
    #   `seq_lengths[b]` is valid for given a point `b` in the batch.
    dim_b = xss.shape[0]                                                        ; assert xss.shape               == (dim_b, dim_m, dim_q)

    yss = []
    for b, aligned_xs in enumerate(xss.split(1, dim=0)):
      dim_s = seq_lengths[b]
      xs = aligned_xs[:, :dim_s, :].squeeze(dim=0)                              ; assert xs.shape                == (dim_s, dim_q)
      scales = xs[:, 0].unsqueeze(dim=1)                                        ; assert scales.shape            == (dim_s, 1)
      fs = xs[:, 1:]                                                            ; assert fs.shape                == (dim_s, dim_f)
      bs = self.change_of_basis(fs)                                             ; assert bs.shape                == (dim_s, dim_c)
      cs = self.base_coordinates(bs, scales)                                    ; assert cs.shape                == (dim_s, dim_c)
      context = self.pool(cs)                                                   ; assert context.shape           == (dim_c, )
      ys = self.regression(context).expand(dim_s, -1)                           ; assert ys.shape                == (dim_s, dim_m)
      # Use projection of ys to bases bs:
      projs = (ys * bs).sum(dim=1).unsqueeze(dim=1)                             ; assert projs.shape             == (dim_s, 1)
      # # Use concat:
      # cs_and_qs = torch.cat((cs, ks), dim=1)                                      ; assert cs_and_qs.shape == (dim_b, 2*dim_m)
      # ys = torch.relu(self.fc_query1(cs_and_qs))                                  ; assert ys.shape       == (dim_b, 2*dim_m)
      # # Optionally second layer
      # ys = torch.relu(self.fc_query2(ys))                                         ; assert ys.shape       == (dim_b, 2*dim_m)
      # ys = self.fc_query_last(ys)                                                 ; assert ys.shape       == (dim_b, 1)
      yss.append(projs)

    out = torch.cat(yss, dim=0)                                                 ; assert out.shape               == (seq_lengths.sum(), 1)
    return out


def particle_model_experiment(num_particles):
  model = ParticleModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
  lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
      optimizer=optimizer, gamma=0.95)
  loss_fn = torch.nn.MSELoss()

  X_train, Y_train, X_train_seq_lens = make_particle_data(
      num_points=dim_b, min_seq_len=num_particles, permute=True)
  X_test, Y_test, X_test_seq_lens = make_particle_data(
      num_points=dim_b, min_seq_len=num_particles, permute=True)

  if dim_b == 2:
    print(X_train)
    print(Y_train)

  try:
    for i in range(100):
      optimizer.zero_grad()
      Y_train_pred = model.forward(X_train, X_train_seq_lens)
      Y_train_subset = subset_of_targets(Y_train, X_train_seq_lens)
      train_loss = loss_fn(Y_train_subset, Y_train_pred)
      train_loss.backward()
      optimizer.step()
      if i % 100 == 0:
        lr_scheduler.step()

      if i % 10 == 0:
        with torch.no_grad():
          Y_test_pred = model.forward(X_test, X_test_seq_lens)
          Y_test_subset = subset_of_targets(Y_test, X_test_seq_lens)
          test_loss = loss_fn(Y_test_subset, Y_test_pred)
          print(f"{i},{train_loss.item()},{test_loss.item()}")
  except KeyboardInterrupt:
    print("Interrupted particle_model_experiment!")


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_particles", type=int)
  args = parser.parse_args()

  print("steps,train_loss,test_loss")
  # particle_model_experiment(dim_m)
  linear_model_experiment()
  # contextual_model_experiment()
  # pw_linear_model_experiment()
