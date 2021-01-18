import torch

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
dim_q = dim_m * 2    # Positioned value and a binary mask.
dim_c = dim_m        # Size of the contextual embedding.
min_seq_len = dim_m  # Minimal subset size of queries.

# Make some random instance which we want to fit.
A = torch.rand((dim_m, dim_m))


def make_data(num_points=dim_b):
  xs = torch.rand((num_points, dim_m))
  ys = A @ xs.T
  # Put batch dim always to the front!
  ys = ys.T                            ; assert ys.shape == (num_points, dim_m)
  return xs, ys


def make_query(x):
  diags = torch.diag_embed(x)
  eyes = torch.eye(dim_m).expand_as(diags)
  return torch.cat((diags, eyes), dim=2)


def make_particle_data(num_points=dim_b, min_seq_len=dim_m):
  x = torch.rand((num_points, dim_m))
  y = A @ x.T
  # Put batch dim always to the front!
  y = y.T                            ; assert y.shape == (num_points, dim_m)

  seq_lens = torch.randint(low=min_seq_len, high=dim_m + 1, size=(dim_b,))
  # We will use only seq_lens, but we align to full sequence for xs.
  xs = make_query(x)                 ; assert xs.shape == (num_points, dim_m, dim_q)
  ys = y.unsqueeze(dim=2)            ; assert ys.shape == (num_points, dim_m, 1)

  # Perform equivariant (pairwise xs and ys) permutation,
  # so we take a random subset according to seq lengths.
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

  for i in range(1000):
    optimizer.zero_grad()
    Y_train_pred = model.forward(X_train)
    train_loss = loss_fn(Y_train, Y_train_pred)
    train_loss.backward()
    optimizer.step()
    if i % 10 == 0:
      with torch.no_grad():
        Y_test_pred = model.forward(X_test)
        test_loss = loss_fn(Y_test, Y_test_pred)
        print(i, train_loss.item(), test_loss.item())


class ParticleModel(torch.nn.Module):
  def __init__(self):
    super(ParticleModel, self).__init__()
    self.fc_context_regression = torch.nn.Linear(dim_c, dim_m, bias=False)
    self.fc_kernel = torch.nn.Linear(dim_q, dim_c, bias=False)
    self.register_parameter("proj", torch.nn.Parameter(torch.randn(dim_c, dim_q)))

  def kernel(self, qs):
    assert qs.shape[1:] == (dim_q, )
    ks = self.fc_kernel.forward(qs)                    ; assert ks.shape[1:]   == (dim_c, )
    return ks

  def pool(self, xs):
    assert xs.shape[1:] == (dim_c, )
    context = torch.sum(xs, dim=0)                     ; assert context.shape  == (dim_c, )
    return context

  def regression(self, xs):
    dim_b = xs.shape[0]                                ; assert xs.shape       == (dim_b, dim_c + dim_q)
    contexts = xs[:, :dim_c]                           ; assert contexts.shape == (dim_b, dim_c)
    queries = xs[:, dim_c:]                            ; assert queries.shape  == (dim_b, dim_q)
    cs = self.fc_context_regression(contexts)          ; assert cs.shape       == (dim_b, dim_m)
    cs = cs.unsqueeze(dim=1)                           ; assert cs.shape       == (dim_b, 1, dim_c)
    qs = queries.unsqueeze(dim=2)                      ; assert qs.shape       == (dim_b, dim_q, 1)
    proj = self.proj.expand(dim_b, *self.proj.shape)   ; assert proj.shape     == (dim_b, dim_c, dim_q)
    ys = cs.bmm(proj).bmm(qs).squeeze(dim=2)           ; assert ys.shape       == (dim_b, 1)
    return ys

  def forward(self, xss, seq_lengths):
    # xss.shape = [b, m, q]
    #   where sequence length is upper bounded to be `m`, but only a subset
    #   `seq_lengths[b]` is valid for given a point `b` in the batch.
    dim_b = xss.shape[0]                                   ; assert xss.shape               == (dim_b, dim_m, dim_q)

    contexts_with_queries = []
    for b, aligned_xs in enumerate(xss.split(1, dim=0)):
      dim_s = seq_lengths[b]
      xs = aligned_xs[:, :dim_s, :].squeeze(dim=0)         ; assert xs.shape                == (dim_s, dim_q)
      context = self.pool(self.kernel(xs))                 ; assert context.shape           == (dim_c, )
      contexts = context.expand(dim_s, -1)                 ; assert contexts.shape          == (dim_s, dim_c)
      context_and_query = torch.cat((contexts, xs), dim=1) ; assert context_and_query.shape == (dim_s, dim_c + dim_q)
      contexts_with_queries.append(context_and_query)
    batches = torch.cat(contexts_with_queries, dim=0)      ; assert batches.shape           == (seq_lengths.sum(), dim_c + dim_q)
    return self.regression(batches)


def particle_model_experiment():
  model = ParticleModel()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
  loss_fn = torch.nn.MSELoss()

  X_train, Y_train, X_train_seq_lens = make_particle_data()
  X_test, Y_test, X_test_seq_lens = make_particle_data()

  try:
    i = 0
    while (True):
      optimizer.zero_grad()
      Y_train_pred = model.forward(X_train, X_train_seq_lens)
      Y_train_subset = subset_of_targets(Y_train, X_train_seq_lens)
      train_loss = loss_fn(Y_train_subset, Y_train_pred)
      train_loss.backward()
      optimizer.step()

      if i % 10 == 0:
        with torch.no_grad():
          Y_test_pred = model.forward(X_test, X_test_seq_lens)
          Y_test_subset = subset_of_targets(Y_test, X_test_seq_lens)
          test_loss = loss_fn(Y_test_subset, Y_test_pred)
          print(i, train_loss.item(), test_loss.item())
      i += 1
  except KeyboardInterrupt:
    print("Interrupted particle_model_experiment!")


if __name__ == '__main__':
  linear_model_experiment()
  particle_model_experiment()
