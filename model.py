import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32
PSQT_BUCKETS = 8
LS_BUCKETS = 64

class LayerStacks(nn.Module):
  def __init__(self, count):
    super(LayerStacks, self).__init__()

    self.count = count
    self.l1 = nn.Linear(2 * L1, L2 * count)
    self.l1_fact = nn.Linear(2 * L1, L2, bias=False)
    self.l2 = nn.Linear(L2, L3 * count)
    self.l2_fact = nn.Linear(L2, L3, bias=False)
    self.output = nn.Linear(L3, 1 * count)
    self.output_fact = nn.Linear(L3, 1, bias=False)

    self.idx_offset = None

  def _correct_init_biases(self):
    l1_fact_weight = self.l1_fact.weight
    l2_fact_weight = self.l2_fact.weight
    output_fact_weight = self.output_fact.weight
    with torch.no_grad():
      l1_fact_weight.fill_(0.0)
      l2_fact_weight.fill_(0.0)
      output_fact_weight.fill_(0.0)
    self.l1_fact.weight = nn.Parameter(l1_fact_weight)
    self.l2_fact.weight = nn.Parameter(l2_fact_weight)
    self.output_fact.weight = nn.Parameter(output_fact_weight)
    l1_bias = self.l1.bias
    l2_bias = self.l2.bias
    output_bias = self.output.bias
    with torch.no_grad():
      l1_bias.add_(0.5)
      l2_bias.add_(0.5)
      output_bias.fill_(0.0)
    self.l1.bias = nn.Parameter(l1_bias)
    self.l2.bias = nn.Parameter(l2_bias)
    self.output.bias = nn.Parameter(output_bias)

  def forward(self, x, ls_indices):
    if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
      self.idx_offset = torch.arange(0,x.shape[0]*self.count,self.count, device=ls_indices.device)

    l1s_ = self.l1(x).reshape((-1, self.count, L2))
    l1f_ = self.l1_fact(x)
    # https://stackoverflow.com/questions/55881002/pytorch-tensor-indexing-how-to-gather-rows-by-tensor-containing-indices
    l1c_ = l1s_.view(-1, L2)[ls_indices.flatten() + self.idx_offset]
    l1x_ = torch.clamp(l1c_ + l1f_, 0.0, 1.0)

    l2s_ = self.l2(l1x_).reshape((-1, self.count, L3))
    l2f_ = self.l2_fact(l1x_)
    l2c_ = l2s_.view(-1, L3)[ls_indices.flatten() + self.idx_offset]
    l2x_ = torch.clamp(l2c_ + l2f_, 0.0, 1.0)

    l3s_ = self.output(l2x_).reshape((-1, self.count, 1))
    l3f_ = self.output_fact(l2x_)
    l3c_ = l3s_.view(-1, 1)[ls_indices.flatten() + self.idx_offset]
    l3x_ = l3c_ + l3f_

    return l3x_

  def get_non_output_params(self):
    yield self.l1.bias
    yield self.l1.weight
    yield self.l1_fact.weight
    yield self.l2.bias
    yield self.l2.weight
    yield self.l2_fact.weight

  def get_output_params(self):
    yield self.output.bias
    yield self.output.weight
    yield self.output_fact.weight

  def get_coalesced_layer_stacks(self):
    for i in range(self.count):
      with torch.no_grad():
        l1 = nn.Linear(2*L1, L2)
        l2 = nn.Linear(L2, L3)
        output = nn.Linear(L3, 1)
        l1.weight.data = self.l1.weight[i*L2:(i+1)*L2, :] + self.l1_fact.weight.data
        l1.bias.data = self.l1.bias[i*L2:(i+1)*L2]
        l2.weight.data = self.l2.weight[i*L3:(i+1)*L3, :] + self.l2_fact.weight.data
        l2.bias.data = self.l2.bias[i*L3:(i+1)*L3]
        output.weight.data = self.output.weight[i:(i+1), :] + self.output_fact.weight.data
        output.bias.data = self.output.bias[i:(i+1)]
        yield l1, l2, output


class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1 + PSQT_BUCKETS)
    self.feature_set = feature_set
    self.layer_stacks = LayerStacks(LS_BUCKETS)
    self.lambda_ = lambda_

    self._zero_virtual_feature_weights()
    self._correct_init_biases()
    self._init_psqt()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  Pytorch initializes biases around 0, but we want them
  to be around activation range. Also the bias for the output
  layer should always be 0.
  '''
  def _correct_init_biases(self):
    input_bias = self.input.bias
    with torch.no_grad():
      input_bias.add_(0.5)
      input_bias[L1] = 0.0
    self.input.bias = nn.Parameter(input_bias)

    self.layer_stacks._correct_init_biases()

  def _init_psqt(self):
    input_weights = self.input.weight
    # 1.0 / kPonanzaConstant
    scale = 1 / 600
    with torch.no_grad():
      initial_values = self.feature_set.get_initial_psqt_features()
      assert len(initial_values) == self.feature_set.num_features
      for i in range(8):
        input_weights[L1 + i, :] = torch.FloatTensor(initial_values) * scale
    self.input.weight = nn.Parameter(input_weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in, psqt_indices, layer_stack_indices):
    wp = self.input(w_in)
    bp = self.input(b_in)
    w, wpsqt = torch.split(wp, L1, dim=1)
    b, bpsqt = torch.split(bp, L1, dim=1)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)

    psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
    wpsqt = wpsqt.gather(1, psqt_indices_unsq)
    bpsqt = bpsqt.gather(1, psqt_indices_unsq)
    x = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score, psqt_indices, layer_stack_indices = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    in_scaling = 410
    out_scaling = 361

    q = self(us, them, white, black, psqt_indices, layer_stack_indices) * nnue2score / out_scaling
    t = outcome
    p = (score / in_scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
    entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    # Train with a lower LR on the output layer
    LR = 1e-3
    train_params = [
      {'params' : self.get_specific_layers([self.input]), 'lr' : LR, 'min_weight' : -(2**15-1)/127, 'max_weight' : (2**15-1)/127 },
      {'params' : list(self.get_non_output_params()), 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64 },
      {'params' : list(self.get_output_params()), 'lr' : LR / 10, 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
    ]
    # increasing the eps leads to less saturated nets with a few dead neurons
    optimizer = ranger.Ranger(train_params, betas=(.9, 0.999), eps=1.0e-7)
    # Drop learning rate after 75 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
    return [optimizer], [scheduler]

  def get_specific_layers(self, layers):
    pred = lambda x: x in layers
    return self.get_layers(pred)

  def get_non_output_params(self):
    for p in self.layer_stacks.get_non_output_params():
      yield p

  def get_output_params(self):
    for p in self.layer_stacks.get_output_params():
      yield p

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
