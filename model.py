import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
from feature_transformer import DoubleFeatureTransformerSlice

# 3 layer fully connected network
L1 = 512
LX = 32

def get_parameters(layers):
  return [p for layer in layers for p in layer.parameters()]

class LayerStacks(nn.Module):
  def __init__(self, count, num_hidden):
    super(LayerStacks, self).__init__()

    self.count = count
    self.num_hidden = num_hidden
    if num_hidden == 0:
      self.layers = [nn.Linear(2 * L1, 1 * count)]
      self.l1_fact = nn.Linear(2 * L1, 1, bias=False)
    else:
      self.layers = [nn.Linear(2 * L1, LX * count)] + [nn.Linear(LX, LX * count) for i in range(num_hidden - 1)] + [nn.Linear(LX, 1 * count)]
      self.l1_fact = nn.Linear(2 * L1, LX, bias=False)
    self.layers = nn.ModuleList(self.layers)

    self.idx_offset = None

    self._init_layers()

  def _init_layers(self):
    for layer in self.layers:
      weight = layer.weight
      bias = layer.bias
      with torch.no_grad():
        for i in range(1, self.count):
          ww = weight.shape[0]//8
          weight[i*ww:(i+1)*ww, :] = weight[:ww, :]
          bias[i*ww:(i+1)*ww] = bias[:ww]
      layer.weight = nn.Parameter(weight)
      layer.bias = nn.Parameter(bias)

    l1_fact_weight = self.l1_fact.weight
    output_bias = self.layers[-1].bias
    with torch.no_grad():
      l1_fact_weight.fill_(0.0)
      output_bias.fill_(0.0)

    self.l1_fact.weight = nn.Parameter(l1_fact_weight)
    self.layers[-1].bias = nn.Parameter(output_bias)

  def forward(self, x, ls_indices):
    # precompute and cache the offset for gathers
    if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
      self.idx_offset = torch.arange(0,x.shape[0]*self.count,self.count, device=ls_indices.device)

    indices = ls_indices.flatten() + self.idx_offset

    l1f_ = self.l1_fact(x)
    for i, layer in enumerate(self.layers):
      ww = layer.weight.shape[0]//8
      x = layer(x).reshape((-1, self.count, ww))
      x = x.view(-1, ww)[indices]
      if i == 0:
        x += l1f_
      if i != self.num_hidden:
        x = torch.clamp(x, 0.0, 1.0)
    return x

  def get_coalesced_layer_stacks(self):
    with torch.no_grad():
      for i in range(self.count):
        ll = []
        for i, layer in enumerate(self.layers):
          ww = layer.weight.shape[0]//8
          mock_layer = nn.Linear(layer.weight.shape[1], ww)
          mock_layer.weight.data = layer.weight[i*ww:(i+1)*ww, :]
          mock_layer.bias.data = layer.bias[i*ww:(i+1)*ww]
          if i == 0:
            mock_layer.weight.data += self.l1_fact.weight.data
          ll.append(mock_layer)
        yield ll


class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0, num_hidden=0):
    super(NNUE, self).__init__()
    self.num_psqt_buckets = feature_set.num_psqt_buckets
    self.num_ls_buckets = feature_set.num_ls_buckets
    self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1 + self.num_psqt_buckets)
    self.feature_set = feature_set
    self.layer_stacks = LayerStacks(self.num_ls_buckets, num_hidden)
    self.lambda_ = lambda_
    self.num_hidden = num_hidden

    self._init_layers()

  '''
  We zero all real feature weights because we want to start the training
  with fewest differences between correlated features.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[a:b, :] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  Pytorch initializes biases around 0, but we want them
  to be around activation range. Also the bias for the output
  layer should always be 0.
  '''
  def _init_layers(self):
    input_bias = self.input.bias
    with torch.no_grad():
      for i in range(8):
        input_bias[L1 + i] = 0.0
    self.input.bias = nn.Parameter(input_bias)

    self._zero_virtual_feature_weights()
    self._init_psqt()

  def _init_psqt(self):
    input_weights = self.input.weight
    # 1.0 / kPonanzaConstant
    scale = 1 / 600
    with torch.no_grad():
      initial_values = self.feature_set.get_initial_psqt_features()
      assert len(initial_values) == self.feature_set.num_features
      for i in range(8):
        input_weights[:, L1 + i] = torch.FloatTensor(initial_values) * scale
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
      padding = weights.new_zeros((new_feature_block.num_virtual_features, weights.shape[1]))
      weights = torch.cat([weights, padding], dim=0)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices):
    wp, bp = self.input(white_indices, white_values, black_indices, black_values)
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
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    in_scaling = 410
    out_scaling = 361

    q = (self(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices) * nnue2score / out_scaling).sigmoid()
    t = outcome
    p = (score / in_scaling).sigmoid()

    loss = (p - q).square().mean()

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
    LR = 8.75e-4
    train_params = [
      {'params' : get_parameters([self.input]), 'lr' : LR },
      {'params' : [self.layer_stacks.l1_fact.weight], 'lr' : LR }
    ]
    if self.num_hidden == 0:
      train_params += [
        {'params' : [self.layer_stacks.layers[0].weight], 'lr' : LR, 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600, 'virtual_params' : self.layer_stacks.l1_fact.weight },
        {'params' : [self.layer_stacks.layers[0].bias], 'lr' : LR }
      ]
    else:
      train_params += [
        {'params' : [self.layer_stacks.layers[0].weight], 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64, 'virtual_params' : self.layer_stacks.l1_fact.weight },
        {'params' : [self.layer_stacks.layers[0].bias], 'lr' : LR }
      ]
      train_params += [
        {'params' : [self.layer_stacks.layers[-1].weight], 'lr' : LR, 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
        {'params' : [self.layer_stacks.layers[-1].bias], 'lr' : LR }
      ]
      for i in range(1, self.num_hidden):
        train_params += [
          {'params' : [self.layer_stacks.layers[i].weight], 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64 },
          {'params' : [self.layer_stacks.layers[i].bias], 'lr' : LR }
        ]

    # increasing the eps leads to less saturated nets with a few dead neurons
    optimizer = ranger.Ranger(train_params, betas=(.9, 0.999), eps=1.0e-7, gc_loc=False, use_gc=False)
    # Drop learning rate after 75 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.992)
    return [optimizer], [scheduler]
