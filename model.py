import chess
import halfkp
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set=halfkp, lambda_=1.0):
    super(NNUE, self).__init__()
    num_inputs = feature_set.INPUTS
    self.input = nn.Linear(num_inputs, L1)

    # Zero out the weights/biases for the factorized features
    # Weights stored as [256][41024]
    weights = self.input.weight.narrow(1, 0, feature_set.INPUTS - feature_set.FACTOR_INPUTS)
    weights = torch.cat((weights, torch.zeros(L1, feature_set.FACTOR_INPUTS)), dim=1)
    self.input.weight = nn.Parameter(weights)

    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    self.ft_clipped_count = 0
    self.l1_clipped_count = 0
    self.l2_clipped_count = 0

  def forward(self, us, them, w_in, b_in, do_stats):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_clamped = torch.clamp(l0_, 0.0, 1.0)
    self.ft_clipped_count = (l0_clamped != l0_).sum(dim=(0, 1))
    l0_ = l0_clamped
    l1_ = self.l1(l0_)
    l1_clamped = torch.clamp(l1_, 0.0, 1.0)
    self.l1_clipped_count = (l1_clamped != l1_).sum(dim=(0, 1))
    l1_ = l1_clamped
    l2_ = self.l2(l1_)
    l2_clamped = torch.clamp(l2_, 0.0, 1.0)
    self.l2_clipped_count = (l2_clamped != l2_).sum(dim=(0, 1))
    l2_ = l2_clamped
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type, do_stats):
    us, them, white, black, outcome, score = batch

    q = self(us, them, white, black, do_stats)
    t = outcome
    # Divide score by 600.0 to match the expected NNUE scaling factor
    p = (score / 600.0).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
    entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()

    self.log(loss_type, loss)

    if do_stats:
      with torch.no_grad():
        print('layer_0:avg_abs_weight', self.input.weight.abs().mean(dim=(0, 1)))
        print('layer_0:avg_abs_bias', self.input.bias.abs().mean())
        print('layer_0:clipped_pct', self.ft_clipped_count / (2 * L1 * us.numel()))
        print('layer_1:avg_abs_weight', self.l1.weight.abs().mean(dim=(0, 1)))
        print('layer_1:avg_abs_bias', self.l1.bias.abs().mean())
        print('layer_2:clipped_pct', self.l1_clipped_count / (2 * L2 * us.numel()))
        print('layer_3:avg_abs_weight', self.l2.weight.abs().mean(dim=(0, 1)))
        print('layer_3:avg_abs_bias', self.l2.bias.abs().mean())
        print('layer_4:clipped_pct', self.l2_clipped_count / (2 * L3 * us.numel()))
        print('layer_5:avg_abs_weight', self.output.weight.abs().mean(dim=(0, 1)))
        print('layer_5:avg_abs_bias', self.output.bias.abs().mean())

    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss', batch_idx % 16 == 15)

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss', False)

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss', False)

  def configure_optimizers(self):
    optimizer = ranger.Ranger(self.parameters())
    return optimizer
