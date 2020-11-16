import chess
import halfkp
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
  def __init__(self, feature_set=halfkp, lambda_=1.0, loss_type='entropy'):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.INPUTS, L1)
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    if (loss_type == 'entropy'):
      self.loss_func = self.entropy_loss
    elif (loss_type == 'noob'):
      assert lambda_ == 1.0
      self.loss_func = self.noob_loss
    elif (loss_type == 'mixed'):
      self.loss_func = self.mixed_loss
    else:
      assert False

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    loss = self.loss_func(batch)
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def entropy_loss(self, batch):
    us, them, white, black, outcome, score = batch

    q = self(us, them, white, black)
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

    return loss

  def noob_loss(self, batch):
    VALUE_KNOWN_WIN = 10000

    us, them, white, black, outcome, score = batch

    q = self(us, them, white, black) * 600.0

    loss = (q.clamp(-VALUE_KNOWN_WIN, VALUE_KNOWN_WIN) - score.clamp(-VALUE_KNOWN_WIN, VALUE_KNOWN_WIN))**2 / 4800.0

    return loss.mean()

  def mixed_loss(self, batch):
    VALUE_KNOWN_WIN = 10000

    us, them, white, black, outcome, score = batch

    q = self(us, them, white, black)
    t = outcome
    epsilon = 1e-12
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))

    loss_win = outcome_loss.mean() - outcome_entropy.mean()

    loss_eval = ((q * 600.0).clamp(-VALUE_KNOWN_WIN, VALUE_KNOWN_WIN) - score.clamp(-VALUE_KNOWN_WIN, VALUE_KNOWN_WIN))**2 / 4800.0

    return loss_eval.mean() * self.lambda_ + loss_win * (1.0 - self.lambda_)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
    return optimizer
