import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = (NUM_SQ * NUM_PT_REAL)
NUM_PLANES_VIRTUAL = (NUM_SQ * NUM_PT_VIRTUAL)
NUM_INPUTS = 2 * NUM_PLANES_REAL * NUM_SQ

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  if p_idx == 11:
    p_idx -= 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES_REAL

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * NUM_INPUTS

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
        idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
        values[idxw] = val
        values[idxw + NUM_INPUTS // 2] = val
        values[idxb] = -val
        values[idxb + NUM_INPUTS // 2] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKAS2v1', 0x5f134cb8, OrderedDict([('HalfKAS2v1', NUM_INPUTS)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader')

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKAS2v1^', 0x5f134cb8, OrderedDict([('HalfKAS2v1', NUM_INPUTS), ('A', NUM_PLANES_VIRTUAL)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES_REAL
    k_idx = idx // NUM_PLANES_REAL

    if a_idx // NUM_SQ == 10 and k_idx != a_idx % NUM_SQ:
      a_idx += NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * NUM_PLANES_VIRTUAL

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
