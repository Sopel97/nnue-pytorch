import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT = 6
NUM_PLANES = NUM_SQ * NUM_PT

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def halfkt_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.PieceType):
  p_idx = p - 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def halfkt_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PLANES * NUM_SQ)

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfkt_idx(True, ksq, s, pt)
        values[idxw] = -2*val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKT', 0x4f134cb8, OrderedDict([('HalfKT', NUM_PLANES * NUM_SQ)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_initial_psqt_features(self):
    return halfkt_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKT^', 0x4f134cb8, OrderedDict([('HalfKT', NUM_PLANES * NUM_SQ), ('T', NUM_SQ * NUM_PT)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    t_idx = idx % NUM_PLANES

    return [idx, self.get_factor_base_feature('T') + t_idx]

  def get_initial_psqt_features(self):
    return halfkt_psqts() + [0] * (NUM_SQ * NUM_PT)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
