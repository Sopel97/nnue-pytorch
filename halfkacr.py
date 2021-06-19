import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_KING_BUCKETS = 64 + 3
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL
NUM_INPUTS = NUM_PLANES_REAL * NUM_KING_BUCKETS

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece, cr: int):
  if cr != 0:
    king_bucket = 63 + cr
  else:
    king_bucket = king_sq
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  if p_idx == 11:
    p_idx -= 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_bucket * NUM_PLANES_REAL

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PLANES_REAL * NUM_KING_BUCKETS)

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE), 0)
        idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK), 0)
        values[idxw] = val
        values[idxb] = -val
        if ksq == 4:
          for i in range(3):
            values[idxw + (60 + i) * NUM_PLANES_REAL] = val
            values[idxb + (60 + i) * NUM_PLANES_REAL] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA_CR', 0x5f234de8, OrderedDict([('HalfKA_CR', NUM_INPUTS)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for support during training')

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA_CR^', 0x5f234de8, OrderedDict([('HalfKA_CR', NUM_INPUTS), ('A', NUM_PLANES_VIRTUAL)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES_REAL
    k_idx = idx // NUM_PLANES_REAL

    cr = 0
    if k_idx >= 64:
      cr = k_idx - 63
      k_idx = 4

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
