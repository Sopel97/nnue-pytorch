import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_KING_BUCKETS = 32
NUM_SQ = 64
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL
NUM_INPUTS = NUM_PLANES_REAL * NUM_KING_BUCKETS * 2

KingBuckets = [
  24, 25, 26, 27, 28, 29, 30, 31,
  16, 17, 18, 19, 20, 21, 22, 23,
  12, 12, 13, 13, 14, 14, 15, 15,
   8,  8,  9,  9, 10, 10, 11, 11,
   4,  4,  5,  5,  6,  6,  7,  7,
   4,  4,  5,  5,  6,  6,  7,  7,
   0,  0,  1,  1,  2,  2,  3,  3,
   0,  0,  1,  1,  2,  2,  3,  3
];

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  king_bucket = KingBuckets[king_sq]
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  if p_idx == 11:
    p_idx -= 1
  offset = 0 if is_white_pov else NUM_INPUTS // 2
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_bucket * NUM_PLANES_REAL + offset

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PLANES_REAL * NUM_KING_BUCKETS * 2)

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
        idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
        values[idxw] = val
        values[idxb] = -val
        values[idxw + NUM_INPUTS // 2] = val
        values[idxb + NUM_INPUTS // 2] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKAv2_b32_s', 0xb3254cb8, OrderedDict([('HalfKAv2_b32_s', NUM_PLANES_REAL * NUM_KING_BUCKETS * 2)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for support during training')

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKAv2_b32_s^', 0xb3254cb8, OrderedDict([('HalfKAv2_b32_s', NUM_PLANES_REAL * NUM_KING_BUCKETS * 2), ('A', NUM_PLANES_VIRTUAL * 2)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    idxx = idx % (NUM_INPUTS // 2)
    idxs = idx // (NUM_INPUTS // 2)

    a_idx = idxx % NUM_PLANES_REAL
    k_idx = idxx // NUM_PLANES_REAL

    if a_idx // NUM_SQ == 10 and k_idx != KingBuckets[a_idx % NUM_SQ]:
      a_idx += NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx + idxs * NUM_PLANES_VIRTUAL]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * (NUM_PLANES_VIRTUAL * 2)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
