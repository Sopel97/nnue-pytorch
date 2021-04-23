import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT = 12
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

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

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  bucket = KingBuckets[king_sq]
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + bucket * NUM_PLANES

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PLANES * NUM_SQ // 2)

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
        idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ // 2)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES * NUM_SQ)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA^', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ // 2), ('A', NUM_SQ * NUM_PT)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES - 1

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * (NUM_SQ * NUM_PT)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
