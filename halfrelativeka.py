import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT = 12
NUM_PLANES = NUM_SQ * NUM_PT

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def get_file(sq: int):
  return sq % 8

def get_rank(sq: int):
  return sq // 8

def halfrelativeka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  sq = orient(is_white_pov, sq)
  relative_file = get_file(sq) - get_file(king_sq) + 7
  relative_rank = get_rank(sq) - get_rank(king_sq) + 7
  return (p_idx * 15 * 15) + 15 * relative_file + relative_rank

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PT * 15 * 15)

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfrelativeka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
        idxb = halfrelativeka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfRelativeKA', 0xcc134cb8, OrderedDict([('HalfRelativeKA', NUM_PT * 15 * 15)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PT * 15 * 15)
      for sq, p in board.piece_map().items():
        indices[halfrelativeka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features]
