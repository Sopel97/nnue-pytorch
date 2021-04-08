import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL
MAX_NUM_PIECES = 32

def orient(is_white_pov: bool, sq: int):
  return (56 * (not is_white_pov)) ^ sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  if p_idx == 11:
    p_idx -= 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES_REAL

def get_pc_index(board: chess.Board):
  all_pieces = \
    board.pieces(chess.PAWN, chess.WHITE) | \
    board.pieces(chess.KNIGHT, chess.WHITE) | \
    board.pieces(chess.BISHOP, chess.WHITE) | \
    board.pieces(chess.ROOK, chess.WHITE) | \
    board.pieces(chess.QUEEN, chess.WHITE) | \
    board.pieces(chess.KING, chess.WHITE) | \
    board.pieces(chess.PAWN, chess.BLACK) | \
    board.pieces(chess.KNIGHT, chess.BLACK) | \
    board.pieces(chess.BISHOP, chess.BLACK) | \
    board.pieces(chess.ROOK, chess.BLACK) | \
    board.pieces(chess.QUEEN, chess.BLACK) | \
    board.pieces(chess.KING, chess.BLACK)
  return max(2, min(32, len(all_pieces))) - 1

def halfka_psqts():
  # values copied from stockfish, in stockfish internal units
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * (NUM_PLANES_REAL * NUM_SQ + NUM_SQ * MAX_NUM_PIECES)

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
    super(Features, self).__init__('HalfKA_PC', 0x5da34cb8, OrderedDict([('HalfKA_PC', NUM_PLANES_REAL * NUM_SQ + NUM_SQ * MAX_NUM_PIECES)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES_REAL * NUM_SQ + NUM_SQ * MAX_NUM_PIECES)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      indices[NUM_PLANES_REAL * NUM_SQ + get_pc_index(board)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA_PC^', 0x5da34cb8, OrderedDict([('HalfKA_PC', NUM_PLANES_REAL * NUM_SQ + NUM_SQ * MAX_NUM_PIECES), ('A', NUM_PLANES_VIRTUAL), ('K_PC', MAX_NUM_PIECES)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    if idx >= NUM_PLANES_REAL * NUM_SQ:
      return [idx, self.get_factor_base_feature('K_PC') + (idx - NUM_PLANES_REAL * NUM_SQ) % MAX_NUM_PIECES]

    a_idx = idx % NUM_PLANES_REAL
    k_idx = idx // NUM_PLANES_REAL

    if a_idx // NUM_SQ == 10 and k_idx != a_idx % NUM_SQ:
        a_idx += NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * (NUM_SQ * NUM_PT_VIRTUAL + MAX_NUM_PIECES)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
