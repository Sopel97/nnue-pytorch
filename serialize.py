import argparse
import features
import math
import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

# hardcoded for now
VERSION = 0x7AF32F20

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash)
    self.int32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.write_feature_transformer_block_sparse(model)
    for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
      self.int32(fc_hash) # FC layers hash
      self.write_fc_layer(l1)
      self.write_fc_layer(l2)
      self.write_fc_layer(output, is_output=True)

  @staticmethod
  def fc_hash(model):
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (M.L1 * 2)

    # Fully connected layers
    layers = [model.layer_stacks.l1, model.layer_stacks.l2, model.layer_stacks.output]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      layer_hash += layer.out_features // model.num_ls_buckets
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if layer.out_features // model.num_ls_buckets != 1:
        # Clipped ReLU hash
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def write_header(self, model, fc_hash):
    self.int32(VERSION) # version
    self.int32(fc_hash ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
    description = b"Features=HalfKA(Friend)[49216->256x2],"
    description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
    description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

  def coalesce_ft_weights(self, model, layer):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
    for i_real, is_virtual in enumerate(indices):
      weight_coalesced[i_real, :] = sum(weight[i_virtual, :] for i_virtual in is_virtual)

    return weight_coalesced

  def write_feature_transformer(self, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    layer_psqt = model.input_psqt
    bias = layer.bias.data[:M.L1]
    bias = bias.mul(127).round().to(torch.int16)
    ascii_hist('ft bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = self.coalesce_ft_weights(model, layer)
    weight_psqt = self.coalesce_ft_weights(model, layer_psqt)
    weight = weight.mul(127).round().to(torch.int16)
    weight_psqt = weight_psqt.mul(9600).round().to(torch.int32) # kPonanzaConstant * FV_SCALE = 9600
    ascii_hist('ft weight:', weight.numpy())
    # weights stored as [41024][256]
    self.buf.extend(weight.flatten().numpy().tobytes())
    self.buf.extend(weight_psqt.flatten().numpy().tobytes())

  def get_nnz_indices_blocks(self, weight, wp):
    all_blocks = weight.reshape((-1, wp.block_width))
    mask_blocks = wp.mask.reshape((-1, wp.block_width))
    is_mask_block_nnz = torch.count_nonzero(mask_blocks, dim=1) > 0
    mask_nnz_indices = torch.nonzero(is_mask_block_nnz).flatten()
    weight_nnz_blocks = all_blocks[mask_nnz_indices]
    return mask_nnz_indices.to(torch.int32), weight_nnz_blocks

  def write_feature_transformer_block_sparse(self, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    layer_psqt = model.input_psqt
    bias = layer.bias.data[:M.L1]
    bias = bias.mul(127).round().to(torch.int16)
    ascii_hist('ft bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = self.coalesce_ft_weights(model, layer)
    weight_psqt = self.coalesce_ft_weights(model, layer_psqt)
    weight = weight.mul(127).round().to(torch.int16)
    weight_psqt = weight_psqt.mul(9600).round().to(torch.int32) # kPonanzaConstant * FV_SCALE = 9600
    ascii_hist('ft weight:', weight.numpy())
    # weights stored as [41024][256]
    nnz_indices, nnz_blocks = self.get_nnz_indices_blocks(weight, model.prune_ft_spec)
    self.buf.extend(nnz_indices.numpy().tobytes())
    self.buf.extend(nnz_blocks.flatten().numpy().tobytes())
    self.buf.extend(weight_psqt.flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)
    ascii_hist('fc bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
    total_elements = torch.numel(weight)
    clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
    print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    ascii_hist('fc weight:', weight.numpy())
    # FC inputs are padded to 32 elements for simd.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.read_header(feature_set, fc_hash)
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)
    for i in range(self.model.num_ls_buckets):
      l1 = nn.Linear(2*M.L1, M.L2)
      l2 = nn.Linear(M.L2, M.L3)
      output = nn.Linear(M.L3, 1)
      self.read_int32(fc_hash) # FC layers hash
      self.read_fc_layer(l1)
      self.read_fc_layer(l2)
      self.read_fc_layer(output, is_output=True)
      self.model.layer_stacks.l1.weight.data[i*M.L2:(i+1)*M.L2, :] = l1.weight
      self.model.layer_stacks.l1.bias.data[i*M.L2:(i+1)*M.L2] = l1.bias
      self.model.layer_stacks.l2.weight.data[i*M.L3:(i+1)*M.L3, :] = l2.weight
      self.model.layer_stacks.l2.bias.data[i*M.L3:(i+1)*M.L3] = l2.bias
      self.model.layer_stacks.output.weight.data[i:(i+1), :] = output.weight
      self.model.layer_stacks.output.bias.data[i:(i+1)] = output.bias

  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(fc_hash ^ feature_set.hash ^ (M.L1*2)) # halfkp network hash
    desc_len = self.read_int32() # Network definition
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer, num_psqt_buckets):
    bias = self.tensor(numpy.int16, [layer.bias.shape[0]-num_psqt_buckets]).divide(127.0)
    layer.bias.data = torch.cat([bias, torch.tensor([0]*num_psqt_buckets)])
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    shape = layer.weight.shape
    weights = self.tensor(numpy.int16, [shape[0], shape[1]-num_psqt_buckets])
    psqtweights = self.tensor(numpy.int32, [shape[0], num_psqt_buckets])
    weights = weights.divide(127.0)
    psqtweights = psqtweights.divide(9600.0)
    layer.weight.data = weights
    layer_psqt.weight.data = psqtweights

  def read_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    # FC inputs are padded to 32 elements for simd.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not args.target.endswith(".nnue"):
      raise Exception("Target file must end with .nnue")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=feature_set)
    nnue.eval()
    writer = NNUEWriter(nnue)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif args.source.endswith(".nnue"):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
    torch.save(reader.model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
