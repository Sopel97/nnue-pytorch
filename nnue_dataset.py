import numpy as np
import ctypes
import torch
import os
import sys
import glob

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self, device_id):
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).clone().cuda(device_id, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone().cuda(non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone().cuda(non_blocking=True)
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2)).transpose()).clone()
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2)).transpose()).clone()
        white = torch.sparse.FloatTensor(iw.long().cuda(device_id, non_blocking=True), torch.ones((self.num_active_white_features), dtype=torch.float32).cuda(device_id, non_blocking=True), (self.size, self.num_inputs))
        black = torch.sparse.FloatTensor(ib.long().cuda(device_id, non_blocking=True), torch.ones((self.num_active_black_features), dtype=torch.float32).cuda(device_id, non_blocking=True), (self.size, self.num_inputs))
        return us, them, white, black, outcome, score

SparseBatchPtr = ctypes.POINTER(SparseBatch)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        device_ids=1):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device_ids = device_ids

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, batch_size, cyclic)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, cyclic)

    def __iter__(self):
        return self

    def __next__(self):
        v_raw = [self.fetch_next(self.stream) for i in self.device_ids]

        if all(v_raw):
            v = [vr.contents.get_tensors(device_id) for vr, device_id in zip(v_raw, self.device_ids)]
            tensors = tuple(v[i][:4] for i in range(len(self.device_ids))), torch.cat(tuple(v[i][4] for i in range(len(self.device_ids)))), torch.cat(tuple(v[i][5] for i in range(len(self.device_ids))))
            for vv in v_raw:
                self.destroy_part(vv)
            return tensors
        else:
            for vv in v_raw:
                if vv:
                    self.destroy_part(vv)
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, device_ids=[0]):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            device_ids)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, device_ids=[0]):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.device_ids = device_ids

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, device_ids=self.device_ids)
