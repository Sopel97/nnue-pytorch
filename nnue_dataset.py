import numpy as np
import ctypes
import torch
import os
import sys
import glob
import multiprocessing as mp

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class InputTensors:
    def __init__(self, us, them, outcome, score, iw, ib, num_white, num_black, size, max_features):
        self.us = us
        self.them = them
        self.outcome = outcome
        self.score = score
        self.iw = iw
        self.ib = ib
        self.white_ones = torch.ones(num_white, dtype=torch.float32)
        self.black_ones = torch.ones(num_black, dtype=torch.float32)
        self.num_white = num_white
        self.num_black = num_black
        self.size = size
        self.max_features = max_features

    def to_batch(self):
        return (self.us,
            self.them,
            torch._sparse_coo_tensor_unsafe(self.iw, self.white_ones, (self.size, self.max_features)),
            torch._sparse_coo_tensor_unsafe(self.ib, self.black_ones, (self.size, self.max_features)),
            self.outcome,
            self.score
            )

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_longlong)),
        ('black', ctypes.POINTER(ctypes.c_longlong))
    ]

    def get_tensors(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(2, self.num_active_white_features))).clone()
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(2, self.num_active_black_features))).clone()
        return InputTensors(us, them, outcome, score, iw, ib, self.num_active_white_features, self.num_active_black_features, self.size, self.num_inputs)

SparseBatchPtr = ctypes.POINTER(SparseBatch)

# An instance of this class should only be created
# by a new process when using multiprocessing.
# This class's constructor runs indefinitely.
class AsyncTrainingDataProvider:
    def __init__(
        self,
        queue,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None):

        torch.set_num_threads(1)

        self.queue = queue
        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, batch_size, cyclic)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, cyclic)

        self.run()

    def run(self):
        torch.set_num_threads(1)

        while True:
            v = self.fetch_next(self.stream)
            self.queue.put(v.contents.get_tensors())
            self.destroy_part(v)

    def __del__(self):
        self.destroy_stream(self.stream)

# An instance of this class should only be created
# by a new process when using multiprocessing.
# This class's constructor runs indefinitely.
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
        batch_size=None):

        torch.set_num_threads(1)

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, batch_size, cyclic)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, cyclic)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors()
            self.destroy_part(v)
            return tensors
        else:
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

class AsyncSparseBatchProvider(AsyncTrainingDataProvider):
    def __init__(self, queue, feature_set, filename, batch_size, cyclic=True, num_workers=1):
        super(AsyncSparseBatchProvider, self).__init__(
            queue,
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size)

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size)


class AsyncSparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1):
    super(AsyncSparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers

    self.queue = mp.Queue(10)
    self.batch_provider = mp.Process(target=AsyncSparseBatchProvider, args=(self.queue, self.feature_set, self.filename, self.batch_size, self.cyclic, self.num_workers))
    self.batch_provider.start()

  def __iter__(self):
    return self

  def __next__(self):
    return self.queue.get().to_batch()

  def __del__(self):
    self.batch_provider.terminate()
    self.batch_provider.join()


class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.batch_provider = SparseBatchProvider(self.feature_set, self.filename, self.batch_size, self.cyclic, self.num_workers)
    self.iter = iter(self.batch_provider)

  def __iter__(self):
    return self

  def __next__(self):
    return self.iter.__next__().to_batch()
