import model as M
import nnue_dataset
import pytorch_lightning as pl
import halfkp
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset;
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)

def main():
  nnue = M.NNUE(halfkp)

  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  epoch_size = 10000000
  validation_size = 1000
  train_batch_size = 8192
  val_batch_size = 1024
  train_infinite_dataset = nnue_dataset.SparseBatchDataset(halfkp.NAME, 'd8_100000.bin', 8192)
  val_infinite_dataset = nnue_dataset.SparseBatchDataset(halfkp.NAME, 'd10_10000.bin', 1024)
  train_data = DataLoader(FixedNumBatchesDataset(train_infinite_dataset, (epoch_size + train_batch_size - 1) // train_batch_size), batch_size=None, batch_sampler=None)
  val_data = DataLoader(FixedNumBatchesDataset(val_infinite_dataset, (validation_size + val_batch_size - 1) // val_batch_size), batch_size=None, batch_sampler=None)
  tb_logger = pl_loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=tb_logger, gpus=1)
  trainer.fit(nnue, train_data, val_data)

if __name__ == '__main__':
  main()
