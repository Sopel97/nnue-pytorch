import torch
import time
from torch_sparse import coalesce
from torch_sparse import spmm
from torch_sparse import transpose

INPUT_SIZE = 40960

#weight = torch.rand(INPUT_SIZE, 256, requires_grad=True).cuda()
#weight = torch.rand(INPUT_SIZE, 256, requires_grad=True).cuda()

'''
indices = (torch.rand(32) * INPUT_SIZE).long()

print(indices)
print(weight[:,indices])
print(torch.sum(weight[:,indices], dim=1))
'''

BATCH_SIZE = 8192
ITERS = 100

'''
indices_batch = (torch.rand(BATCH_SIZE, 32) * INPUT_SIZE).long().cuda()

start = time.time()

for i in range(ITERS):
    output = torch.sum(weight[indices_batch,:], dim=1)
    torch.sum(output, dim=0).mean().backward()
    torch.cuda.synchronize()
    print(output.shape)

end = time.time()
print((ITERS * BATCH_SIZE) / (end - start))
'''

'''
indices_batch = torch.stack([torch.repeat_interleave(torch.arange(BATCH_SIZE), 32).long(), (torch.rand(BATCH_SIZE * 32) * INPUT_SIZE).long()]).cuda()
values = torch.full((BATCH_SIZE * 32,), 1).cuda()
print(indices_batch)
index, value = coalesce(indices_batch, values, m=BATCH_SIZE, n=INPUT_SIZE)

start = time.time()

for i in range(ITERS):
    output = spmm(index, value, BATCH_SIZE, INPUT_SIZE, weight)
    torch.sum(output, dim=0).mean().backward()
    torch.cuda.synchronize()
    print(output.shape)

end = time.time()
print((ITERS * BATCH_SIZE) / (end - start))
'''

import cupy as cp

smm = cp.RawKernel(r'''

extern "C" __global__

void smm(const int* indices, int max_indices, const float* matrix, float* output, int stride, int sub_stride) {
    // indices has shape (blockDim.x, 32)
    // matrix has shape (*, stride)
    // output is of shape (blockDim.x, stride)

    int b = blockIdx.x;
    int t = threadIdx.x;

    float* output_row = output + b * stride;
    const int* index_row = indices + b * max_indices;
    for (int index_id = 0; index_id < max_indices; ++index_id) {
        int index = index_row[index_id];
        if (index != -1) {
            const float* input_row = matrix + index * stride;
            for (int s = 0; s < sub_stride; ++s) {
                output_row[t + s] += input_row[t + s];
            }
        }
    }
}

''', 'smm')

smm_backward = cp.RawKernel(r'''

extern "C" __global__

void smm_backward(const int* indices, int max_indices, float* weight_grad, const float* out_grad, int stride, int sub_stride) {
    // indices has shape (blockDim.x, 32)
    // weight_grad has shape (*, stride)
    // output_grad is of shape (blockDim.x, stride)

    int b = blockIdx.x;
    int t = threadIdx.x;

    const float* out_grad_row = out_grad + b * stride;
    const int* index_row = indices + b * max_indices;
    for (int index_id = 0; index_id < max_indices; ++index_id) {
        int index = index_row[index_id];
        if (index != -1) {
            float* weight_grad_row = weight_grad + index * stride;
            for (int s = 0; s < sub_stride; ++s) {
                atomicAdd(weight_grad_row + (t + s), out_grad_row[t + s]);
            }
        }
    }
}

''', 'smm_backward')

# TODO: try doing the above with additional value instead of just 1.0
# TODO: try doing a 2d index in a 1d tensor instead of a 1d index in a 2d tensor.
#       see if it's better to spawn a shit ton of blocks or on for a few indices

BATCH_SIZE = 8192
ITERS = 100

stride = 256
max_indices = 32
weight = cp.random.rand(INPUT_SIZE, stride, dtype=cp.float32)
weight_grad = cp.zeros((INPUT_SIZE, stride), dtype=cp.float32)
indices = (cp.random.rand(BATCH_SIZE, max_indices) * INPUT_SIZE).astype(cp.int32)
output0 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
output1 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
num_threads = 128
start = time.time()

for i in range(ITERS):
    smm((BATCH_SIZE,), (num_threads,), (indices, max_indices, weight, output0, stride, stride//num_threads))  # grid, block and arguments
    smm((BATCH_SIZE,), (num_threads,), (indices, max_indices, weight, output1, stride, stride//num_threads))  # grid, block and arguments
    smm_backward((BATCH_SIZE,), (num_threads,), (indices, max_indices, weight_grad, output0 - output1, stride, stride//num_threads))
    smm_backward((BATCH_SIZE,), (num_threads,), (indices, max_indices, weight_grad, output0 - output1, stride, stride//num_threads))
    print(output0)
    print(output1)
    print(weight_grad)
    print(output0.shape)

end = time.time()
print((ITERS * BATCH_SIZE * 2) / (end - start))
