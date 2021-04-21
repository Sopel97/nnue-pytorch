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

void smm(const int* indices, const float* values, int max_indices, const float* matrix, float* output, int stride, int sub_stride) {
    // indices has shape (blockDim.x, 32)
    // matrix has shape (*, stride)
    // output is of shape (blockDim.x, stride)

    int b = blockIdx.x;
    int t = threadIdx.x * sub_stride;

    float* output_row = output + b * stride;
    const int* index_row = indices + b * max_indices;
    const float* value_row = values + b * max_indices;
    for (int index_id = 0; index_id < max_indices; ++index_id) {
        int index = index_row[index_id];
        float value = value_row[index_id];
        if (index != -1) {
            const float* input_row = matrix + index * stride;
            for (int s = 0; s < sub_stride; ++s) {
                output_row[t + s] += input_row[t + s] * value;
            }
        }
    }
}

''', 'smm')

smm_backward = cp.RawKernel(r'''

extern "C" __global__

void smm_backward(const int* indices, const float* values, int max_indices, float* weight_grad, const float* out_grad, int stride, int sub_stride) {
    // indices has shape (blockDim.x, 32)
    // weight_grad has shape (*, stride)
    // output_grad is of shape (blockDim.x, stride)

    int b = blockIdx.x;
    int t = threadIdx.x * sub_stride;

    const float* out_grad_row = out_grad + b * stride;
    const int* index_row = indices + b * max_indices;
    const float* value_row = values + b * max_indices;
    for (int index_id = 0; index_id < max_indices; ++index_id) {
        int index = index_row[index_id];
        float value = value_row[index_id];
        if (index != -1) {
            float* weight_grad_row = weight_grad + index * stride;
            for (int s = 0; s < sub_stride; ++s) {
                atomicAdd(&weight_grad_row[t + s], out_grad_row[t + s] * value);
            }
        }
    }
}

''', 'smm_backward')

feature_transformer_slice_forward = cp.RawKernel(r'''

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.

    @param: output_thread_slice_size
        The number of output elements to be processed by a single thread.
        It is assumed that N * output_thread_slice_size == output_size,
        where N is the number of threads with which this kernel is launched.
*/
void feature_transformer_slice_forward(
    const int32_t* feature_indices,
    const float*   feature_values,
    const int32_t  max_active_features,
    const float*   weight,
    const float*   bias,
          float*   output,
    const uint32_t output_size,
    const uint32_t output_thread_slice_size
) {
    const uint32_t       block_idx         = blockIdx.x;
    const uint32_t       slice_offset      = threadIdx.x * output_thread_slice_size;

          float*   const output_slice      = output + block_idx * output_size + slice_offset;
    const float*   const bias_slice        = bias + slice_offset;

    const int32_t* const feature_index_row = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row = feature_values  + block_idx * max_active_features;

    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        output_slice[s] = bias_slice[s];
    }

    for (uint32_t k = 0; k < max_active_features; ++k)
    {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {
            const float* const weight_slice = weight + feature_index * output_size + slice_offset;
            for (uint32_t s = 0; s < output_thread_slice_size; ++s)
            {
                output_slice[s] += weight_slice[s] * feature_value;
            }
        }
    }
}

''', 'feature_transformer_slice_forward')

BATCH_SIZE = 8192
ITERS = 64

stride = 256
max_indices = 32
weight = cp.random.rand(INPUT_SIZE, stride, dtype=cp.float32)
bias = cp.random.rand(stride, dtype=cp.float32)
weight_grad = cp.zeros((INPUT_SIZE, stride), dtype=cp.float32)
indices0 = (cp.random.rand(BATCH_SIZE, max_indices) * INPUT_SIZE).astype(cp.int32)
indices1 = (cp.random.rand(BATCH_SIZE, max_indices) * INPUT_SIZE).astype(cp.int32)
values0 = cp.random.rand(BATCH_SIZE, max_indices, dtype=cp.float32)
values1 = cp.random.rand(BATCH_SIZE, max_indices, dtype=cp.float32)
output0 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
output1 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
num_threads = 256
start = time.time()

for i in range(ITERS):
    feature_transformer_slice_forward(
        (BATCH_SIZE,),
        (num_threads,),
        (
            indices0,
            values0,
            max_indices,
            weight,
            bias,
            output0,
            stride,
            stride//num_threads
        )
    )
    feature_transformer_slice_forward(
        (BATCH_SIZE,),
        (num_threads,),
        (
            indices1,
            values1,
            max_indices,
            weight,
            bias,
            output1,
            stride,
            stride//num_threads
        )
    )
    smm_backward((BATCH_SIZE,), (num_threads,), (indices0, values0, max_indices, weight_grad, output0 - output1, stride, stride//num_threads))
    smm_backward((BATCH_SIZE,), (num_threads,), (indices1, values1, max_indices, weight_grad, output0 - output1, stride, stride//num_threads))
    print(output0)
    print(output1)
    print(weight_grad)
    print(output0.shape)

end = time.time()
print((ITERS * BATCH_SIZE * 2) / (end - start))
