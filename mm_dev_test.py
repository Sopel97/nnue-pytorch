import torch
import time
import cupy as cp

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
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const int32_t        max_active_features,
    const float*   const weight,
    const float*   const bias,
          float*   const output,
    const uint32_t       output_size,
    const uint32_t       output_thread_slice_size
) {
    const uint32_t       block_idx         = blockIdx.x;
    const uint32_t       slice_offset      = threadIdx.x * output_thread_slice_size;

          float*   const output_slice      = output + block_idx * output_size + slice_offset;
    const float*   const bias_slice        = bias                             + slice_offset;

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

feature_transformer_slice_backward = cp.RawKernel(r'''

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

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Weights must be of type float32.

    @param: bias_grad
        The bias gradient vector of shape (output_size,).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Bias values must be of type float32.

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
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
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const int32_t        max_active_features,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad,
    const uint32_t       output_size,
    const uint32_t       output_thread_slice_size
) {
    const uint32_t       block_idx         = blockIdx.x;
    const uint32_t       slice_offset      = threadIdx.x * output_thread_slice_size;

    const float*   const output_grad_slice = output_grad + block_idx * output_size + slice_offset;
          float*   const bias_grad_slice   = bias_grad                             + slice_offset;

    const int32_t* const feature_index_row = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row = feature_values  + block_idx * max_active_features;

    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        atomicAdd(&bias_grad_slice[s], output_grad_slice[s]);
    }

    for (uint32_t k = 0; k < max_active_features; ++k) {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1) {
            float* const weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
            for (int s = 0; s < output_thread_slice_size; ++s) {
                atomicAdd(&weight_grad_slice[s], output_grad_slice[s] * feature_value);
            }
        }
    }
}

''', 'feature_transformer_slice_backward')

def find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x:x[1])
    return divisors[0][0]

INPUT_SIZE = 40960
BATCH_SIZE = 8192
ITERS = 64
stride = 256
max_indices = 32

weight = cp.random.rand(INPUT_SIZE, stride, dtype=cp.float32)
bias = cp.random.rand(stride, dtype=cp.float32)
weight_grad = cp.zeros((INPUT_SIZE, stride), dtype=cp.float32)
bias_grad = cp.zeros((stride,), dtype=cp.float32)
indices0 = (cp.random.rand(BATCH_SIZE, max_indices) * INPUT_SIZE).astype(cp.int32)
indices1 = (cp.random.rand(BATCH_SIZE, max_indices) * INPUT_SIZE).astype(cp.int32)
values0 = cp.random.rand(BATCH_SIZE, max_indices, dtype=cp.float32)
values1 = cp.random.rand(BATCH_SIZE, max_indices, dtype=cp.float32)
output0 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
output1 = cp.zeros((BATCH_SIZE, stride), dtype=cp.float32)
num_threads = find_nearest_divisor(stride, 256)
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

    g = output0 - output1

    feature_transformer_slice_backward(
        (BATCH_SIZE,),
        (num_threads,),
        (
            indices0,
            values0,
            max_indices,
            weight_grad,
            bias_grad,
            g,
            stride,
            stride//num_threads
        )
    )
    feature_transformer_slice_backward(
        (BATCH_SIZE,),
        (num_threads,),
        (
            indices1,
            values1,
            max_indices,
            weight_grad,
            bias_grad,
            g,
            stride,
            stride//num_threads
        )
    )

    print(output0)
    print(output1)
    print(weight_grad)
    print(output0.shape)

end = time.time()
print((ITERS * BATCH_SIZE * 2) / (end - start))
