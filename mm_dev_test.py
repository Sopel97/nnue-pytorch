import torch
from torch import nn
from torch import autograd
import time
import cupy as cp
import sys
import math
import os

optimal_num_threads = 256
num_threads_cache = dict()

def get_num_threads(output_size):
    if output_size not in num_threads_cache:
        num_threads_cache[output_size] = find_nearest_divisor(output_size, optimal_num_threads)

    return num_threads_cache[output_size]

def run_kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f

feature_transformer_slice_forward_kernel_cache = dict()

def make_feature_transformer_slice_forward_kernel(max_active_features, output_size):
    '''
        @param: max_active_features
            The maximum number of features that are active
            (non-zero) for a single position. This value determines
            the shape of the inputs.
            This value is of type uint32_t.

        @param: output_size
            The number of outputs. Must match the shape of weights
            and biases.
            This value is of type uint32.
    '''
    num_threads = get_num_threads(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in feature_transformer_slice_forward_kernel_cache:
        kernel = cp.RawKernel(r'''

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
*/
void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output,
    const uint32_t       output_size
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float*   const output_slice        = output + block_idx * output_size + slice_offset;
    const float*   const bias_slice          = bias                             + slice_offset;
          float*         shared_output_slice = shared_output                    + slice_offset;

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    #pragma unroll
    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            const float* const weight_slice = weight + feature_index * output_size + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }}
        }}
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}

'''.format(
                max_active_features=max_active_features,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size),
            'feature_transformer_slice_forward')
        kernel.compile()
        feature_transformer_slice_forward_kernel_cache[key] = run_kernel_with_threads(kernel, (num_threads,))
    return feature_transformer_slice_forward_kernel_cache[key]

feature_transformer_slice_backward_kernel_cache = dict()
def make_feature_transformer_slice_backward_kernel(max_active_features, output_size):
    ''''
        @param: max_active_features
            The maximum number of features that are active
            (non-zero) for a single position. This value determines
            the shape of the inputs.
            This value is of type uint32_t.

        @param: output_size
            The number of outputs. Must match the shape of weights
            and biases.
            This value is of type uint32.
    '''
    num_threads = get_num_threads(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in feature_transformer_slice_backward_kernel_cache:
        kernel = cp.RawKernel(r'''

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
*/
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad,
    const uint32_t       output_size
) {{
    __shared__
          float          shared_output_grad[{output_size}];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * {output_thread_slice_size};

    const float*   const output_grad_slice        = output_grad + block_idx * output_size + slice_offset;
          float*   const bias_grad_slice          = bias_grad                             + slice_offset;
          float*         shared_output_grad_slice = shared_output_grad                    + slice_offset;

    const int32_t* const feature_index_row        = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row        = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_grad_slice[s] = output_grad_slice[s];
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        atomicAdd(&bias_grad_slice[s], shared_output_grad_slice[s]);
    }}

    #pragma unroll
    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            float* const weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
            #pragma unroll
            for (int s = 0; s < {output_thread_slice_size}; ++s)
            {{
                atomicAdd(&weight_grad_slice[s], shared_output_grad_slice[s] * feature_value);
            }}
        }}
    }}
}}

'''.format(
                max_active_features=max_active_features,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size),
            'feature_transformer_slice_backward')
        kernel.compile()
        feature_transformer_slice_backward_kernel_cache[key] = run_kernel_with_threads(kernel, (num_threads,))
    return feature_transformer_slice_backward_kernel_cache[key]

def find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x:x[1])
    return divisors[0][0]

class FeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        kernel = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr(),
                output_size
            )
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_feature_transformer_slice_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
                output_size
            )
        )

        return None, None, weight_grad, bias_grad

def FeatureTransformerSliceFunctionEmulate(feature_indices, feature_values, weight, bias):
    batch_size = feature_indices.shape[0]
    num_inputs = weight.shape[0]
    max_active_features = feature_indices.shape[1]
    inputs = torch.zeros(batch_size, num_inputs, dtype=torch.float32, device=weight.device)
    for i in range(batch_size):
        for j in range(max_active_features):
            feature = feature_indices[i, j]
            value = feature_values[i, j]
            inputs[i, feature] += value

    return torch.mm(inputs, weight) + bias

class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(feature_indices, feature_values, self.weight, self.bias)

"""
BATCH_SIZE = 16
INPUT_SIZE = 10
MAX_ACTIVE_FEATURES = 32
STRIDE = 128

torch.manual_seed(0)
weight0 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
torch.manual_seed(0)
weight1 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
indices = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
values = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

output0 = FeatureTransformerSliceFunctionEmulate(indices.clone(), values.clone(), weight0, bias0)
output1 = FeatureTransformerSliceFunction.apply(indices.clone().cuda(), values.clone().cuda(), weight1.cuda(), bias1.cuda())

print(output0)
print(output1)
output0.sum().backward()
output1.sum().backward()
print(weight0.grad)
print(bias0.grad)
print(weight1.grad)
print(bias1.grad)

sys.exit(0)
"""

INPUT_SIZE = 40960
BATCH_SIZE = 8192
ITERS = 64
STRIDE = 256
MAX_ACTIVE_FEATURES = 32

layer = FeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
layer = FeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32).cuda()
values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()
indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32).cuda()
values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()

output0 = layer(indices0, values0)
output1 = layer(indices1, values1)

device = indices0.device
weight_grad0 = torch.zeros(layer.weight.shape[0], layer.weight.shape[1], dtype=torch.float32, device=device)
bias_grad0 = torch.zeros(STRIDE, dtype=torch.float32, device=device)

start = time.time()

for i in range(ITERS):
    '''
    output0 = layer(indices0, values0)
    output1 = layer(indices1, values1)

    g = ((output0 - output1)**2).mean()
    #g.backward()
    '''

    weight_grad0.zero_()
    bias_grad0.zero_()
    kernel = make_feature_transformer_slice_backward_kernel(MAX_ACTIVE_FEATURES, STRIDE)
    kernel(
        grid=(BATCH_SIZE,),
        args=(
            indices0.data_ptr(),
            values0.data_ptr(),
            weight_grad0.data_ptr(),
            bias_grad0.data_ptr(),
            output0.data_ptr(),
            STRIDE
        )
    )

    weight_grad0.zero_()
    bias_grad0.zero_()
    kernel(
        grid=(BATCH_SIZE,),
        args=(
            indices1.data_ptr(),
            values1.data_ptr(),
            weight_grad0.data_ptr(),
            bias_grad0.data_ptr(),
            output0.data_ptr(),
            STRIDE
        )
    )

    torch.cuda.synchronize()

for param in layer.parameters():
    print(param.grad)

end = time.time()
print((ITERS * BATCH_SIZE) / (end - start))
