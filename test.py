import torch
from torch import nn
import copy
import time

def test(batch_size, device_ids):
    # Seed the rng to have deterministic tests
    torch.manual_seed(12345)

    print('Device ids: ' + str(device_ids))

    # For some reason MSE loss requires very low lr otherwise it blows up
    learning_rate = 0.001

    # Whatever arch
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, 1)
        ).cuda(device_ids[0])

    # Whatever loss
    loss_fn = nn.MSELoss()

    # Whatever optimizer
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    # 0. We have 1 model, N devices, N batches, N outcome tensors
    def step(model, batches, outcomes, device_ids):
        # 1. Replicate the model to all devices
        local_models = [model] + [copy.deepcopy(model).cuda(device_id) for device_id in device_ids[1:]]

        # 2. Make each model do forward on 1 batch -> N x forward
        outs = [m(batch.cuda(device_id, non_blocking=True)) for batch, m, device_id in zip(batches, local_models, device_ids)]

        # 3. Compute loss for each separate forward -> N losses
        losses = [loss_fn(out, outcome.cuda(device_id, non_blocking=True)) for outcome, out, device_id in zip(outcomes, outs, device_ids)]

        # 4. Remove gradients from all parameters. This has to be done before backwards.
        #    This should be better than zero_grad because it doesn't block and makes
        #    the first backward pass assign instead of add - less memory usage
        for m in local_models:
            for param in m.parameters():
                param.grad = None

        # 5. Do backward for each loss separately. This *should* not block
        for loss in losses:
            loss.backward()

        # 6. Non blocking transfer of all gradients to the main device
        #    This shouldn't be that much data for our small net
        for m in local_models[1:]:
            for param in m.parameters():
                param.cuda(device_ids[0], non_blocking=True)

        # 7. Accumualate gradients. We don't want to average them because we're not
        #    splitting the batch, we're taking multiple batches in one step.
        for m in local_models[1:]:
            for main_param, param in zip(model.parameters(), m.parameters()):
                main_param.grad += param.grad

        # 8. Optimizer runs with the accumulated gradients on the main model only.
        optimizer.step()

        # Return loss for diagnostic
        return sum(loss.item() for loss in losses) / len(losses)

    # Random batches and outcomes. We don't care whether they are different for each iteration
    # so we do it once because it's faster.
    # Note that we're scaling the batch size by the number of devices so that
    # it's transparent to the user.
    batches = [torch.rand(batch_size // len(device_ids), 512).cuda(device_id, non_blocking=True) for device_id in device_ids]
    outcomes = [torch.rand(batch_size // len(device_ids), 1).cuda(device_ids[0], non_blocking=True) for device_id in device_ids]

    start_time = time.time()

    losses = []
    # We do a fixed number of batch_size chunks, as the user expects
    for i in range(200):
        losses.append(step(model, batches, outcomes, device_ids))

    # Ensure everything completed before measuring time
    torch.cuda.synchronize()

    end_time = time.time()
    print('{:6.3f} seconds'.format(end_time-start_time))

    print('Loss went from {} to {}'.format(losses[0], losses[-1]))



batch_size = 2**12
# Run twice to prevent initialization from skewing the results
test(batch_size, [0])
test(batch_size, [0, 0])
test(batch_size, [0])
test(batch_size, [0, 0])