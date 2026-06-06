import torch


def masked_mse_loss(output, target, padding_mask):
    valid = ~padding_mask
    if valid.sum() == 0:
        return torch.zeros((), dtype=output.dtype, device=output.device)
    valid = valid.unsqueeze(-1).expand_as(output)
    return torch.mean((output[valid] - target[valid]) ** 2)
