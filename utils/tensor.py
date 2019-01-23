
import numpy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from utils.types import rset_attr


def mask_from_length(lens, batch_first=False, maxlen=None, dtype=None):
    if not maxlen:
        maxlen = lens.max()

    range = torch.arange(0, maxlen, dtype=torch.int64).to(lens).expand(lens.size(0), -1)
    mask = range > lens[:, None]
    if batch_first:
        mask = mask.t()

    if dtype:
        mask = mask.to(dtype)

    return mask


def dropout(input, training:bool, dropout:nn.Dropout):
    if isinstance(input, PackedSequence):
        data = dropout(input.data)
        output = PackedSequence(data, input.batch_sizes)
    else:
        output = dropout(input)
    return output


def clip_grad_norm_(tensor, max_norm):
    grad_norm = torch.norm(tensor).item()
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm


def to_cuda(tensor_or_module):
    if isinstance(tensor_or_module, numpy.ndarray):
        tensor_or_module = torch.tensor(tensor_or_module)

    if torch.cuda.is_available():
        tensor_or_module = tensor_or_module.cuda()

    return tensor_or_module




def get_parameters(outer, *exclude_from):
    paras = []
    for para in outer.parameters():
        exclude = False
        for item in exclude_from:
            if isinstance(item, nn.Module):
                other = list(item.parameters())
            else:
                other = [item]
            for para2 in other:
                if para is para2:
                    exclude = True
                    break
            if exclude:
                break
        if not exclude:
            paras.append(para)

    return paras


def share_parameters(module, share_to, strict=True):
    assert isinstance(module, nn.Module)
    assert isinstance(share_to, nn.Module)

    is_parameter = lambda name, module: any(name == name_ for name_, _ in module.named_parameters())

    for name, para in module.named_parameters():
        if strict and not is_parameter(name, share_to):
            raise RuntimeError(f'{name} is not an attribute of to_module'
                               f' or it\'s not an instance of nn.Parameter')
        rset_attr(share_to, attr=name, val=para)

def pack_tensors(tensors, padding_value, dtype=None):
    dtype = dtype or tensors[0].dtype
    device=tensors[0].device
    lens = tensors[0].new_tensor([a.shape[0] for a in tensors])
    max_len = torch.max(lens)
    a = torch.zeros((len(tensors), int(max_len.item()))) + padding_value
    a=a.to(device=device,dtype=dtype)
    mask = torch.arange(max_len).to(device=device) < lens[:, None]
    a[mask] = torch.cat(tensors)
    return a
