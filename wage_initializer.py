import torch
import math
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def scale_limit(float_std, bits_W):
    delta = 1 / (2**(bits_W-1))
    limit = 1 - delta
    if bits_W >2 :
        limit_std = limit / math.sqrt(3)
    else:
        limit_std = 0.75 / math.sqrt(3)
    scale = 2 ** np.ceil(np.log2(limit_std/float_std))
    return limit,scale

def wage_init_(tensor, bits_W, factor=2.0, mode="fan_in"):
    if mode != "fan_in":
        raise NotImplementedError("support only wage normal")

    dimensions = tensor.ndimension()
    if dimensions < 2: raise ValueError("tensor at least is 2d")
    elif dimensions == 2: fan_in = tensor.size(1)
    elif dimensions > 2:
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size # in_channel * kw * kh
    # This is a magic number, copied
    float_limit = math.sqrt( 3 * factor / fan_in)
    float_std = math.sqrt(2 / fan_in)
    quant_limit,scale = scale_limit(float_limit, bits_W)
    tensor.data.uniform_(-quant_limit, quant_limit)

    print("fan_in {:6d}, float_limit {:.6f}, float std {:.6f}, quant limit {}, scale {}".format(fan_in, float_limit, float_std, quant_limit, scale))
    return scale
    #import pdb; pdb.set_trace()

