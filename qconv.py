import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary
from utils import adc_quantize
import math
import numpy as np


class TBNConv2d(torch.nn.Conv2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', pim_mode=False, sub_array = 128, adc_mode='none', adc_bits=5, name='TBNConv2d'):
        """
        pim_mode: using PIM array results
        sub_array: number of sub-arrays
        adc_mode: 'none', 'linear', 'non-linear'

        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.input_quantizer = Ternary(TBNConv2d.qa_config)
        self.weight_quantizer = BinaryWeight(TBNConv2d.qw_config)
        self.sub_array = sub_array
        self.pim_mode = pim_mode
        self.adc_mode = adc_mode
        self.adc_bits = adc_bits
        self.name = name

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        if self.pim_mode:
            ori_out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups) 
            out = torch.zeros_like(ori_out)
            if input_t.shape[1] > self.sub_array: # input channel > sub-array so divide input to multi sub-array
                num_sub_array = input_t.shape[1] // self.sub_array
                for sub_arr in range(num_sub_array):
                    for i in range(weight_b.shape[2]):
                        for j in range(weight_b.shape[3]):
                            mask = torch.zeros_like(weight_b)
                            mask[:, sub_arr*self.sub_array:(sub_arr+1)*self.sub_array, i, j] = 1
                            scale_input = input_t.max() # future work, support channelwise quantization
                            scale_weight = weight_b.max() # future work, support channelwise quantization
                            inputT = torch.sign(input_t) # ternary bit
                            weightB = torch.sign(weight_b) # binary weight bit
                            weightB = weightB * mask
                            out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            out_adc_result = adc_quantize(out_int, self.sub_array, self.adc_mode, self.adc_bits)
                            out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                            out += out_adc_result

            else:
                for i in range(weight_b.shape[2]):
                    for j in range(weight_b.shape[3]):
                        mask = torch.zeros_like(weight_b)
                        mask[:,:,i,j] = 1
                        scale_input = input_t.max() # delta * 2
                        scale_weight = weight_b.max() # weight's scale
                        inputT = torch.sign(input_t) # ternary bit
                        weightB = torch.sign(weight_b) # binary weight bit
                        weightB = weightB * mask
                        out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        out_adc_result = adc_quantize(out_int, self.sub_array, self.adc_mode, self.adc_bits)
                        out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                        out += out_adc_result

        else:
            out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QConv2d(torch.nn.Conv2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', input_bits=8, weight_bits=8, pim_mode=False, sub_array = 128, adc_mode='none', adc_bits=5, name='TBNConv2d'):
        """
        pim_mode: using PIM array results
        sub_array: number of sub-arrays
        adc_mode: 'none', 'linear', 'non-linear'

        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.wl_input = input_bits
        self.wl_weight = weight_bits
        self.input_quantizer = wage_quantizer
        self.weight_quantizer = wage_quantizer
        self.sub_array = sub_array
        self.pim_mode = pim_mode
        self.adc_mode = adc_mode
        self.adc_bits = adc_bits
        self.name = name
        self.scale = wage_init_(self.weight, self.wl_weight, factor=1)

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f, self.wl_input)
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach() # original
        weight_q = weight1 + (self.weight_quantizer(self.weight1, self.wl_weight) - weight1).detach() # quantized

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.pim_mode:
            ori_out = F.conv2d(input_t, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups) 
            out = torch.zeros_like(ori_out)
            if input_t.shape[1] > self.sub_array: # input channel > sub-array so divide input to multi sub-array
                num_sub_array = input_t.shape[1] // self.sub_array
                for i in range(weight_q.shape[2]):
                    for j in range(weight_q.shape[3]):
                        inputQ = torch.round((2**bitActivation - 1)* (input_f -0) + 0)
                        OutIn = torch.zeros_like(ori_out)
                        for sub_arr in range(num_sub_array):
                
                            mask = torch.zeros_like(weight_q)
                            mask[:, sub_arr*self.sub_array:(sub_arr+1)*self.sub_array, i, j] = 1
                            X_decimal = torch.round((2**bitWeight -1)/2 * (weight_q +1) +0) * mask
                            outputD = torch.zeros_like(ori_out)
                            weightB = weightB * mask
                            out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            out_adc_result = adc_quantize(out_int, self.sub_array, self.adc_mode, self.adc_bits)
                            out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                            out += out_adc_result

            else:
                for i in range(weight_q.shape[2]):
                    for j in range(weight_q.shape[3]):
                        mask = torch.zeros_like(weight_q)
                        mask[:,:,i,j] = 1
                        scale_input = input_t.max() # delta * 2
                        scale_weight = weight_q.max() # weight's scale
                        inputT = torch.sign(input_t) # ternary bit
                        weightB = torch.sign(weight_q) # binary weight bit
                        weightB = weightB * mask
                        out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        out_adc_result = adc_quantize(out_int, self.sub_array, self.adc_mode, self.adc_bits)
                        out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                        out += out_adc_result

        else:
            out = F.conv2d(input_t, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


def wage_quantizer(inputs, bits):
    scale = 2.**(bits-1)
    assert bits != -1
    if bits==1:
        return torch.sign(inputs)
    if bits > 15:
        return inputs
    return torch.round(inputs*scale)/scale

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
        fan_in = num_input_fmaps * receptive_field_size
    # This is a magic number, copied
    float_limit = math.sqrt( 3 * factor / fan_in)
    float_std = math.sqrt(2 / fan_in)
    quant_limit,scale = scale_limit(float_limit, bits_W)
    tensor.data.uniform_(-quant_limit, quant_limit)

    print("fan_in {:6d}, float_limit {:.6f}, float std {:.6f}, quant limit {}, scale {}".format(fan_in, float_limit, float_std, quant_limit, scale))
    return scale
    #import pdb; pdb.set_trace()

def scale_limit(float_std, bits_W):
    delta = 1 / (2**(bits_W-1))
    limit = 1 - delta
    if bits_W >2 :
        limit_std = limit / math.sqrt(3)
    else:
        limit_std = 0.75 / math.sqrt(3)
    scale = 2 ** np.ceil(np.log2(limit_std/float_std))
    return limit,scale
