import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary
from utils import adc_quantize


class TBNLinear(torch.nn.Linear):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_features, out_features, pim_mode=False, sub_array = 128, adc_mode='none', adc_bits=5, name='TBNLinear', bias=True):
        super().__init__(in_features, out_features, bias)
        self.input_quantizer = Ternary(TBNLinear.qa_config)
        self.weight_quantizer = BinaryWeight(TBNLinear.qw_config)
        self.sub_array = sub_array
        self.pim_mode = pim_mode
        self.adc_mode = adc_mode
        self.adc_bits = adc_bits

        self.name = name

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        if self.pim_mode:
            ori_out = F.linear(input_t, weight_b, self.bias)
            out = torch.zeros_like(ori_out)

            if input_t.shape[1] > self.sub_array: # input channel > sub-array so divide input to multi sub-array
                num_sub_array = input_t.shape[1] // self.sub_array
                for sub_arr in range(num_sub_array):
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
            out = F.linear(input_t, weight_b, self.bias)
        return out
