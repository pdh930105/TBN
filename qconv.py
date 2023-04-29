import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary


class TBNConv2d(torch.nn.Conv2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', pim_mode=False, sub_array = 128, adc_mode='none', adc_bits=5):
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

def adc_quantize(inputs, sub_array, adc_mode, adc_bits):
    """
    inputs: input tensor
    sub_array: number of sub-array
    adc_bits: number of adc bits
    """
    if adc_mode =='linear':
        min_val = -sub_array
        max_val = sub_array
        step_size = (max_val - min_val) / (2 ** adc_bits)
        index = torch.clamp(torch.floor((inputs-min_val) / step_size), 0, 2**adc_bits-1) # 0 ~ 2**adc_bits -1 mapping
        y = index * step_size + min_val
    elif adc_mode == 'none':
        y = inputs
    elif adc_mode =='original':
        min_val = inputs.min()
        max_val = inputs.max()
        step_size = (max_val - min_val) / (2 ** adc_bits)
        index = torch.clamp(torch.floor((inputs-min_val) / step_size), 0, 2**adc_bits-1) # 0 ~ 2**adc_bits -1 mapping
        y = index * step_size + min_val
    else:
        raise NotImplementedError
    return y