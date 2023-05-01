import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary
import torch.nn as nn
import wage_quantizer
import wage_initializer
import math
import numpy as np


class TBNConv2d(torch.nn.Conv2d):
    qw_config = {}
    qa_config = {}
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128, ADCprecision=5, adc_mode = 'original',vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'TBNConv', model = None):
        super(TBNConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        """
        pim_mode: using PIM array results
        subarray: number of sub-arrays
        adc_mode: 'none', 'original' , 'linear', 'non-linear(not implemented)'

        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.weight_quantizer = BinaryWeight(self.qw_config)
        self.wl_weight = 1 # TBN weight always (binary) 1 bit
        self.quantize_output = quantize_output
        self.input_quantizer = Ternary(self.qa_config)
        self.wl_activate = ADCprecision # TBN activation always (ternary) 2 bit
        self.wl_error = wl_error
        self.wl_input = 2
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.adc_mode = adc_mode
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale  = None

    def forward(self, input):
        
        weight_b = self.weight_quantizer(self.weight)
        input_t = self.input_quantizer(input)
        weight_scale = weight_b.abs()
        input_scale = input_t.abs()
    
        if self.inference == 1:
            ori_out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups) 
            out = torch.zeros_like(ori_out)
            if input_t.shape[1] > self.subArray: # input channel > sub-array so divide input to multi sub-array
                num_sub_array = input_t.shape[1] // self.subArray
                for sub_arr in range(num_sub_array):
                    for i in range(weight_b.shape[2]):
                        for j in range(weight_b.shape[3]):
                            mask = torch.zeros_like(weight_b)
                            mask[:, sub_arr*self.subArray:(sub_arr+1)*self.subArray, i, j] = 1
                            scale_input = input_t.max() # future work, support channelwise quantization
                            scale_weight = weight_b.max() # future work, support channelwise quantization
                            inputT = torch.sign(input_t) # ternary bit
                            weightB = torch.sign(weight_b) # binary weight bit
                            weightB = weightB * mask
                            out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            out_adc_result = wage_quantizer.LinearQuantizeOut(out_int, self.ADCprecision,  self.adc_mode, self.subArray)
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
                        out_adc_result = wage_quantizer.LinearQuantizeOut(out_int, self.ADCprecision,  self.adc_mode, self.subArray)
                        out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                        out += out_adc_result

        else:
            out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
        

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,adc_mode = 'original', vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'Qconv', model = None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.adc_mode = adc_mode
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    def forward(self, input):
        
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach() # original weight의 gradient를 scale을 곱한 채로 추적, 하지만 실제 연산에서는 weight를 사용 [self.weight*scale - self.weight*scale + self.weight]
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach() # weight를 quantize한 값으로 추적, 하지만 실제 연산에서는 weight를 사용 [weight1 - weight1 + weight]
        outputOrignal= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups) 

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1: # resnet18, alexnet의 경우 not support?
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
        
            output = torch.zeros_like(outputOrignal) # outputOriginal에 대한 gradient 추적을 안하는 듯?
            del outputOrignal
            cellRange = 2**self.cellBit   # cell precision is 4
        
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2 # dummyP is the weight mapped to Hardware, so we introduce on/off ratio in this value

            for i in range (self.weight.shape[2]):
                for j in range (self.weight.shape[3]):
                    # need to divide to different subArray
                    numSubArray = int(weight.shape[1]/self.subArray) 
                    # cut into different subArrays
                    if numSubArray == 0: # sub array안에 input의 channel이 다 들어갈 시 (한 sub-array에서 1 개의 필터에 대해서 연산 가능)
                        mask = torch.zeros_like(weight)
                        mask[:,:,i,j] = 1
                        if weight.shape[1] == 3: # input channel이 3개일 때
                            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                            X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask # wage quantize된 값을 가지고 weight의 decimal을 계산 (weight => wage quantization으로 변환 => decimal로 변환 (0~ 2**bit-width -1))
                            outputP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range (int(bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask # X_decimal을 cellRange으로 나눈 나머지를 구함 (remainder = X_decimal % cellRange)
                                # retention
                                remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target) # 나머지에 대해 retention을 계산함 (SRAM에서 retention이 발생하는가?)
                                X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask # retention이 발생한 remainder를 빼고 cellRange으로 나눈 몫을 구함 (X_decimal = (X_decimal-remainder)/cellRange)
                                # Now also consider weight has on/off ratio effects
                                # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full (remainderQ.size(),self.vari, device='cuda'))
                                outputPartial= F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial= F.conv2d(input, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k
                                outputP = outputP + outputPartial*scaler*2/(1-1/onoffratio)
                                outputD = outputD + outputDummyPartial*scaler*2/(1-1/onoffratio)
                            outputP = outputP - outputD
                            output = output + outputP
                        else:
                            # quantize input into binary sequence
                            inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                            outputIN = torch.zeros_like(output)
                            for z in range(bitActivation):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ-inputB)/2)
                                outputP = torch.zeros_like(output)
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision, self.adc_mode, self.subArray)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision, self.adc_mode, self.subArray)
                                    scaler = cellRange**k
                                    outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                scalerIN = 2**z
                                outputIN = outputIN + (outputP - outputD)*scalerIN
                            output = output + outputIN/(2**bitActivation)
                    else:
                        # quantize input into binary sequence
                        inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                        outputIN = torch.zeros_like(output)
                        for z in range(bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)
                            outputP = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight)
                                mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputSP = torch.zeros_like(output)
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                # !!! Important !!! the dummy need to be multiplied by a ratio
                                outputSP = outputSP - outputD  # minus dummy column
                                outputP = outputP + outputSP
                            scalerIN = 2**z
                            outputIN = outputIN + outputP*scalerIN
                        output = output + outputIN/(2**bitActivation)
            output = output/(2**bitWeight)   # since weight range was convert from [-1, 1] to [-256, 256]
            
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output/self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        
        return output

if __name__ == '__main__':
    import torch
    inputs = torch.randn(1, 128, 32, 32)
    tbnconv = TBNConv2d(128, 128, 3, 1, 1, adc_mode='original')
    output = tbnconv(inputs)
    tbnconv = TBNConv2d(128, 128, 3, 1, 1, adc_mode='linear', subArray=16)
    output = tbnconv(inputs)
    qconv = QConv2d(128, 128, 3, 1, 1, adc_mode='original', wl_weight=2)
    output = qconv(inputs)
    qconv = QConv2d(128, 128, 1, 1, 1, adc_mode='linear', wl_weight=8,subArray=16)
    output = qconv(inputs)    

