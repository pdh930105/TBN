import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary
import wage_quantizer
import wage_initializer

class TBNLinear(torch.nn.Linear):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_features, out_features, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128, ADCprecision=5, adc_mode = 'original',vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'TBNConv', model = None):
        super().__init__(in_features, out_features, bias)
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
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)


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
                        out_adc_result = wage_quantizer.LinearQuantizeOut(out_int,  self.adc_bits, self.adc_mode, self.sub_array)
                        out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                        out += out_adc_result
            else:
                scale_input = input_t.max() # future work, support channelwise quantization
                scale_weight = weight_b.max() # future work, support channelwise quantization
                inputT = torch.sign(input_t) # ternary bit
                weightB = torch.sign(weight_b) # binary weight bit
                weightB = weightB * mask
                out_int = F.conv2d(inputT, weightB, self.bias, self.stride, self.padding, self.dilation, self.groups)
                out_adc_result = wage_quantizer.LinearQuantizeOut(out_int,  self.adc_bits, self.adc_mode, self.sub_array)
                out_adc_result = out_adc_result * scale_input * scale_weight # rescale
                out += out_adc_result

        else:
            out = F.linear(input_t, weight_b, self.bias)
        return out

class QLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
	             wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5, adc_mode='original',vari=0,t=0,v=0,detect=0,target=0,debug = 0, name ='Qlinear', model = None):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
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

        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1 and self.model=='VGG8':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
            output = torch.zeros_like(outputOrignal)
            cellRange = 2**self.cellBit   # cell precision is 4
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:] = (cellRange-1)*(upper+lower)/2
            # need to divide to different subArray
            numSubArray = int(weight.shape[1]/self.subArray)

            if numSubArray == 0:
                mask = torch.zeros_like(weight)
                mask[:,:] = 1
                # quantize input into binary sequence
                inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                    outputP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
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
                        outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                        outputDummyPartial= F.linear(inputB, dummyP*mask, self.bias)
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
                inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    outputP = torch.zeros_like(outputOrignal)
                    for s in range(numSubArray):
                        mask = torch.zeros_like(weight)
                        mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                        outputSP = torch.zeros_like(outputOrignal)
                        outputD = torch.zeros_like(outputOrignal)
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
                            outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                            outputDummyPartial= F.linear(inputB, dummyP*mask, self.bias)
                            # Add ADC quanization effects here !!!
                            outputPartialQ = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision, self.adc_mode, self.sub_array)
                            outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision, self.adc_mode, self.sub_array)
                            scaler = cellRange**k
                            outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                            outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2**z
                    outputIN = outputIN + outputP*scalerIN
                output = output + outputIN/(2**bitActivation)
            output = output/(2**bitWeight)
        
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision, self.adc_mode, self.sub_array)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output = F.linear(input, weight, self.bias)
        
        output = output/self.scale
        output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        
        return output

