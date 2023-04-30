import torch
from qconv import TBNConv2d
from qlinear import QLinear
import numpy as np
import torch.nn as nn
import os

class View(torch.nn.Module):
    """
    reshape input tensor to a new tensor with <new_size> by use torch.view()
    size is not include batch_size
    """

    def __init__(self, *new_size: int):
        super(View, self).__init__()
        self.new_size = new_size

    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), *self.new_size)
        return y

    def __repr__(self):
        return 'view{}'.format(self.new_size)


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
        step_size = (max_val - min_val).item() / (2 ** adc_bits)
        index = torch.clamp(torch.floor((inputs-min_val) / step_size), 0, 2**adc_bits-1) # 0 ~ 2**adc_bits -1 mapping
        y = index * step_size + min_val.item()
    else:
        raise NotImplementedError
    return y


def Neural_Sim(self, input, output):
    global model_n
    print("quantize layer ", self.name)
    input_file_name = './layer_record_' + str(model_n) + '/input' + self.name + '.csv'
    weight_file_name = './layer_record_' + str(model_n) + '/weight' + self.name + '.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', 'a')
    f.write(weight_file_name+' '+input_file_name+' ')
    # FP mode, quantize mode check
    weight_q = self.weight_quantizer(self.weight)
    if self.mode == 'tbnconv' or self.mode == 'tbnlinear':
        weight_q = weight_q.sign() # -1, 1
        inputs_q = self.input_quantizer(input[0]).sign() # -1, 0, 1
        length = 2 # 2 bit (-1 = 10, 0 = 00, 1 = 01)
    else:
        pass
    write_matrix_weight(weight_q.cpu().detach().numpy(), weight_file_name)
    if len(self.weight.shape) > 2 : # convolution
        k = self.weight.shape[-1] # kw, kh size
        padding = self.padding
        stride = self.stride
        write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride),None, length, input_file_name)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(),None , length, input_file_name)
    

def write_matrix_weight(input_matrix, filename):
    cout = input_matrix.shape[0] # output channel
    weight_matrix = input_matrix.reshape(cout, -1).transpose()
    np.savetxt(filename ,weight_matrix, delimiter=',', fmt='%10.5f')

def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    """
    input_matrix: im2col ended input matrix ([batch, output_width(output_w*output_h), (kernel_w*kernel_h*input_channel)])

    """
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    output_shape_row = int((input_shape[2] + 2*padding[0] -window_size) / stride[0] + 1)
    output_shape_col = int((input_shape[3] + 2*padding[1] -window_size) / stride[1] + 1)
    item_num = int(output_shape_row * output_shape_col)
    output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    iter = 0
    if (padding[0] != 0):
        input_tmp = np.zeros((input_shape[0], input_shape[1], input_shape[2] + padding[0]*2, input_shape[3] + padding[1] *2))
        input_tmp[:, :, padding[0]: -padding[0], padding[1]: -padding[1]] = input_matrix
        input_matrix = input_tmp
    for i in range(output_shape_row):
        for j in range(output_shape_col):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i*stride[0]:i*stride[0]+window_size,j*stride[1]:j*stride[1]+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix

def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model,wl_weight,wl_activation,model_name,mode): 
    global model_n
    model_n = model_name
    
    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(model_name)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' ')
    
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (TBNConv2d, nn.Conv2d)) or isinstance(layer, (QLinear, nn.Linear)):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list
