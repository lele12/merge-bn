import numpy as np


def read_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    lines = [x for x in lines if len(x) > 0]
    block = {}
    blocks = []
    for line in lines:
        # print(len(line))
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

def read_weight(weight_file):
    with open(weight_file, 'r') as fp:
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        weights = np.fromfile(fp, dtype = np.float32)
    return header, weights

def get_layer_size(cfg_args):
    layer_args = []
    output_filters = []
    output_channels = 0
    # input_channels = 0
    for inx, blocks in enumerate(cfg_args):

        layer_arg = ()
        if blocks['type'] == 'net':
            output_channels = int(blocks['channels'])
        if blocks['type'] == 'convolutional':
            kernel_size = int(blocks['size'])
            output_channels = int(blocks['filters'])
            try:
                blocks['batch_normalize'] == '1'
                bn_switch = 1
            except:
                bn_switch = 0
            layer_arg = (bn_switch, [kernel_size, output_filters[-1], output_channels])
        if blocks['type'] == 'route':
            route_layers = blocks['layers'].split(',')
            input_channels = 0
            route_layers = [int(x) for x in route_layers]
            for route_layer in route_layers:
                if route_layer < 0:
                    input_channels += int(output_filters[inx + route_layer])
                else:
                    input_channels += int(output_filters[route_layer+1])
            output_channels = input_channels
        if layer_arg != ():
            layer_args.append(layer_arg)
        output_filters.append(output_channels)
    # print(output_filters)
    return layer_args
        
def merge_bn(layer_args, header, weights, out_weight_file):

    fb = open(out_weight_file, 'wb')
    fb.write(header)

    ptr = 0
    for blocks in layer_args:
        bn_switch, layer_arg = blocks
        kernel_size, input_channels, output_channels = layer_arg
        if bn_switch == 1:
            bn_bias = weights[ptr:ptr + output_channels]
            ptr += output_channels
            bn_weights = weights[ptr:ptr + output_channels]
            ptr += output_channels
            bn_mean = weights[ptr:ptr + output_channels]
            ptr += output_channels
            bn_var = weights[ptr:ptr + output_channels]
            ptr += output_channels

            new_bias = bn_bias - (bn_weights / np.sqrt(bn_var + 0.000001)) * bn_mean
            new_bias = np.array(new_bias, dtype = np.float32)

            conv_weights = weights[ptr:ptr + kernel_size*kernel_size*input_channels*output_channels]
            ptr += kernel_size * kernel_size * input_channels * output_channels

            temp = bn_weights / np.sqrt(bn_var + 0.000001)
            temp = temp.reshape(output_channels, 1)
            conv_weights = conv_weights.reshape(output_channels, -1)
            new_weights = conv_weights * temp

            new_weights = new_weights.flatten()
            new_weights = np.array(new_weights, dtype = np.float32)

            fb.write(new_bias)
            fb.write(new_weights)
        if bn_switch == 0:
            new_bias = weights[ptr:ptr + output_channels]
            ptr += output_channels

            new_weights = weights[ptr:ptr + kernel_size*kernel_size*input_channels*output_channels]
            ptr += kernel_size * kernel_size * input_channels * output_channels
            
            fb.write(new_bias)
            fb.write(new_weights)
    fb.close()

def main():
    cfg_name = 'PED_DET_001_middle.cfg'
    weight_file = 'PED_DET_001.weights'
    out_weight_file = 'PED_DET_001_0.weights'
    
    cfg_args = read_cfg(cfg_name)
    layer_args = get_layer_size(cfg_args)
    header, weights = read_weight(weight_file)
    merge_bn(layer_args, header, weights, out_weight_file)

if __name__ == "__main__":
    main()
