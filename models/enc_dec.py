import torch
import torch.nn as nn


class Encoder(nn.Module):
    '''
    Here we build encoder network in a layer by layer way
    '''
    def __init__(self, args, topology):
        super(Encoder, self).__init__()

        self.topologies = [topology]
        if args.rotation == 'euler_angle': 
            self.channel_base = [3]
        elif args.rotation == 'quaternion':
            self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        bias = True
        if args.skeleton_info == 'concat': 
            add_offset = True
        else: 
            add_offset = False

        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)


        for i in range(args.num_layers):
            seq = []


            skeleton_layer
            seq.append(skeleton_layer)
            pool = 
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))



    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat' and offset is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


    
class Decoder():
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2


        if args.skeleton_info == 'concat':
            add_offset = True
        else: 
            add_offset = False

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1:
                bias = False
            else:
                bias = True


    def forward(self,input, offset):
        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat':
                self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # throw the padded rwo for global position
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = input[:, :-1, :]

        return input


class AE(nn.Module):
    def __init__(self, args, topology):
        super(AE, self).__init__()
        self.enc = Encoder(args, topology)
        self.dec = Decoder(args, self.enc)

    def forward(self, input, offset=None):
        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, result


class StaticEncoder(nn.Module):
    def __init__(self, args, edges):
        super(StaticEncoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(args.num_layers):
            neighbor_list = find_neighbor(edges, args.skeleton_dist)
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < args.num_layers - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
            seq.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input: torch.Tensor):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))
        return output
