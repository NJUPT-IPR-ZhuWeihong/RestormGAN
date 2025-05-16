    '''
    自己改的基础模块
    '''
class ResidualBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1), relu_type='prelu', norm_type='bn', hg_depth=3, att_name='spar'):
        super(ResidualBlock, self).__init__()

        self.hg_depth = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # 16->16
        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        # 16->32
        self.conv2 = ConvLayer(
            in_channels, out_channels, 3, downsample=True, resample_kernel=resample_kernel, bias=True, activate=True)
        # 16->32
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=True, resample_kernel=resample_kernel, bias=False, activate=False)

        if att_name.lower() == 'spar':
            c_attn = 1
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, out_channels, c_attn, **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (self.att_func(out) + skip) / math.sqrt(2)
        return out

class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment
    --------------------------
    """

    #   depth = 4, c_in = 64, c_out = 1,
    def __init__(self, depth, c_in, c_out,
                 c_mid=64,
                 norm_type='bn',
                 relu_type='prelu',
                 ):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            #   经过注意力层之后再完成一次卷积与激活 channel 64-->1
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                nn.Sigmoid()
            )

    '''
    第一层嵌套：level = 4
    c1 = 64, c2 = 64
    1.构建b1_4与b2_4下采样两个卷积层
    2.嵌套一层level = 3
    3.构建b3_4上采样卷积层

    第二层嵌套：level = 3
    c1 = 64, c2 = 64
    1.构建b1_3与b2_3下采样两个卷积层
    2.嵌套一层level = 2
    3.构建b3_3上采样卷积层
    .
    .
    .
    第四层嵌套：level = 1
    c1 = 64, c2 = 64
    1.构建b1_1与b2_1下采样两个卷积层
    2.构建b2_plus_1普通卷积层
    3.构建b3_1上采样卷积层
    '''

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        #   常规卷积层
        up1 = self._modules['b1_' + str(level)](in_x)
        #   卷积下采样层
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)
        #   卷积上采样层
        up2 = self._modules['b3_' + str(level)](low2)
        #   up1 = {Tensor:(2, 64, 8, 8)} up2 = {Tensor:(2, 64, 8, 8)}
        #   2表示batch_size, 64表示channel
        #   若大小不一，则将up2放大为up1的大小
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        x = self._forward(self.depth, x)
        self.att_map = self.out_block(x)class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)
        x = input_x * self.att_map
        return x

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)

for i in range(self.log_size, 2, -1):
    # i = 9, out_channels = 32
    out_channels = channels[f'{2 ** (i - 1)}']
    # resample_kernel = [1, 3, 3, 1]
    if i > 6:
        self.conv_body_down.append(ResidualBlock(in_channels, out_channels, resample_kernel))
    else:
        self.conv_body_down.append(ResBlock(in_channels, out_channels, resample_kernel))
    # in_channels = 32
    in_channels = out_channels

    # ----------- define face-parsing loss ----------- #
    if 'network_face-parsing' in self.opt:
        self.use_parsing = True
    else:
        self.use_parsing = False

    if self.use_parsing:
        # define face-parsing network
        # gfpgan/archs/face_parsing_arch.py
        self.network_parsing = build_network((self.opt['network_face-parsing']))
        self.network_parsing = self.model_to_device(self.network_parsing)
        self.print_network(self.network_parsing)
        load_path = self.opt['path'].get('pretrain_network_parsing')
        if load_path is not None:
            self.load_network(self.network_parsing, load_path, True, None)
        self.network_parsing.eval()
        for param in self.network_parsing.parameters():
            param.requires_grad = False

            # parsing loss
            if self.use_parsing:
                parsing_weight = self.opt['train']['parsing_weight']
                # get images resized
                out_resize = self.resize_for_parsing(self.output)
                gt_resize = self.resize_for_parsing(self.gt)

                # parsing_gt = self.network_parsing(gt_resize).detach()
                parsing_gt = self.network_parsing(gt_resize)[0].detach()
                parsing_out = self.network_parsing(out_resize)[0]
                l_parsing = self.cri_l1(parsing_out, parsing_gt) * parsing_weight
                l_g_total += l_parsing
                loss_dict['l_parsing'] = l_parsing



