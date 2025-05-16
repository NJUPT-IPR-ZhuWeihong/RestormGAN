from torch import nn


class PriorEstimationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(PriorEstimationNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)

        self.norm1 = norm_layer(64)
        self.relu = nn.ReLU(True)

        self.conv2 = Residual(64, 128)

        # self.resnet = nn.ModuleList()
        # for i in range(2):
        #     self.resnet.append(ResnetBlock(128, 'reflect', norm_layer, False, False))
        self.resnet = ResnetBlock(128, 'reflect', norm_layer, False, False)

        # self.hg = nn.ModuleList()
        # for i in range(2):
        #     self.hg.append(HourGlass(128, 3, norm_layer))
        self.hg = HourGlass(128, 3, norm_layer)

        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # for i in range(2):
        #     out = self.resnet[i](out)
        # for i in range(2):
        #     out = self.hg[i](out)
        out = self.resnet(out)
        out = self.hg(out)
        out = self.final_conv(out)
        out = self.norm2(out)
        # out = self.relu(out)
        return out


# def define_prior_Estimation_Network(norm_layer):
# 	prior_Estimation_Network = [nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False),
# 								norm_layer(64),
# 								nn.ReLU(True)]
# 	prior_Estimation_Network += [Residual(64, 128)]
# 	for i in range(2):
# 		prior_Estimation_Network += [ResnetBlock(128, 'reflect', norm_layer, False, False)]
# 	for i in range(2):
# 		prior_Estimation_Network += [HourGlass(128, 3, norm_layer)]
#
# 	prior_Estimation_Network += [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)]
#
# 	prior_Estimation_Network = nn.Sequential(*prior_Estimation_Network)
# 	return prior_Estimation_Network

# Define a hourglass block

class HourGlass(nn.Module):
    def __init__(self, dim, n, norm_layer):
        super(HourGlass, self).__init__()
        self._dim = dim
        self._n = n
        self._norm_layer = norm_layer
        self._init_layers(self._dim, self._n, self._norm_layer)

    def _init_layers(self, dim, n, norm_layer):
        setattr(self, 'res' + str(n) + '_1', Residual(dim, dim))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n - 1, norm_layer)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self, 'res' + str(n) + '_3', Residual(dim, dim))
        setattr(self, 'unsample' + str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n - 1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'unsample' + str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)


class Residual(nn.Module):
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, int(outs / 2), 1),
            nn.BatchNorm2d(int(outs / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outs / 2), int(outs / 2), 3, 1, 1),
            nn.BatchNorm2d(int(outs / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outs / 2), outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, updimension)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            # nn.ReflectionPad2d():镜像填充，宽高各增加2
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        in_chan = dim
        if updimension == True:
            out_chan = in_chan * 2
        else:
            out_chan = dim
        conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
